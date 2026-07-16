"""
Documents -> SPEC induction: the LLM-facing half of ``template from-docs``.

Per document, three structured-output passes (class inventory -> fields,
batched <=6 classes per call -> relationships) run through an **injected**
``llm_call_fn`` — this package never imports ``llm_clients`` and never
constructs a client, which keeps the path testable and dependency-clean.

``llm_call_fn`` contract (mirrors ``reconcile_graph_aliases``'s injected
callable, alias_reconciler.py): it is invoked with keyword arguments as ::

    llm_call_fn(prompt=<{"system": str, "user": str}>,
                schema_json=<JSON string of the pass schema>,
                context=<str tag naming the pass>) -> dict

and must return the parsed JSON payload. The CLI binds it to
``LiteLLMClient.get_json_response(prompt, schema_json, structured_output=True)``;
malformed-JSON repair happens inside ``get_json_response`` and
**truncation-escalation retries are the caller's concern**: wrap them inside
``llm_call_fn`` (check ``client.last_call_diagnostics["truncated"]`` and retry
once with escalated ``max_tokens``). This module's own retry policy is one
retry per failed pass with a no-progress guard: a retry that returns the
identical invalid payload aborts immediately instead of burning tokens.

Deterministic evidence gates (design §4.3) run before anything enters the
candidate set:

- **verbatim gate** — every identity/field example must be a
  whitespace-normalized substring of the document text the model saw;
- **digit-honesty** — ``*_number``/``*_no``/``ref_*`` identity candidates
  whose surviving examples hold no digit are renamed to ``name``;
- **cardinality gate** — ``documented_max_count`` needs an evidence quote
  with a digit/number word adjacent to the class concept.

Gated per-document candidates then merge deterministically (``merge.py``)
into a loose draft that ``linter.repair_draft`` turns into a valid, linted
:class:`~docling_graph.templategen.spec.TemplateSpec`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.exceptions import ExtractionError

from ..linter import LintReport, repair_draft
from ..naming import to_snake_case
from ..spec import MAX_FIELD_EXAMPLES, SpecGap, TemplateSpec
from .merge import (
    ClassCandidate,
    DocumentCandidates,
    FieldCandidate,
    MergeReport,
    canonical_key,
    merge_documents,
)
from .prompts import (
    PromptDict,
    get_class_inventory_prompt,
    get_fields_prompt,
    get_relationships_prompt,
)
from .schemas import class_inventory_schema, fields_schema, relationships_schema

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BUDGET_CHARS",
    "ELISION_MARKER",
    "MAX_PASS2_BATCH",
    "MIN_DOC_CHARS",
    "DocumentStats",
    "InductionReport",
    "LlmCallFn",
    "PreparedDoc",
    "induce_spec_from_documents",
    "prepare_document_text",
]

LlmCallFn = Callable[..., Any]
"""Injected LLM callable; see the module docstring for the exact contract."""

DEFAULT_BUDGET_CHARS = 24_000
"""Default per-document character budget for the sampler (design §4.1)."""

MAX_PASS2_BATCH = 6
"""Pass 2 sends at most this many classes per call (design §4.2)."""

MIN_DOC_CHARS = 200
"""Documents yielding less text are skipped (scanned-PDF guard, design §9)."""

ELISION_MARKER = "[... elided ...]"
"""Marker inserted where the budget sampler removed content."""

_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_STRUCTURE_LINE_RE = re.compile(r"^\s*(#{1,6}\s|\|.*\|)")
_NUMBER_NAME_RE = re.compile(r"(_number$|_no$|^ref_|_ref$)")
_NUMBER_WORD_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b",
    re.IGNORECASE,
)
_TYPE_ALIASES = {
    "str": "str",
    "string": "str",
    "text": "str",
    "int": "int",
    "integer": "int",
    "long": "int",
    "float": "float",
    "number": "float",
    "double": "float",
    "decimal": "float",
    "bool": "bool",
    "boolean": "bool",
    "date": "date",
    "datetime": "datetime",
    "date-time": "datetime",
}
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# Input preparation (deterministic budget sampler)
# ---------------------------------------------------------------------------


class PreparedDoc(BaseModel):
    """One source document, converted and budget-sampled for induction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    markdown: str
    sampled: bool = False
    cache_path: Path | None = None
    """The cached ``<stem>.document.json`` when a ``cache_dir`` was given and
    the source needed conversion; re-enters the pipeline as DOCLING_DOCUMENT
    input (no re-OCR) for --trial-run/evaluate (design §4.1/§7.2)."""


def _sample_text(text: str, budget_chars: int) -> str:
    """Deterministic head 60% / tail 20% / structure-dense middle 20% sample.

    Structure-dense lines are headings and table rows from the elided middle,
    kept in document order. Elisions are marked with :data:`ELISION_MARKER`
    and a header line reports "sampled X of Y chars" so the model knows
    content was removed and does not invent fields for the gaps. The header
    and markers add a small constant overhead on top of ``budget_chars``.
    """
    total = len(text)
    head_budget = (budget_chars * 6) // 10
    tail_budget = (budget_chars * 2) // 10
    middle_budget = budget_chars - head_budget - tail_budget

    head = text[:head_budget]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[: cut + 1]
    tail = text[total - tail_budget :]
    cut = tail.find("\n")
    if cut >= 0:
        tail = tail[cut + 1 :]

    middle_region = text[len(head) : total - len(tail)]
    structure_lines: list[str] = []
    used = 0
    for line in middle_region.splitlines():
        if not _STRUCTURE_LINE_RE.match(line):
            continue
        cost = len(line) + 1
        if used + cost > middle_budget:
            break
        structure_lines.append(line)
        used += cost

    parts = [head.rstrip("\n"), ELISION_MARKER]
    if structure_lines:
        parts.append("\n".join(structure_lines))
        parts.append(ELISION_MARKER)
    parts.append(tail)
    kept = len(head) + used + len(tail)
    header = (
        f"[docling-graph] sampled {kept} of {total} chars "
        "(head / structure-dense middle / tail; elided regions are marked)"
    )
    return header + "\n" + "\n".join(parts)


def _cache_docling_document(document: Any, source: Path, cache_dir: Path) -> Path | None:
    """Export a converted DoclingDocument as ``<stem>.document.json``.

    Reuses :class:`~docling_graph.core.exporters.docling_exporter.DoclingExporter`
    (imported lazily to keep this module import-light). Best-effort: a cache
    failure only costs the no-re-OCR shortcut, never the induction itself.
    """
    try:
        from docling_graph.core.exporters.docling_exporter import DoclingExporter

        exported = DoclingExporter(output_dir=cache_dir).export_document(
            document,
            base_name=f"{source.stem}.document",
            include_json=True,
            include_markdown=False,
            include_doclang=False,
        )
        raw = exported.get("document_json")
        return Path(raw) if isinstance(raw, str) else None
    except Exception as e:  # pragma: no cover - defensive: cache is an optimization
        logger.warning("Could not cache the DoclingDocument for %s: %s", source, e)
        return None


def prepare_document_text(
    source: str | Path,
    *,
    doc_processor: Any | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    cache_dir: str | Path | None = None,
) -> PreparedDoc:
    """Convert one source to induction text and budget-sample it.

    ``.md``/``.markdown``/``.txt`` files are read as-is; every other input
    goes through ``doc_processor`` — a
    :class:`~docling_graph.core.extractors.document_processor.DocumentProcessor`
    (duck-typed: ``convert_to_docling_doc(source)`` +
    ``extract_full_markdown(document)``), so the user's ``docling_config`` and
    docling-serve settings are honored. The processor is injected rather than
    constructed here to keep this module import-light and testable.

    With ``cache_dir`` set, a converted (non-text) source is additionally
    exported as ``<stem>.document.json`` there and the path is returned on
    ``PreparedDoc.cache_path`` — the pipeline's DOCLING_DOCUMENT input type
    re-enters that file without re-conversion, so trial runs never re-OCR
    (design §4.1/§7.2).

    Raises:
        ValueError: A non-text input was given without a ``doc_processor``.
    """
    path = Path(str(source))
    name = path.name or str(source)
    cache_path: Path | None = None
    if path.suffix.lower() in _TEXT_SUFFIXES:
        markdown = path.read_text(encoding="utf-8")
    else:
        if doc_processor is None:
            raise ValueError(
                f"prepare_document_text: '{source}' needs a DocumentProcessor "
                "(only .md/.markdown/.txt are read directly)"
            )
        document = doc_processor.convert_to_docling_doc(str(source))
        markdown = doc_processor.extract_full_markdown(document)
        if cache_dir is not None:
            cache_path = _cache_docling_document(document, path, Path(cache_dir))
    sampled = len(markdown) > budget_chars
    if sampled:
        markdown = _sample_text(markdown, budget_chars)
    return PreparedDoc(name=name, markdown=markdown, sampled=sampled, cache_path=cache_path)


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class DocumentStats(BaseModel):
    """Per-document induction statistics for the report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sampled: bool = False
    cache_path: Path | None = None
    """Cached ``<stem>.document.json`` (see :class:`PreparedDoc`)."""
    classes_proposed: int = 0
    classes_kept: int = 0
    overflow_classes: list[str] = Field(default_factory=list)
    examples_dropped_by_gate: int = 0
    identity_candidates_dropped: list[str] = Field(default_factory=list)
    digit_honesty_renames: list[str] = Field(default_factory=list)
    cardinality_bounds_dropped: list[str] = Field(default_factory=list)
    edges_dropped: list[str] = Field(default_factory=list)
    retries: int = 0


class InductionReport(BaseModel):
    """Everything ``induce_spec_from_documents`` decided along the way."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentStats] = Field(default_factory=list)
    skipped_sources: list[str] = Field(default_factory=list)
    merge: MergeReport
    lint: LintReport
    gaps: list[SpecGap] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence gates (design §4.3)
# ---------------------------------------------------------------------------


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _verbatim_filter(examples: Sequence[str], document_text: str) -> list[str]:
    """Keep examples that are whitespace-normalized substrings of the text."""
    haystack = _normalize_ws(document_text)
    surviving: list[str] = []
    seen: set[str] = set()
    for example in examples:
        needle = _normalize_ws(example)
        if needle and needle in haystack and needle not in seen:
            seen.add(needle)
            surviving.append(example)
    return surviving


def _cardinality_evidence(class_name: str, quotes: Sequence[str]) -> bool:
    """A quote must hold a digit/number word adjacent to the class concept."""
    words = [w for w in to_snake_case(class_name).split("_") if len(w) >= 3]
    for quote in quotes:
        normalized = _normalize_ws(quote).lower()
        if not _NUMBER_WORD_RE.search(normalized):
            continue
        if not words or any(word in normalized for word in words):
            return True
    return False


def _apply_digit_honesty(
    field_name: str, examples: Sequence[str], class_name: str, stats: DocumentStats
) -> str:
    """Rename number-named identity candidates whose examples hold no digits."""
    if _NUMBER_NAME_RE.search(field_name) and not any(
        any(ch.isdigit() for ch in example) for example in examples
    ):
        stats.digit_honesty_renames.append(f"{class_name}.{field_name}->name")
        return "name"
    return field_name


# ---------------------------------------------------------------------------
# LLM pass plumbing (retry + no-progress guard)
# ---------------------------------------------------------------------------


def _call_llm_pass(
    llm_call_fn: LlmCallFn,
    prompt: PromptDict,
    schema: Mapping[str, Any],
    context: str,
    *,
    payload_key: str,
    stats: DocumentStats,
) -> dict[str, Any]:
    """One pass call with one retry and the no-progress guard.

    A payload is valid when it is a dict carrying a list under
    ``payload_key``. One retry is attempted on an exception or invalid
    payload; a retry returning the **identical** invalid payload aborts
    immediately (no progress), and a second distinct failure raises naming
    the pass.
    """
    schema_json = json.dumps(schema)
    first_invalid: Any = _UNSET
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt:
            stats.retries += 1
        try:
            payload = llm_call_fn(prompt=dict(prompt), schema_json=schema_json, context=context)
        except Exception as e:  # llm_call_fn is injected; anything can surface
            last_error = e
            logger.warning("Induction pass %s failed (attempt %d): %s", context, attempt + 1, e)
            continue
        if isinstance(payload, dict) and isinstance(payload.get(payload_key), list):
            return cast("dict[str, Any]", payload)
        if first_invalid is not _UNSET and payload == first_invalid:
            raise ExtractionError(
                f"Induction pass '{context}' made no progress: the retry returned an "
                f"identical invalid payload (no '{payload_key}' list)",
                details={"context": context},
            )
        first_invalid = payload
        logger.warning(
            "Induction pass %s returned an invalid payload (attempt %d): no '%s' list",
            context,
            attempt + 1,
            payload_key,
        )
    raise ExtractionError(
        f"Induction pass '{context}' failed after one retry",
        details={"context": context},
        cause=last_error,
    )


def _batches(items: Sequence[ClassCandidate], size: int) -> Iterator[list[ClassCandidate]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# ---------------------------------------------------------------------------
# Pass payload parsing
# ---------------------------------------------------------------------------


def _str_items(value: Any, *, cap: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(x) for x in value if isinstance(x, str | int | float) and str(x).strip()]
    return items[:cap] if cap is not None else items


def _parse_type(raw: str) -> tuple[str, str | None]:
    """Return ``(scalar_or_"enum", enum_name)`` from a pass-2 type string."""
    value = raw.strip()
    lowered = value.lower()
    if lowered.startswith("enum:"):
        return "enum", value.split(":", 1)[1].strip() or None
    if lowered == "enum":
        return "enum", None
    return _TYPE_ALIASES.get(lowered, "str"), None


def _parse_pass1(
    payload: dict[str, Any],
    prepared: PreparedDoc,
    stats: DocumentStats,
    max_models: int,
) -> list[ClassCandidate]:
    kept: list[ClassCandidate] = []
    seen: set[str] = set()
    for entry in payload["classes"]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        key = canonical_key(name)
        if not name or not key or key in seen:
            continue
        seen.add(key)
        stats.classes_proposed += 1
        if len(kept) >= max_models:
            stats.overflow_classes.append(name)
            continue
        kind = entry.get("kind")
        candidate = ClassCandidate(
            name=name,
            kind=kind if kind in ("entity", "component") else "component",
            is_root=bool(entry.get("is_root")),
            what_it_is=str(entry.get("what_it_is") or "").strip(),
            confusable_with=str(entry.get("confusable_with") or "").strip(),
            evidence_quotes=_str_items(entry.get("evidence_quotes"), cap=3),
        )
        _gate_identity(entry.get("identity_candidate"), candidate, prepared, stats)
        _gate_cardinality(entry.get("documented_max_count"), candidate, stats)
        kept.append(candidate)
    return kept


def _gate_identity(
    raw: Any, candidate: ClassCandidate, prepared: PreparedDoc, stats: DocumentStats
) -> None:
    if not isinstance(raw, dict):
        return
    field_name = str(raw.get("field") or "").strip()
    if not field_name:
        return
    proposed = _str_items(raw.get("verbatim_examples"), cap=MAX_FIELD_EXAMPLES)
    surviving = _verbatim_filter(proposed, prepared.markdown)
    stats.examples_dropped_by_gate += len(proposed) - len(surviving)
    if not surviving:
        stats.identity_candidates_dropped.append(f"{candidate.name}.{field_name}")
        return
    field_name = _apply_digit_honesty(field_name, surviving, candidate.name, stats)
    candidate.fields.append(
        FieldCandidate(
            name=field_name,
            type="str",
            role="identity",
            description=str(raw.get("why") or "").strip(),
            examples=surviving,
        )
    )
    candidate.identity_survived = True


def _gate_cardinality(raw: Any, candidate: ClassCandidate, stats: DocumentStats) -> None:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        return
    if _cardinality_evidence(candidate.name, candidate.evidence_quotes):
        candidate.documented_max_count = raw
    else:
        stats.cardinality_bounds_dropped.append(candidate.name)


def _apply_pass2(
    payload: dict[str, Any],
    classes_by_key: Mapping[str, ClassCandidate],
    prepared: PreparedDoc,
    stats: DocumentStats,
) -> None:
    for entry in payload["classes"]:
        if not isinstance(entry, dict):
            continue
        cls = classes_by_key.get(canonical_key(str(entry.get("class_name") or "")))
        if cls is None:
            continue
        existing = {canonical_key(f.name): f for f in cls.fields}
        raw_fields = entry.get("fields")
        for raw_field in raw_fields if isinstance(raw_fields, list) else []:
            if isinstance(raw_field, dict):
                _apply_pass2_field(raw_field, cls, existing, prepared, stats)


def _apply_pass2_field(
    raw_field: dict[str, Any],
    cls: ClassCandidate,
    existing: dict[str, FieldCandidate],
    prepared: PreparedDoc,
    stats: DocumentStats,
) -> None:
    name = str(raw_field.get("name") or "").strip()
    fkey = canonical_key(name)
    if not name or not fkey:
        return
    proposed = _str_items(raw_field.get("verbatim_examples"), cap=MAX_FIELD_EXAMPLES)
    surviving = _verbatim_filter(proposed, prepared.markdown)
    stats.examples_dropped_by_gate += len(proposed) - len(surviving)
    scalar, enum_name = _parse_type(str(raw_field.get("type") or "str"))
    description = str(raw_field.get("description") or "").strip()

    target = existing.get(fkey)
    if target is not None:
        # Same field re-proposed (typically the pass-1 identity candidate):
        # enrich, never displace the identity role or gated examples.
        merged = list(target.examples)
        seen = {_normalize_ws(x) for x in merged}
        for example in surviving:
            if _normalize_ws(example) not in seen:
                merged.append(example)
        target.examples = merged[:MAX_FIELD_EXAMPLES]
        if not target.description:
            target.description = description
        if target.role == "identity":
            if scalar != "enum":
                target.type = scalar
        else:
            target.type = target.type if scalar == "enum" else scalar
        return

    enum_members = _str_items(raw_field.get("enum_members"))
    synonyms: dict[str, list[str]] = {}
    raw_synonyms = raw_field.get("enum_synonyms")
    for pair in raw_synonyms if isinstance(raw_synonyms, list) else []:
        if isinstance(pair, dict):
            member = str(pair.get("member") or "").strip()
            phrases = _str_items(pair.get("phrases"))
            if member and phrases:
                synonyms.setdefault(member, []).extend(phrases)
    is_enum = scalar == "enum" or bool(enum_members)
    field = FieldCandidate(
        name=name,
        type="str" if is_enum else scalar,
        is_list=bool(raw_field.get("is_list")),
        description=description,
        examples=surviving,
        role="property",
        enum_name=enum_name if is_enum else None,
        enum_members=enum_members if is_enum else [],
        enum_synonyms=synonyms if is_enum else {},
        unit_varies=bool(raw_field.get("unit_varies")),
    )
    existing[fkey] = field
    cls.fields.append(field)


def _apply_pass3(
    payload: dict[str, Any],
    classes_by_key: Mapping[str, ClassCandidate],
    stats: DocumentStats,
) -> None:
    for raw_edge in payload["edges"]:
        if not isinstance(raw_edge, dict):
            continue
        source_name = str(raw_edge.get("source") or "").strip()
        target_name = str(raw_edge.get("target") or "").strip()
        source = classes_by_key.get(canonical_key(source_name))
        target = classes_by_key.get(canonical_key(target_name))
        if source is None or target is None:
            stats.edges_dropped.append(f"{source_name or '?'} -> {target_name or '?'}")
            continue
        field_name = str(raw_edge.get("field_name") or "").strip() or to_snake_case(target.name)
        fkey = canonical_key(field_name)
        evidence = _str_items(raw_edge.get("evidence"), cap=2)
        is_list = bool(raw_edge.get("is_list"))
        reference = raw_edge.get("target_described_fully_here") is False
        label = str(raw_edge.get("label") or "").strip() or None

        existing = next((f for f in source.fields if canonical_key(f.name) == fkey), None)
        if existing is not None:
            if existing.role == "identity":
                stats.edges_dropped.append(
                    f"{source.name}.{field_name} -> {target.name} (collides with identity)"
                )
                continue
            existing.role = "edge"
            existing.type = target.name
            existing.edge_label = label
            existing.reference = reference
            existing.is_list = existing.is_list or is_list
            existing.examples = []
            existing.enum_name = None
            existing.enum_members = []
            existing.enum_synonyms = {}
            existing.evidence = (existing.evidence + evidence)[:5]
            continue
        source.fields.append(
            FieldCandidate(
                name=field_name,
                type=target.name,
                is_list=is_list,
                role="edge",
                edge_label=label,
                reference=reference,
                evidence=evidence,
            )
        )


# ---------------------------------------------------------------------------
# Per-document induction
# ---------------------------------------------------------------------------


def _induce_document(
    prepared: PreparedDoc,
    llm_call_fn: LlmCallFn,
    *,
    root_name: str | None,
    max_models: int,
    stats: DocumentStats,
) -> DocumentCandidates:
    payload = _call_llm_pass(
        llm_call_fn,
        get_class_inventory_prompt(
            prepared.markdown,
            doc_name=prepared.name,
            max_models=max_models,
            root_hint=root_name,
        ),
        class_inventory_schema(),
        f"templategen_pass1_classes:{prepared.name}",
        payload_key="classes",
        stats=stats,
    )
    classes = _parse_pass1(payload, prepared, stats, max_models)
    if not classes:
        logger.warning("No usable classes induced from %s", prepared.name)
        return DocumentCandidates(name=prepared.name, classes=[])
    classes_by_key = {canonical_key(cls.name): cls for cls in classes}

    for batch_index, batch in enumerate(_batches(classes, MAX_PASS2_BATCH)):
        payload = _call_llm_pass(
            llm_call_fn,
            get_fields_prompt(
                prepared.markdown,
                doc_name=prepared.name,
                classes=[(cls.name, cls.what_it_is) for cls in batch],
            ),
            fields_schema(),
            f"templategen_pass2_fields:{prepared.name}:batch{batch_index}",
            payload_key="classes",
            stats=stats,
        )
        _apply_pass2(payload, classes_by_key, prepared, stats)

    payload = _call_llm_pass(
        llm_call_fn,
        get_relationships_prompt(
            prepared.markdown,
            doc_name=prepared.name,
            classes=[cls.name for cls in classes],
        ),
        relationships_schema(),
        f"templategen_pass3_edges:{prepared.name}",
        payload_key="edges",
        stats=stats,
    )
    _apply_pass3(payload, classes_by_key, stats)
    stats.classes_kept = len(classes)
    return DocumentCandidates(name=prepared.name, classes=classes)


def _dedupe_gaps(gaps: Sequence[SpecGap]) -> list[SpecGap]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[SpecGap] = []
    for gap in gaps:
        key = (gap.model, gap.field, gap.kind)
        if key not in seen:
            seen.add(key)
            unique.append(gap)
    return unique


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def induce_spec_from_documents(
    sources: Sequence[str | Path],
    llm_call_fn: LlmCallFn,
    *,
    root_name: str | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    max_models: int = 30,
    strict: bool = False,
    doc_processor: Any | None = None,
    max_enum_members: int | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[TemplateSpec, InductionReport]:
    """Induce a validated :class:`TemplateSpec` from example documents.

    Per document: pass 1 (class inventory) -> evidence gates -> pass 2
    (fields, batched <= :data:`MAX_PASS2_BATCH` classes per call) -> pass 3
    (relationships). Per-document candidates merge deterministically
    (``merge.merge_documents``), the draft goes through
    ``linter.repair_draft`` (which doubles documented ``max_instances``
    exactly once and derives missing edge labels), and the repaired spec is
    returned with a full :class:`InductionReport`.

    Args:
        sources: Document paths; ``.md``/``.markdown``/``.txt`` are read
            directly, everything else needs ``doc_processor``.
        llm_call_fn: The injected LLM callable (module docstring has the
            exact contract, including the truncation-retry recommendation).
        root_name: Optional root class name (the CLI's ``--name``); overrides
            the ``is_root`` vote and renames the elected root.
        budget_chars: Per-document sampler budget (design §4.1).
        max_models: Cap on induced classes per document and after the merge;
            overflow is reported, never silently truncated.
        strict: Passed to ``repair_draft`` — any repair raises
            ``TemplateLintError`` instead of being applied silently.
        doc_processor: Optional ``DocumentProcessor`` for non-text sources.
        max_enum_members: Enum unions wider than this demote to ``str``
            (defaults to ``merge.MAX_ENUM_MEMBERS``).
        cache_dir: Optional directory for the converted-document cache
            (``prepare_document_text``): converted sources export a
            ``<stem>.document.json`` there and report it via
            ``DocumentStats.cache_path`` so trial runs re-enter without
            re-conversion (design §4.1/§7.2).

    Raises:
        ValueError: ``sources`` is empty.
        ExtractionError: No source yielded usable text, or a pass failed
            after its retry (including the no-progress guard).
        TemplateLintError: ``strict=True`` and the draft required repairs.
    """
    if not sources:
        raise ValueError("induce_spec_from_documents requires at least one source")

    per_document: list[DocumentCandidates] = []
    document_stats: list[DocumentStats] = []
    skipped: list[str] = []
    for source in sources:
        prepared = prepare_document_text(
            source, doc_processor=doc_processor, budget_chars=budget_chars, cache_dir=cache_dir
        )
        if len(prepared.markdown) < MIN_DOC_CHARS:
            logger.warning(
                "Skipping %s: only %d chars of text (check OCR settings)",
                prepared.name,
                len(prepared.markdown),
            )
            skipped.append(prepared.name)
            continue
        stats = DocumentStats(
            name=prepared.name, sampled=prepared.sampled, cache_path=prepared.cache_path
        )
        candidates = _induce_document(
            prepared, llm_call_fn, root_name=root_name, max_models=max_models, stats=stats
        )
        document_stats.append(stats)
        if candidates.classes:
            per_document.append(candidates)

    if not per_document:
        raise ExtractionError(
            "Template induction found no usable content in any source "
            "(near-empty text or no classes proposed); check docling OCR settings",
            details={"sources": [str(s) for s in sources], "skipped": skipped},
        )

    merge_kwargs: dict[str, Any] = {"root_name": root_name, "max_models": max_models}
    if max_enum_members is not None:
        merge_kwargs["max_enum_members"] = max_enum_members
    draft, merge_report, merge_gaps = merge_documents(per_document, **merge_kwargs)
    spec, lint_report = repair_draft(draft, strict=strict)
    report = InductionReport(
        documents=document_stats,
        skipped_sources=skipped,
        merge=merge_report,
        lint=lint_report,
        gaps=_dedupe_gaps([*merge_gaps, *lint_report.gaps]),
    )
    return spec, report
