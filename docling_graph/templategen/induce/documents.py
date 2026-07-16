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

Sources come in three shapes (``prepare_document_text``): local file paths
(``.md``/``.markdown``/``.txt`` read directly, everything else through the
injected ``doc_processor``), ``http(s)://`` URLs (always through the
``doc_processor`` — Docling fetches and converts them), and
:class:`DocumentContent` objects carrying in-memory text for programmatic
callers that already hold the document contents.

Scale is handled by two mechanisms sharing one unit abstraction
(``prepare_document_windows``): an **oversized document** becomes up to
:data:`MAX_WINDOWS_PER_DOC` evenly spread windows, each inducted as its own
unit and merged back as one document (a 200-page report contributes evidence
from its whole body instead of a head-biased sample); a **large corpus**
(> :data:`SATURATION_MIN_UNITS` units) processes in deterministic name-hash
order and stops once :data:`SATURATION_STREAK` consecutive units add no new
structure — skipped units are reported, never silent.

Deterministic evidence gates (design §4.3) run before anything enters the
candidate set:

- **verbatim gate** — every identity/field example must be a
  whitespace-normalized substring of the document text the model saw;
- **digit-honesty** — ``*_number``/``*_no``/``ref_*`` identity candidates
  whose surviving examples hold no digit are renamed to ``name``;
- **cardinality gate** — ``documented_max_count`` needs an evidence quote
  with a digit/number word adjacent to the class concept.

Pass 2 (fields) has one extra recovery lever the other passes lack: a batch
that keeps failing on **truncated** output (the escalation retry inside the
injected callable already fired) is split in half and re-tried, down to
single-class batches — smaller batches need proportionally less output room,
mirroring the dense contract's skeleton-batch splitting.

Gated per-document candidates then merge deterministically (``merge.py``)
into a loose draft that ``linter.repair_draft`` turns into a valid, linted
:class:`~docling_graph.templategen.spec.TemplateSpec`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.exceptions import ExtractionError
from docling_graph.logging_utils import get_component_logger

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

logger = get_component_logger("TemplateInduction", __name__)

__all__ = [
    "DEFAULT_BUDGET_CHARS",
    "ELISION_MARKER",
    "MAX_PASS2_BATCH",
    "MAX_WINDOWS_PER_DOC",
    "MIN_DOC_CHARS",
    "SATURATION_MIN_UNITS",
    "SATURATION_STREAK",
    "DocumentContent",
    "DocumentStats",
    "InductionReport",
    "LlmCallFn",
    "PreparedDoc",
    "SourceInput",
    "induce_spec_from_documents",
    "prepare_document_text",
    "prepare_document_windows",
    "source_display_name",
]

LlmCallFn = Callable[..., Any]
"""Injected LLM callable; see the module docstring for the exact contract."""

DEFAULT_BUDGET_CHARS = 24_000
"""Default per-document character budget for the sampler (design §4.1)."""

MAX_PASS2_BATCH = 6
"""Pass 2 sends at most this many classes per call (design §4.2)."""

MIN_DOC_CHARS = 200
"""Documents yielding less text are skipped (scanned-PDF guard, design §9)."""

MAX_WINDOWS_PER_DOC = 6
"""An oversized document splits into at most this many induction windows."""

SATURATION_MIN_UNITS = 10
"""The saturation stop only engages above this many induction units — small,
curated corpora are always induced in full, in source order."""

SATURATION_STREAK = 6
"""Consecutive quiet units (no new classes, ~no new fields) that end induction."""

_QUIET_NEW_FIELDS_MAX = 1
"""A unit is quiet when it adds no new class and at most this many new fields
(one stray field key absorbs small-model naming jitter)."""

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


class DocumentContent(BaseModel):
    """Document text passed directly by API callers — no file, no conversion.

    Use this when the document contents are already in hand (fetched, piped,
    or produced upstream): the ``text`` goes straight to the budget sampler
    under the given ``name``, which is what reports and gap traces show.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    text: str


SourceInput = str | Path | DocumentContent
"""One induction source: a file path, an ``http(s)://`` URL, or direct text."""


class PreparedDoc(BaseModel):
    """One induction unit: a source document (or one window of an oversized
    one), converted and budgeted for induction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    markdown: str
    sampled: bool = False
    cache_path: Path | None = None
    """The cached ``<stem>.document.json`` when a ``cache_dir`` was given and
    the source needed conversion; re-enters the pipeline as DOCLING_DOCUMENT
    input (no re-OCR) for --trial-run/evaluate (design §4.1/§7.2)."""
    window_index: int | None = None
    """0-based window position when this unit is one slice of an oversized
    document (``prepare_document_windows``); ``None`` for whole documents."""
    window_count: int | None = None


def _is_url(source: str) -> bool:
    """Mirror of ``InputTypeDetector._is_url``: http(s) schemes only."""
    return source.startswith(("http://", "https://"))


def source_display_name(source: SourceInput) -> str:
    """The stable per-source name used in ``PreparedDoc`` and every report.

    File paths use their basename, URLs the last path segment (falling back
    to the host), and :class:`DocumentContent` its explicit ``name``. The CLI
    relies on this to match report entries back to its argument list.
    """
    if isinstance(source, DocumentContent):
        return source.name
    raw = str(source)
    if _is_url(raw):
        split = urlsplit(raw)
        return PurePosixPath(split.path).name or split.netloc or raw
    return Path(raw).name or raw


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


def _cache_docling_document(document: Any, name: str, cache_dir: Path) -> Path | None:
    """Export a converted DoclingDocument as ``<stem>.document.json``.

    ``name`` is the source's display name; its stem is sanitized so URL-derived
    names always yield a writable filename. Reuses
    :class:`~docling_graph.core.exporters.docling_exporter.DoclingExporter`
    (imported lazily to keep this module import-light). Best-effort: a cache
    failure only costs the no-re-OCR shortcut, never the induction itself.
    """
    try:
        from docling_graph.core.exporters.docling_exporter import DoclingExporter

        stem = re.sub(r"[^\w.-]+", "_", Path(name).stem) or "document"
        exported = DoclingExporter(output_dir=cache_dir).export_document(
            document,
            base_name=f"{stem}.document",
            include_json=True,
            include_markdown=False,
            include_doclang=False,
        )
        raw = exported.get("document_json")
        return Path(raw) if isinstance(raw, str) else None
    except Exception as e:  # pragma: no cover - defensive: cache is an optimization
        logger.warning("Could not cache the DoclingDocument for %s: %s", name, e)
        return None


def _load_source_text(
    source: SourceInput,
    doc_processor: Any | None,
    cache_dir: str | Path | None,
) -> tuple[str, str, Path | None]:
    """Read/convert one source to ``(name, markdown, cache_path)``.

    :class:`DocumentContent` carries its text directly; ``.md``/``.markdown``/
    ``.txt`` files are read as-is; every other input — including ``http(s)://``
    URLs, which Docling fetches itself — goes through ``doc_processor``.

    Raises:
        ValueError: A URL or non-text input was given without a
            ``doc_processor``.
    """
    name = source_display_name(source)
    cache_path: Path | None = None
    if isinstance(source, DocumentContent):
        markdown = source.text
    elif not _is_url(str(source)) and Path(str(source)).suffix.lower() in _TEXT_SUFFIXES:
        markdown = Path(str(source)).read_text(encoding="utf-8")
    else:
        if doc_processor is None:
            raise ValueError(
                f"prepare_document_text: '{source}' needs a DocumentProcessor "
                "(only .md/.markdown/.txt files and DocumentContent are read directly)"
            )
        document = doc_processor.convert_to_docling_doc(str(source))
        markdown = doc_processor.extract_full_markdown(document)
        if cache_dir is not None:
            cache_path = _cache_docling_document(document, name, Path(cache_dir))
    return name, markdown, cache_path


def prepare_document_text(
    source: SourceInput,
    *,
    doc_processor: Any | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    cache_dir: str | Path | None = None,
) -> PreparedDoc:
    """Convert one source to induction text and budget-sample it.

    The single-unit view of a source (head/tail-weighted 60/20/20 sample when
    the text exceeds ``budget_chars``). The induction pipeline itself uses
    :func:`prepare_document_windows`, which covers oversized documents with
    evenly spread full-text windows instead of one head-biased sample; this
    function remains the one-sample entry point for API callers.

    With ``cache_dir`` set, a converted (file or URL) source is additionally
    exported as ``<stem>.document.json`` there and the path is returned on
    ``PreparedDoc.cache_path`` — the pipeline's DOCLING_DOCUMENT input type
    re-enters that file without re-conversion, so trial runs never re-OCR
    (design §4.1/§7.2).

    Raises:
        ValueError: A URL or non-text input was given without a
            ``doc_processor``.
    """
    name, markdown, cache_path = _load_source_text(source, doc_processor, cache_dir)
    sampled = len(markdown) > budget_chars
    if sampled:
        markdown = _sample_text(markdown, budget_chars)
    return PreparedDoc(name=name, markdown=markdown, sampled=sampled, cache_path=cache_path)


def _snap_to_line(text: str, position: int) -> int:
    """The start of the line containing ``position`` (window cuts stay clean)."""
    if position <= 0:
        return 0
    if position >= len(text):
        return len(text)
    newline = text.rfind("\n", 0, position)
    return newline + 1 if newline >= 0 else 0


def _window_bounds(total: int, budget_chars: int, max_windows: int) -> list[tuple[int, int]]:
    """Evenly spread window ``(start, end)`` bounds over ``total`` chars.

    ``ceil(total / budget)`` windows, capped at ``max_windows``. Below the cap
    the windows tile the document (window length ``ceil(total / count)``,
    full coverage with slight overlaps from rounding); at the cap they are
    ``budget_chars`` long and evenly strided, leaving gaps — still far more
    representative than one head-biased sample.
    """
    if total <= budget_chars:
        return [(0, total)]
    count = min(max_windows, -(-total // budget_chars))
    window_len = min(budget_chars, -(-total // count))
    bounds: list[tuple[int, int]] = []
    for index in range(count):
        start = round(index * (total - window_len) / (count - 1)) if count > 1 else 0
        bounds.append((start, min(total, start + window_len)))
    return bounds


def prepare_document_windows(
    source: SourceInput,
    *,
    doc_processor: Any | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    cache_dir: str | Path | None = None,
    max_windows: int = MAX_WINDOWS_PER_DOC,
) -> list[PreparedDoc]:
    """Convert one source into one or more induction units (windows).

    A document within ``budget_chars`` is a single full-text unit. An
    oversized one becomes up to ``max_windows`` line-aligned windows spread
    evenly across the whole text, each entering induction as its own unit
    named ``"<name> [i/k]"`` — the cross-document merge then unions their
    candidates exactly like separate documents (with window units grouped
    back to their physical document for document-count semantics). This
    replaces the head-biased single sample for the induction path: a
    200-page document contributes evidence from its whole body, and each
    window's verbatim gate checks against text the model actually saw.

    The conversion cache (``cache_dir``) is exported once and carried on the
    first window only.

    Raises:
        ValueError: A URL or non-text input was given without a
            ``doc_processor``.
    """
    name, markdown, cache_path = _load_source_text(source, doc_processor, cache_dir)
    bounds = _window_bounds(len(markdown), budget_chars, max(1, max_windows))
    if len(bounds) == 1:
        return [PreparedDoc(name=name, markdown=markdown, sampled=False, cache_path=cache_path)]
    total = len(markdown)
    covered = 0
    snapped: list[tuple[int, int]] = []
    for start, end in bounds:
        start = _snap_to_line(markdown, start)
        end = _snap_to_line(markdown, end) if end < total else total
        if end <= start:
            continue
        snapped.append((start, end))
        covered += end - start
    units: list[PreparedDoc] = []
    for index, (start, end) in enumerate(snapped):
        header = (
            f"[docling-graph] window {index + 1} of {len(snapped)} over {name}: "
            f"chars {start}-{end} of {total} (content outside this window is not shown)"
        )
        units.append(
            PreparedDoc(
                name=f"{name} [{index + 1}/{len(snapped)}]",
                markdown=header + "\n" + markdown[start:end],
                sampled=covered < total,
                cache_path=cache_path if index == 0 else None,
                window_index=index,
                window_count=len(snapped),
            )
        )
    return units


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
    pass2_splits: int = 0
    """Times a pass-2 class batch was halved because its output kept hitting
    the model's max_tokens even after the escalation retry."""


class InductionReport(BaseModel):
    """Everything ``induce_spec_from_documents`` decided along the way."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentStats] = Field(default_factory=list)
    skipped_sources: list[str] = Field(default_factory=list)
    merge: MergeReport
    lint: LintReport
    gaps: list[SpecGap] = Field(default_factory=list)
    units_total: int = 0
    """Induction units planned (a document = 1 unit; oversized documents
    split into windows). 0 on reports predating unit planning."""
    skipped_capped: list[str] = Field(default_factory=list)
    """Units never induced because ``max_units`` capped the run."""
    skipped_saturated: list[str] = Field(default_factory=list)
    """Units never induced because the schema stopped changing
    (:data:`SATURATION_STREAK` consecutive quiet units)."""


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


def _truncation_in_chain(error: BaseException | None) -> bool:
    """Did any exception in the ``__cause__`` chain flag a truncated response?

    ``DoclingGraphError`` subclasses carry a ``details`` dict; the LLM clients
    set ``details["truncated"]`` when the response hit max_tokens and could
    not be repaired. Walking the chain (both the ``DoclingGraphError.cause``
    attribute and dunder ``__cause__``) keeps this module free of any
    ``llm_clients`` import (the injected-callable discipline).
    """
    depth = 0
    while error is not None and depth < 8:
        details = getattr(error, "details", None)
        if isinstance(details, dict) and details.get("truncated"):
            return True
        error = getattr(error, "cause", None) or error.__cause__
        depth += 1
    return False


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


_BatchKey = tuple[int, ...]
"""Pass-2 batch position: ``(index,)`` initially, extended per truncation split.
Lexicographic order over keys is the original class order, so payloads always
apply deterministically no matter which call finishes first."""


def _run_pass2(
    prepared: PreparedDoc,
    llm_call_fn: LlmCallFn,
    classes: Sequence[ClassCandidate],
    classes_by_key: Mapping[str, ClassCandidate],
    stats: DocumentStats,
    *,
    batch_size: int,
    workers: int,
) -> None:
    """Pass 2 coordinator: bounded concurrency, truncation splits, ordered apply.

    Batches are independent LLM calls (they only read the prepared markdown),
    so with ``workers > 1`` they run concurrently. A batch whose output keeps
    hitting max_tokens (the injected callable's escalation retry already
    fired) is halved and re-queued — smaller batches need proportionally less
    output room. Payloads are applied strictly in batch order after all calls
    return, so concurrency can never change the induced draft.
    """
    work: list[tuple[_BatchKey, list[ClassCandidate]]] = [
        ((index,), batch) for index, batch in enumerate(_batches(classes, batch_size))
    ]
    results: dict[_BatchKey, dict[str, Any]] = {}
    splits = 0
    retries = 0
    retries_lock = threading.Lock()

    def call(key: _BatchKey, batch: list[ClassCandidate]) -> dict[str, Any]:
        nonlocal retries
        scratch = DocumentStats(name=prepared.name)
        try:
            return _call_llm_pass(
                llm_call_fn,
                get_fields_prompt(
                    prepared.markdown,
                    doc_name=prepared.name,
                    classes=[(cls.name, cls.what_it_is) for cls in batch],
                ),
                fields_schema(),
                f"templategen_pass2_fields:{prepared.name}:batch{'_'.join(map(str, key))}",
                payload_key="classes",
                stats=scratch,
            )
        finally:
            with retries_lock:
                retries += scratch.retries

    def split(
        key: _BatchKey, batch: list[ClassCandidate], error: ExtractionError
    ) -> list[tuple[_BatchKey, list[ClassCandidate]]]:
        nonlocal splits
        if len(batch) <= 1 or not _truncation_in_chain(error):
            raise error
        half = len(batch) // 2
        splits += 1
        logger.warning(
            "Fields pass output for %s kept hitting max_tokens with %d classes; "
            "splitting the batch (%d+%d) and retrying",
            prepared.name,
            len(batch),
            half,
            len(batch) - half,
        )
        return [((*key, 0), batch[:half]), ((*key, 1), batch[half:])]

    if workers <= 1 or len(work) <= 1:
        pending = deque(work)
        while pending:
            key, batch = pending.popleft()
            try:
                results[key] = call(key, batch)
            except ExtractionError as e:
                pending.extendleft(reversed(split(key, batch, e)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future[dict[str, Any]], tuple[_BatchKey, list[ClassCandidate]]] = {
                pool.submit(call, key, batch): (key, batch) for key, batch in work
            }
            while futures:
                done, _running = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    key, batch = futures.pop(future)
                    try:
                        results[key] = future.result()
                    except ExtractionError as e:
                        for sub_key, sub_batch in split(key, batch, e):
                            futures[pool.submit(call, sub_key, sub_batch)] = (sub_key, sub_batch)

    stats.retries += retries
    stats.pass2_splits += splits
    for key in sorted(results):
        _apply_pass2(results[key], classes_by_key, prepared, stats)


def _induce_document(
    prepared: PreparedDoc,
    llm_call_fn: LlmCallFn,
    *,
    root_name: str | None,
    max_models: int,
    stats: DocumentStats,
    pass2_batch_size: int = MAX_PASS2_BATCH,
    pass2_workers: int = 1,
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

    _run_pass2(
        prepared,
        llm_call_fn,
        classes,
        classes_by_key,
        stats,
        batch_size=pass2_batch_size,
        workers=pass2_workers,
    )

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


def _unit_novelty(
    candidates: DocumentCandidates,
    seen_classes: set[str],
    seen_fields: set[tuple[str, str]],
) -> tuple[int, int]:
    """(new class keys, new field keys) a unit contributed; updates the sets."""
    new_classes = 0
    new_fields = 0
    for cls in candidates.classes:
        class_key = canonical_key(cls.name)
        if class_key and class_key not in seen_classes:
            seen_classes.add(class_key)
            new_classes += 1
        for field in cls.fields:
            field_key = canonical_key(field.name)
            if field_key and (class_key, field_key) not in seen_fields:
                seen_fields.add((class_key, field_key))
                new_fields += 1
    return new_classes, new_fields


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
    sources: Sequence[SourceInput],
    llm_call_fn: LlmCallFn,
    *,
    root_name: str | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    max_models: int = 30,
    strict: bool = False,
    doc_processor: Any | None = None,
    max_enum_members: int | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    pass2_batch_size: int | None = None,
    max_windows_per_doc: int = MAX_WINDOWS_PER_DOC,
    max_units: int = 0,
    saturation: bool = True,
) -> tuple[TemplateSpec, InductionReport]:
    """Induce a validated :class:`TemplateSpec` from example documents.

    Sources expand into **induction units** (``prepare_document_windows``): a
    document within the budget is one unit; an oversized one becomes up to
    ``max_windows_per_doc`` windows spread across its whole text. Per unit:
    pass 1 (class inventory) -> evidence gates -> pass 2 (fields, batched
    <= ``pass2_batch_size`` classes per call) -> pass 3 (relationships).
    Per-unit candidates merge deterministically (``merge.merge_documents``,
    windows grouped back to their physical document), the draft goes through
    ``linter.repair_draft`` (which doubles documented ``max_instances``
    exactly once and derives missing edge labels), and the repaired spec is
    returned with a full :class:`InductionReport`.

    Large corpora don't pay linear spend for sub-linear information: above
    :data:`SATURATION_MIN_UNITS` units, processing follows a deterministic
    name-hash order (decorrelating vendor/date filename grouping) and stops
    once :data:`SATURATION_STREAK` consecutive units contribute no new
    classes and ~no new fields; ``max_units`` is the hard cap on top.
    Skipped units are reported (``skipped_saturated`` / ``skipped_capped``),
    and induced candidates still merge in source order, so precedence
    semantics never depend on the processing order.

    Conversion (``prepare_document_windows``) runs sequentially — document
    converters are not safely reentrant — but with ``workers > 1`` the LLM
    passes run concurrently: units in parallel waves, and pass-2 field
    batches in parallel within each unit (the worker budget is divided
    between the two levels). Results are deterministic regardless of
    ``workers``. ``llm_call_fn`` must then be safe to call from multiple
    threads (the CLI binds one LLM client per thread).

    Args:
        sources: Document sources — file paths, ``http(s)://`` URLs, or
            :class:`DocumentContent` objects with the text in hand.
            ``.md``/``.markdown``/``.txt`` files and ``DocumentContent`` are
            read directly, everything else (URLs included) needs
            ``doc_processor``.
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
        workers: Concurrent LLM calls (1 = fully sequential). ``llm_call_fn``
            must be thread-safe when > 1.
        pass2_batch_size: Classes per pass-2 call (defaults to
            :data:`MAX_PASS2_BATCH`). The CLI sizes this against the model's
            output budget so small-budget models stop truncating.
        max_windows_per_doc: Windows an oversized document may split into.
        max_units: Hard cap on induced units (0 = unlimited). Excess units
            are skipped and reported under ``skipped_capped``.
        saturation: Enable the diminishing-returns stop for large corpora
            (only engages above :data:`SATURATION_MIN_UNITS` units).

    Raises:
        ValueError: ``sources`` is empty.
        ExtractionError: No source yielded usable text, or a pass failed
            after its retry (including the no-progress guard).
        TemplateLintError: ``strict=True`` and the draft required repairs.
    """
    if not sources:
        raise ValueError("induce_spec_from_documents requires at least one source")

    # Phase 1 (sequential): expand sources into induction units. Windows of
    # one physical document share its group index for the merge's
    # document-count semantics (rare-field flag, source_ref labels).
    units: list[tuple[PreparedDoc, int, DocumentStats]] = []
    group_names: list[str] = []
    skipped: list[str] = []
    for group_index, source in enumerate(sources):
        group_names.append(source_display_name(source))
        for prepared in prepare_document_windows(
            source,
            doc_processor=doc_processor,
            budget_chars=budget_chars,
            cache_dir=cache_dir,
            max_windows=max_windows_per_doc,
        ):
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
            units.append((prepared, group_index, stats))

    # Phase 2: pick the processing order and budget. Above the saturation
    # floor, a deterministic name-hash order decorrelates filename grouping
    # (vendor/date prefixes) so the quiet-streak stop cannot falsely converge
    # on a homogeneous alphabetical prefix.
    processing = list(range(len(units)))
    saturation_active = saturation and len(units) > SATURATION_MIN_UNITS
    if saturation_active:
        processing.sort(key=lambda i: hashlib.blake2b(units[i][0].name.encode()).hexdigest())
        logger.info(
            "Corpus mode: %d induction units from %d source(s) (~%d+ LLM calls if "
            "exhaustive) — hash-ordered processing with saturation stop after %d "
            "quiet units",
            len(units),
            len(sources),
            3 * len(units),
            SATURATION_STREAK,
        )
    skipped_capped: list[str] = []
    if max_units and len(processing) > max_units:
        skipped_capped = [units[i][0].name for i in processing[max_units:]]
        processing = processing[:max_units]
        logger.warning(
            "Unit cap: inducing %d of %d units (max_units=%d); %d skipped",
            len(processing),
            len(units),
            max_units,
            len(skipped_capped),
        )

    batch_size = pass2_batch_size if pass2_batch_size and pass2_batch_size > 0 else MAX_PASS2_BATCH
    wave_size = min(max(1, workers), len(processing)) if processing else 1
    pass2_workers = max(1, max(1, workers) // wave_size)

    def induce(unit_index: int) -> DocumentCandidates:
        prepared, _group, stats = units[unit_index]
        return _induce_document(
            prepared,
            llm_call_fn,
            root_name=root_name,
            max_models=max_models,
            stats=stats,
            pass2_batch_size=batch_size,
            pass2_workers=pass2_workers,
        )

    # Phase 3: induce in waves of ``wave_size``; between waves, the quiet
    # streak (units adding no new classes and ~no new fields, evaluated in
    # processing order) decides whether the rest is worth paying for.
    results: dict[int, DocumentCandidates] = {}
    seen_classes: set[str] = set()
    seen_fields: set[tuple[str, str]] = set()
    quiet_streak = 0
    position = 0
    while position < len(processing):
        wave = processing[position : position + wave_size]
        if len(wave) > 1:
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                wave_results = list(pool.map(induce, wave))
        else:
            wave_results = [induce(wave[0])]
        for unit_index, candidates in zip(wave, wave_results, strict=True):
            results[unit_index] = candidates
            new_classes, new_fields = _unit_novelty(candidates, seen_classes, seen_fields)
            quiet = new_classes == 0 and new_fields <= _QUIET_NEW_FIELDS_MAX
            quiet_streak = quiet_streak + 1 if quiet else 0
        position += len(wave)
        if saturation_active and quiet_streak >= SATURATION_STREAK and position < len(processing):
            break

    skipped_saturated = [units[i][0].name for i in processing[position:]]
    if skipped_saturated:
        logger.info(
            "Saturation stop: the last %d unit(s) added no new structure — %d of %d "
            "units induced, %d skipped (pass --exhaustive to induce everything)",
            quiet_streak,
            position,
            len(units),
            len(skipped_saturated),
        )

    # Merge in SOURCE order regardless of processing order: precedence
    # semantics (first-seen names, descriptions) stay the caller's order.
    induced = sorted(processing[:position])
    document_stats = [units[i][2] for i in induced]
    per_document = [results[i] for i in induced if results[i].classes]
    doc_groups = [units[i][1] for i in induced if results[i].classes]

    if not per_document:
        raise ExtractionError(
            "Template induction found no usable content in any source "
            "(near-empty text or no classes proposed); check docling OCR settings",
            details={"sources": [source_display_name(s) for s in sources], "skipped": skipped},
        )

    merge_kwargs: dict[str, Any] = {
        "root_name": root_name,
        "max_models": max_models,
        "doc_groups": doc_groups,
        "group_names": group_names,
    }
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
        units_total=len(units),
        skipped_capped=skipped_capped,
        skipped_saturated=skipped_saturated,
    )
    return spec, report
