"""
Ontology -> SPEC-draft compilers (the deterministic Stage-1b front-end).

Each submodule compiles one ontology format into a *loose draft* dict shaped
like :class:`~docling_graph.templategen.spec.TemplateSpec` plus a list of
declared :class:`~docling_graph.templategen.spec.SpecGap`:

- :mod:`.owl` — OWL/RDFS/SKOS via rdflib (``templategen`` extra);
- :mod:`.linkml` — LinkML via linkml-runtime ``SchemaView`` (``templategen`` extra);
- :mod:`.jsonschema` — JSON Schema, stdlib only.

The compilers are pure functions of their input: **zero LLM calls**. Drafts
apply the design's own demotion rules (no-identity class -> component, root
identity synthesis) so they validate directly in the common case; the linter's
``repair_draft`` remains a safety net, not a crutch.

This module owns format sniffing (:func:`sniff_ontology_format`), dispatch
(:func:`spec_draft_from_ontology`), and the small helpers every compiler
shares (identity ladder, cardinality sentences, name deduplication).
"""

from __future__ import annotations

import fnmatch
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from docling_graph.templategen.spec import SpecGap

ONTOLOGY_FORMATS: tuple[str, ...] = ("owl", "linkml", "jsonschema")
"""Formats the dispatcher accepts (plus ``"auto"`` for sniffing)."""

RDF_SUFFIXES: frozenset[str] = frozenset(
    {".ttl", ".turtle", ".rdf", ".owl", ".nt", ".n3", ".trig", ".nq", ".jsonld"}
)
"""File suffixes that identify an RDF serialization without content sniffing."""

IDENTITY_LADDER: tuple[str, ...] = ("id", "identifier", "name", "title", "label", "code", "number")
"""Heuristic identity-field local names, best first (design §5.1). A field
matches by exact snake_case name or by ``_<token>`` suffix (``policy_number``)."""

DOCUMENT_REFERENCE_FIELD = "document_reference"
"""Synthesized root identity when the ontology gives the root no identity."""

DOCUMENT_REFERENCE_DESCRIPTION = (
    "Identifier printed on the document, e.g. reference number or title."
)

MIN_CARDINALITY_NOTE = "Always present in conforming documents."
"""Appended to descriptions for ``minCardinality >= 1`` — never rendered as a
required field (the Optionality Law outranks the ontology)."""


def cardinality_sentence(documented_max: int) -> str:
    """Docstring sentence that satisfies the R13 docstring-states-cardinality rule.

    ``max_instances`` ownership (R13 contract): drafts carry the **documented**
    maximum — this sentence and the draft value both state ``documented_max``
    verbatim. The linter's ``repair_draft`` doubles it exactly once into the
    stored ``graph_max_instances`` (the "~2x documented maximum" rule,
    best-practices.md); compilers never pre-multiply.
    """
    return f"At most {documented_max} per document."


def identity_ladder_rank(field_name: str) -> int | None:
    """Rank of ``field_name`` on :data:`IDENTITY_LADDER` (lower = better), or None."""
    for rank, token in enumerate(IDENTITY_LADDER):
        if field_name == token or field_name.endswith(f"_{token}"):
            return rank
    return None


def pick_ladder_identity(field_names: Sequence[str]) -> str | None:
    """Best identity candidate among ``field_names`` per the heuristic ladder.

    Deterministic: ties on ladder rank break alphabetically.
    """
    ranked = sorted(
        (rank, name) for name in field_names if (rank := identity_ladder_rank(name)) is not None
    )
    return ranked[0][1] if ranked else None


def document_reference_field() -> dict[str, Any]:
    """The synthesized root identity FieldSpec draft (design §5.1 root rule)."""
    return {
        "name": DOCUMENT_REFERENCE_FIELD,
        "type": "str",
        "role": "identity",
        "description": DOCUMENT_REFERENCE_DESCRIPTION,
        "examples": [],
    }


def placeholder_docstring(name: str) -> str:
    """Non-empty stand-in docstring for classes the ontology leaves undescribed.

    Recorded alongside a ``missing_docstring`` gap so gap-fill or the user can
    replace it; the draft stays constructible either way.
    """
    return f"A {name} as defined by the source ontology."


def unique_name(base: str, taken: set[str]) -> str:
    """Return ``base`` or the first ``base_N`` (N >= 2) not in ``taken``.

    Mutates ``taken`` by adding the returned name (distinct concepts colliding
    after PascalCase get a ``_2`` suffix plus a hard report line — design §9).
    """
    name = base
    counter = 2
    while name in taken:
        name = f"{base}_{counter}"
        counter += 1
    taken.add(name)
    return name


def class_passes_filters(
    local_name: str,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    *,
    is_root: bool = False,
) -> bool:
    """Apply ``--include``/``--exclude`` glob pruning to a class local name.

    The root class is always kept; ``exclude`` wins over ``include``.
    """
    if is_root:
        return True
    if exclude and any(fnmatch.fnmatchcase(local_name, pattern) for pattern in exclude):
        return False
    if include and not any(fnmatch.fnmatchcase(local_name, pattern) for pattern in include):
        return False
    return True


def require_optional_dependency(module: str, *, purpose: str) -> Any:
    """Import an optional dependency or fail with the install hint.

    Mirrors the ``check_provider_installed`` UX in ``cli/validators.py``: the
    error names the exact extra to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for {purpose} but is not installed. "
            "Install the template-generation extra: "
            "pip install 'docling-graph[templategen]'"
        ) from exc


def sniff_ontology_format(path: str | Path) -> str:
    """Detect the ontology format of ``path``: ``owl``, ``linkml`` or ``jsonschema``.

    Order of checks (design §1.1): a known RDF suffix short-circuits to
    ``owl``; JSON content with ``$schema``/``properties``/``$defs`` is
    ``jsonschema`` (JSON is valid YAML, so JSON is tested first); YAML content
    with top-level ``classes:`` and ``slots:`` keys is ``linkml``; anything
    rdflib can parse is ``owl``.
    """
    source = Path(path)
    if source.suffix.lower() in RDF_SUFFIXES:
        return "owl"
    text = source.read_text(encoding="utf-8", errors="replace")

    json_data: Any = None
    try:
        json_data = json.loads(text)
    except json.JSONDecodeError:
        json_data = None
    if isinstance(json_data, dict):
        if any(key in json_data for key in ("$schema", "properties", "$defs", "definitions")):
            return "jsonschema"
        # A JSON mapping without schema keywords may be JSON-LD.
        return "owl"

    yaml_data: Any = None
    try:
        yaml_data = yaml.safe_load(text)
    except yaml.YAMLError:
        yaml_data = None
    if isinstance(yaml_data, dict) and "classes" in yaml_data and "slots" in yaml_data:
        return "linkml"

    rdflib = require_optional_dependency("rdflib", purpose="ontology format sniffing")
    graph = rdflib.Graph()
    try:
        graph.parse(str(source))
    except Exception as exc:
        raise ValueError(
            f"Could not detect the ontology format of '{source}': not JSON Schema, "
            "not a LinkML YAML schema (needs top-level 'classes:' and 'slots:' keys), "
            f"and rdflib could not parse it ({exc}). "
            "Pass --format owl|linkml|jsonschema explicitly."
        ) from exc
    return "owl"


def _enforce_max_models(draft: dict[str, Any], max_models: int) -> None:
    """Refuse over-large drafts instead of silently truncating (design §9)."""
    models = draft.get("models", [])
    if len(models) <= max_models:
        return
    largest = sorted(models, key=lambda m: len(m.get("fields", [])), reverse=True)
    listing = ", ".join(f"{m['name']} ({len(m.get('fields', []))} fields)" for m in largest[:10])
    raise ValueError(
        f"Ontology compiled to {len(models)} models, over the max_models={max_models} cap. "
        f"Largest classes: {listing}. Prune with --exclude/--depth or raise "
        "templategen.max_models."
    )


def spec_draft_from_ontology(
    path: str | Path,
    *,
    fmt: str = "auto",
    root: str | None = None,
    depth: int = 4,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_models: int = 30,
) -> tuple[dict[str, Any], list[SpecGap]]:
    """Compile an ontology file into a loose ``TemplateSpec`` draft + gaps.

    Dispatches on ``fmt`` (``auto`` sniffs via :func:`sniff_ontology_format`)
    to the per-format compilers. Deterministic — no LLM is ever called.

    Raises:
        ValueError: Unknown ``fmt``, unsniffable content, or a draft larger
            than ``max_models`` (never silently truncated).
        ImportError: The format's optional dependency is missing.
    """
    resolved = fmt or "auto"
    if resolved == "auto":
        resolved = sniff_ontology_format(path)

    if resolved == "owl":
        from docling_graph.templategen.ontology.owl import spec_draft_from_owl as compile_fn
    elif resolved == "linkml":
        from docling_graph.templategen.ontology.linkml import (
            spec_draft_from_linkml as compile_fn,
        )
    elif resolved == "jsonschema":
        from docling_graph.templategen.ontology.jsonschema import (
            spec_draft_from_jsonschema as compile_fn,
        )
    else:
        raise ValueError(
            f"Unknown ontology format '{fmt}'. "
            f"Expected one of: auto, {', '.join(ONTOLOGY_FORMATS)}."
        )

    draft, gaps = compile_fn(path, root=root, depth=depth, include=include, exclude=exclude)
    _enforce_max_models(draft, max_models)
    return draft, gaps
