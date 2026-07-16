"""
Targeted LLM gap-fill: one call, declared gaps only, content never structure.

Both SPEC producers (document induction and the ontology compilers) leave
:class:`~docling_graph.templategen.spec.SpecGap` entries where deterministic
evidence ran out — missing docstrings, descriptions, identity examples.
:func:`fill_gaps` closes what it can with a **single** LLM call whose output
schema (``schemas.gapfill_schema``) has no slot for a class, field, or edge:
structure cannot change by construction, no matter what the model emits.

Guarantees:

- only fills whose ``(model, field, kind)`` key matches a declared gap are
  applied; everything else is silently discarded;
- touched models are marked ``provenance="gapfill"`` for review;
- gap-fill examples are **not** verbatim-gated (they are guidance, not
  document data — there is no source text to gate against) but each applied
  example fill is reported via the logger;
- a failed call degrades gracefully: the spec is returned unchanged with all
  gaps still open (gap-fill is an opt-in nicety, never a hard dependency);
- the input spec is never mutated; the returned spec is re-validated.

Uses the same injected ``llm_call_fn`` contract as ``documents.py``.
"""

from __future__ import annotations

import json
from typing import Any

from docling_graph.logging_utils import get_component_logger

from ..spec import MAX_FIELD_EXAMPLES, FieldSpec, ModelSpec, SpecGap, TemplateSpec
from .documents import LlmCallFn
from .prompts import get_gapfill_prompt
from .schemas import gapfill_schema

logger = get_component_logger("GapFill", __name__)

__all__ = ["fill_gaps"]


def _spec_summary(spec: TemplateSpec) -> str:
    """Compact schema listing for the gap-fill prompt (models and fields)."""
    lines: list[str] = []
    for model in spec.models:
        docstring = " ".join(model.docstring.split())[:160]
        lines.append(f"{model.name} ({model.kind}): {docstring}")
        for field in model.fields:
            role = f", {field.role}" if field.role != "property" else ""
            description = " ".join(field.description.split())[:100]
            suffix = f" — {description}" if description else ""
            lines.append(f"  - {field.name} ({field.type}{role}){suffix}")
    return "\n".join(lines)


def _find_field(model: ModelSpec, field_name: str | None) -> FieldSpec | None:
    if not field_name:
        return None
    return next((f for f in model.fields if f.name == field_name), None)


def _apply_fill(model: ModelSpec, field_name: str | None, kind: str, fill: dict[str, Any]) -> bool:
    """Apply one fill entry; returns True when it changed content."""
    if kind == "missing_docstring":
        docstring = str(fill.get("docstring") or "").strip()
        if not docstring:
            return False
        model.docstring = docstring
        return True
    if kind == "missing_description":
        field = _find_field(model, field_name)
        description = str(fill.get("description") or "").strip()
        if field is None or not description:
            return False
        field.description = description
        return True
    if kind in ("missing_examples", "missing_identity"):
        field = _find_field(model, field_name)
        if field is None:
            return False
        raw = fill.get("examples")
        examples: list[str] = []
        seen: set[str] = set()
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, str | int | float):
                continue
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                examples.append(text)
        if not examples:
            return False
        field.examples = examples[:MAX_FIELD_EXAMPLES]
        logger.info(
            "Gap-fill examples for %s.%s are LLM guidance, not verbatim document values: %s",
            model.name,
            field.name,
            field.examples,
        )
        return True
    # ambiguous_kind / missing_edge_label are structural or semantic judgments
    # the content-only gap-fill schema deliberately cannot express.
    return False


def fill_gaps(
    spec: TemplateSpec,
    gaps: list[SpecGap],
    llm_call_fn: LlmCallFn,
) -> tuple[TemplateSpec, list[SpecGap]]:
    """Fill declared documentation gaps with one LLM call.

    Args:
        spec: A valid spec (never mutated).
        gaps: The declared gaps; only these may be filled.
        llm_call_fn: The injected LLM callable (``documents.py`` contract).

    Returns:
        ``(new_spec, remaining_gaps)`` — the re-validated spec with content
        fills applied and touched models marked ``provenance="gapfill"``, plus
        every gap that stayed open (unfillable kinds, empty fills, discarded
        undeclared output, or a failed call).
    """
    if not gaps:
        return spec, []
    prompt = get_gapfill_prompt(_spec_summary(spec), gaps)
    try:
        payload = llm_call_fn(
            prompt=dict(prompt),
            schema_json=json.dumps(gapfill_schema()),
            context="templategen_gapfill",
        )
    except Exception as e:  # gap-fill must never break generation
        logger.warning("Gap-fill call failed: %s; all gaps left open", e)
        return spec, list(gaps)
    fills = payload.get("fills") if isinstance(payload, dict) else None
    if not isinstance(fills, list):
        logger.warning("Gap-fill returned no 'fills' list; all gaps left open")
        return spec, list(gaps)

    working = spec.model_copy(deep=True)
    models_by_name = {m.name: m for m in working.models}
    declared = {(gap.model, gap.field, gap.kind) for gap in gaps}
    filled: set[tuple[str, str | None, str]] = set()
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        model_name = str(fill.get("model") or "")
        field_name = str(fill.get("field") or "").strip() or None
        kind = str(fill.get("kind") or "")
        key = (model_name, field_name, kind)
        if key not in declared or key in filled:
            continue  # undeclared output is discarded — gaps are the contract
        model = models_by_name.get(model_name)
        if model is None:
            continue
        if _apply_fill(model, field_name, kind, fill):
            model.provenance = "gapfill"
            filled.add(key)

    remaining = [gap for gap in gaps if (gap.model, gap.field, gap.kind) not in filled]
    result = TemplateSpec.model_validate(working.model_dump())
    return result, remaining
