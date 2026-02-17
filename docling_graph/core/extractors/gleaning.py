"""
Optional second-pass extraction (gleaning) to improve recall.

After a first extraction, a gleaning pass asks the LLM "what did you miss?"
and merges additional entities/relationships. Configurable per contract.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..utils.dict_merger import deep_merge_dicts

logger = logging.getLogger(__name__)

DEFAULT_DESCRIPTION_MERGE_FIELDS = frozenset({"description", "summary"})
DEFAULT_DESCRIPTION_MERGE_MAX_LENGTH = 4096


@dataclass
class GleaningConfig:
    """Config for optional gleaning (second-pass extraction)."""

    max_passes: int = 1
    prompt_builder: Callable[..., dict[str, str]] | None = None


def get_gleaning_prompt_direct(
    markdown: str,
    existing_result: dict[str, Any],
    schema_json: str,
) -> dict[str, str]:
    """Build system and user prompts for a direct gleaning pass."""
    existing_summary = json.dumps(existing_result, indent=0, ensure_ascii=False)
    if len(existing_summary) > 8000:
        existing_summary = existing_summary[:8000] + "\n... (truncated)"
    system = (
        "You are an extraction assistant. You will be shown what was ALREADY extracted from a document. "
        "Your task is to extract any ADDITIONAL information (entities, relations, facts) that were "
        "missed. Return ONLY valid JSON that matches the provided schema. Do not duplicate or repeat "
        "information already in the 'already extracted' section."
    )
    user = (
        "=== ALREADY EXTRACTED ===\n"
        f"{existing_summary}\n"
        "=== END ALREADY EXTRACTED ===\n\n"
        "=== DOCUMENT ===\n"
        f"{markdown}\n"
        "=== END DOCUMENT ===\n\n"
        "=== TARGET SCHEMA ===\n"
        f"{schema_json}\n"
        "=== END SCHEMA ===\n\n"
        "Extract any ADDITIONAL information from the document that is not already in the "
        "'ALREADY EXTRACTED' section. Return ONLY a JSON object that follows the target schema."
    )
    return {"system": system, "user": user}


def run_gleaning_pass_direct(
    markdown: str,
    existing_result: dict[str, Any],
    schema_json: str,
    llm_call_fn: Callable[[dict[str, str]], dict | list | None],
) -> dict[str, Any] | None:
    """
    Run one gleaning pass for direct extraction: prompt with existing result, parse response.

    Args:
        markdown: Full document markdown.
        existing_result: Template-shaped dict from first extraction.
        schema_json: JSON schema string.
        llm_call_fn: Callable that takes prompt dict (system/user) and returns parsed JSON or None.

    Returns:
        Parsed gleaned dict or None on failure.
    """
    prompt = get_gleaning_prompt_direct(markdown, existing_result, schema_json)
    try:
        parsed = llm_call_fn(prompt)
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception as e:
        logger.warning("Gleaning pass failed: %s", e)
        return None


def merge_gleaned_direct(
    existing: dict[str, Any],
    extra: dict[str, Any],
    description_merge_fields: frozenset[str] | None = None,
    description_merge_max_length: int = DEFAULT_DESCRIPTION_MERGE_MAX_LENGTH,
) -> dict[str, Any]:
    """
    Merge gleaned (extra) result into existing direct result.

    Uses deep_merge_dicts with description merge for configured fields.
    """
    if description_merge_fields is None:
        description_merge_fields = DEFAULT_DESCRIPTION_MERGE_FIELDS
    merged = copy.deepcopy(existing)
    deep_merge_dicts(
        merged,
        extra,
        description_merge_fields=description_merge_fields,
        description_merge_max_length=description_merge_max_length,
    )
    return merged


def build_already_found_summary_delta(
    merged_graph: dict[str, Any],
    max_nodes: int = 100,
    max_rels: int = 50,
) -> str:
    """
    Build a compact text summary of merged graph for delta gleaning prompt.

    Used to inject "already extracted from other batches" into batch prompts.
    """
    lines: list[str] = []
    nodes = merged_graph.get("nodes") or []
    rels = merged_graph.get("relationships") or []
    for i, n in enumerate(nodes[:max_nodes]):
        if not isinstance(n, dict):
            continue
        path = n.get("path", "")
        ids = n.get("ids") or {}
        props = n.get("properties") or {}
        ids_str = json.dumps(ids, ensure_ascii=False)
        desc = (props.get("description") or props.get("summary") or "")[:200]
        lines.append(f"- Node path={path} ids={ids_str} description={desc!r}...")
    lines.append("")
    for i, r in enumerate(rels[:max_rels]):
        if not isinstance(r, dict):
            continue
        src = r.get("source_key") or r.get("source_id") or ""
        tgt = r.get("target_key") or r.get("target_id") or ""
        label = r.get("label") or r.get("edge_label") or ""
        lines.append(f"- Relationship {src} -> {tgt} label={label}")
    return "\n".join(lines) if lines else "No nodes or relationships yet."
