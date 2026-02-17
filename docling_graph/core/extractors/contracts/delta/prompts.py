"""Prompt templates for the delta extraction contract."""

from __future__ import annotations

from typing import Sequence


def get_delta_batch_prompt(
    *,
    batch_markdown: str,
    schema_semantic_guide: str,
    path_catalog_block: str,
    batch_index: int,
    total_batches: int,
    global_context: str | None = None,
    already_found: str | None = None,
) -> dict[str, str]:
    """Build system/user prompts for one delta batch extraction."""

    system_prompt = (
        "You are an expert extraction engine for graph construction. "
        "Return ONLY strict JSON with top-level keys 'nodes' and 'relationships'.\n\n"
        "Rules:\n"
        "1. Use exact catalog paths for 'path' and parent; never invent paths or use class names. "
        "Put only identity fields in ids; other values go in properties. ids keys must match catalog.\n"
        "2. Model nested entities as separate nodes (flat properties only; no nested objects in properties). "
        "For any list-entity path in the catalog (paths ending in [] with id_fields): set identity in ids from the "
        "document (tables, section titles, captions that name the entity). Put child entities on the child path with "
        "parent reference; when emitting children whose parent is a list path, also emit a parent-path node with ids "
        "set from the document so parent lookup can attach them. Never put child content under the parent's id field.\n"
        "3. Identity MUST come from the document (tables, captions, section titles that name entities). "
        "Keep identifiers stable and consistent across the entire document so they merge across batches. "
        "Omit when not evidenced in this batch.\n"
        "4. Use catalog and guidance to decide instances; omit generic headings. Emit list-entity nodes (path ending in []) "
        "only when this batch contains the defining structure for that identity.\n"
        "5. Canonicalize: trim whitespace, stable casing, numeric/date in machine form. Valid JSON only; no markdown "
        "or batch metadata in node content."
    )

    user_prompt = f"[Batch {batch_index + 1}/{total_batches} — for context only; do not put this into any field.]\n\n"
    if already_found:
        user_prompt += (
            "=== ALREADY EXTRACTED (from other batches; do not duplicate) ===\n"
            f"{already_found}\n"
            "=== END ALREADY EXTRACTED ===\n\n"
            "Extract any ADDITIONAL nodes/relationships from this batch not already covered above.\n\n"
        )
    if global_context:
        user_prompt += (
            "=== DOCUMENT CONTEXT (use for stable identity values across batches) ===\n"
            f"{global_context}\n"
            "=== END DOCUMENT CONTEXT ===\n\n"
        )
    user_prompt += (
        "=== BATCH DOCUMENT ===\n"
        f"{batch_markdown}\n"
        "=== END BATCH DOCUMENT ===\n\n"
        "=== TEMPLATE PATH CATALOG ===\n"
        f"{path_catalog_block}\n"
        "=== END CATALOG ===\n\n"
        "=== SEMANTIC FIELD GUIDANCE ===\n"
        f"{schema_semantic_guide}\n"
        "=== END GUIDANCE ===\n\n"
        'Identity from document only; use catalog ids=[...] per path. Parent: {"path": "<catalog path>", "ids": {}} or null for root. '
        "For list-entity paths in the catalog, set ids from the document; when emitting children under a list parent, "
        "also emit the parent-path node with ids set so parent lookup can attach.\n\n"
        'Return JSON: {"nodes": [...], "relationships": [...]} with each node: {path, node_type?, ids, parent, properties}.'
    )

    return {"system": system_prompt, "user": user_prompt}


def format_batch_markdown(chunks: Sequence[str]) -> str:
    """Join chunk payloads with stable delimiters for one LLM batch call."""

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks):
        blocks.append(f"--- CHUNK {idx + 1} ---\n{chunk}")
    return "\n\n".join(blocks)
