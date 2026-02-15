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
) -> dict[str, str]:
    """Build system/user prompts for one delta batch extraction."""

    system_prompt = (
        "You are an expert extraction engine for graph construction. "
        "Return ONLY strict JSON with top-level keys 'nodes' and 'relationships'.\n\n"
        "Rules:\n"
        "1. Model nested entities as separate nodes and relationships; do not emit nested objects as properties.\n"
        "2. Keep node and relationship properties flat: primitives or lists of primitives only. "
        "If a property would be nested, omit it.\n"
        "3. Use exact catalog paths for 'path' and parent references. Never invent new paths. "
        "Do not use class names or slash-separated paths (e.g., ScholarlyRheologyPaper, a/b).\n"
        "4. ids keys must exactly match the template identity fields for that path when applicable. "
        "If catalog ids are [none], set ids to {}.\n"
        "5. ONLY put identity fields in ids. Put all non-identity extracted values in properties "
        "on valid catalog paths.\n"
        "6. For entity paths with ids=[...], repeat the same identity values in flat properties "
        "(e.g., ids.line_number=2 implies properties.line_number=2 when known).\n"
        "7. Do not create synthetic pseudo-path nodes for scalar fields. "
        "For example, never emit nodes for property-like labels as paths unless those exact paths exist "
        "in the catalog.\n"
        "8. If data is not evidenced in this batch, omit it instead of fabricating placeholders.\n"
        "9. Keep identifiers stable across the entire document (not only this batch), "
        "and keep ID formatting canonical.\n"
        "10. Canonicalize scalar values: trim whitespace, keep stable casing for names, "
        "use uppercase codes (e.g., currencies), normalize units, and emit numeric/date scalars "
        "in machine form when possible.\n"
        "   Example amount: 'CHF 3360.00' -> 3360.00\n"
        "   Example date: '18 May 2024' -> '2024-05-18'\n"
        "11. Return valid JSON only (no markdown).\n"
        "12. Do not copy batch metadata (e.g. batch numbers or 'Delta extraction batch' text) into any "
        "node property; extract only from the BATCH DOCUMENT content below.\n"
        "13. Use TEMPLATE PATH CATALOG descriptions and SEMANTIC FIELD GUIDANCE to decide what qualifies as "
        "an instance for each path. Do not treat generic section headings or layout labels as entity instances "
        "unless the schema guidance explicitly describes them as such.\n"
        "14. For list-entity paths that have identity examples in the catalog (e.g. paths ending with []): "
        "emit nodes ONLY when this batch clearly contains the corresponding document structure (e.g. guarantee table, "
        "formula names). If this batch contains only a sommaire, section headings, or article titles, emit ZERO nodes "
        "for that path."
    )

    user_prompt = (
        f"[Batch {batch_index + 1}/{total_batches} — for context only; do not put this into any field.]\n\n"
        "=== BATCH DOCUMENT ===\n"
        f"{batch_markdown}\n"
        "=== END BATCH DOCUMENT ===\n\n"
        "=== TEMPLATE PATH CATALOG ===\n"
        f"{path_catalog_block}\n"
        "=== END CATALOG ===\n\n"
        "=== SEMANTIC FIELD GUIDANCE ===\n"
        f"{schema_semantic_guide}\n"
        "=== END GUIDANCE ===\n\n"
        "Important: Use each catalog line's ids=[...] as the required identity contract for that path. "
        'Identity values must be strings (e.g. line_number: "1" not 1). '
        'Parent must be an object: {"path": "<catalog path>", "ids": {}} or null for root.\n'
        "Important: Use schema-derived descriptions/examples in the catalog/guidance to decide entity "
        "membership for each path; when uncertain, omit instead of classifying from heading style alone.\n"
        'Example good root scalar placement: node path="" with properties.<root_field>=<scalar_value> '
        '(NOT a node path "<root_field>").\n'
        'Example good list-entity placement: node path="<list_entity_path>" with ids.<id_field>="..." and '
        "properties.<entity_field>=<scalar_value>.\n\n"
        'Return JSON: {"nodes": [...], "relationships": [...]} where each node contains '
        "{path, node_type?, ids, parent, properties}."
    )

    return {"system": system_prompt, "user": user_prompt}


def format_batch_markdown(chunks: Sequence[str]) -> str:
    """Join chunk payloads with stable delimiters for one LLM batch call."""

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks):
        blocks.append(f"--- CHUNK {idx + 1} ---\n{chunk}")
    return "\n\n".join(blocks)
