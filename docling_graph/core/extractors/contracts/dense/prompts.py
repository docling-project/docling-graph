"""Dense extraction prompts: skeleton (Phase 1) and fill (Phase 2)."""

from __future__ import annotations

import json
from typing import Any

from .catalog import NodeCatalog, NodeSpec


def format_batch_markdown(chunks: list[tuple[int, str, int]]) -> str:
    """Format chunk list for skeleton batch user message."""
    blocks: list[str] = []
    for idx, chunk, _ in chunks:
        blocks.append(f"--- CHUNK {idx + 1} ---\n{chunk}")
    return "\n\n".join(blocks)


def build_skeleton_catalog_block(catalog: NodeCatalog) -> str:
    """Build catalog text for skeleton prompt: paths and id_fields only."""
    lines: list[str] = []
    for spec in catalog.nodes:
        path_label = '""' if spec.path == "" else spec.path
        ids_label = ", ".join(spec.id_fields) if spec.id_fields else "none (use ids={})"
        lines.append(f"- {path_label} ({spec.node_type}) ids=[{ids_label}]")
    return "\n".join(lines)


def get_skeleton_batch_prompt(
    *,
    batch_markdown: str,
    catalog_block: str,
    batch_index: int,
    total_batches: int,
    allowed_paths: list[str],
    global_context: str | None = None,
    already_found: str | None = None,
    semantic_guide: str | None = None,
) -> dict[str, str]:
    """Build system and user prompts for one skeleton (Phase 1) batch."""
    system_prompt = (
        "You are an expert extraction engine for document structure. "
        "Your task is to identify every distinct entity instance per catalog path. "
        "Do NOT extract any data fields, measurements, or property values. "
        "Scope boundary: Extract ONLY entities that are the primary subject, direct output, or original creation of the provided document. Do NOT extract external references, cited works, historical examples, or third-party entities that are mentioned only for context or background.\n\n"
        "When identifying entities from the schema, you must extract both localized and global objects: "
        "(1) Localized entities: items tied to distinct identifiers in the text (e.g. specific section headers, figures, tables, or line items). Create separate, distinct instances for each identifier found. "
        "(2) Global / shared entities: singleton items that apply broadly across the document or serve as shared references for other nodes (e.g. overarching policies, global configurations, or general methodologies). Extract these global entities even if they lack a specific localized identifier or label. "
        "Do not ignore an entity just because it lacks a distinct sub-label; if the schema defines it and the text describes it, it must be included in the skeleton.\n\n"
        "Rules:\n"
        "1. Use ONLY the catalog paths listed. Each node must have: path, ids (identifier values from the document), and parent (path + ids of parent, or null for root).\n"
        "2. For every node except the root, you **MUST** provide the ancestry array. This array must list the exact {path, ids} of every ancestor from the root down to the immediate parent. Do not use the parent field alone; you must prove the lineage via ancestry. Root nodes have empty ancestry or omit ancestry.\n"
        "3. Set ids from the document: figure labels, table rows, section titles that name entities (e.g. FIG-4, Sample A, Protocol 1). Do not use generic section/chapter titles as identities for localized entities. For global/singleton entities (e.g. one protocol or one setup for the whole document), use a short descriptive id from the text or section (e.g. Materials and Methods, General Protocol) or ids={} if the schema allows; ensure at least one root and such singletons are still emitted.\n"
        "4. For list-entity paths (e.g. studies[], experiments[]), emit one node per distinct instance. Parent must reference the parent path and its ids.\n"
        '5. Output valid JSON only: {"nodes": [{"path": "...", "ids": {...}, "parent": null|{"path": "...", "ids": {...}}, "ancestry": [...]}]}. Do not include a "properties" field.\n'
        '6. Root path is "" (empty string); its parent must be null.\n'
        "7. When identifying container nodes (nodes that have children with identity fields, e.g. Dataset with Curves), create separate instances if the contained data comes from distinct sources (e.g. different Figure numbers, different Tables) or represents distinct conditions. If child identifiers differ (e.g. Figure 2a vs Figure 7c), create separate parent instances so each logical group has its own container."
    )
    user_prompt = f"[Batch {batch_index + 1}/{total_batches}]\n\n"
    if already_found:
        user_prompt += (
            "=== ALREADY EXTRACTED (do not duplicate) ===\n"
            f"{already_found}\n"
            "=== END ===\n\n"
            "Extract ADDITIONAL node instances not already listed above.\n\n"
        )
    if global_context:
        user_prompt += (
            "=== DOCUMENT CONTEXT ===\n"
            f"{global_context}\n"
            "=== END ===\n\n"
        )
    user_prompt += (
        "=== BATCH DOCUMENT ===\n"
        f"{batch_markdown}\n"
        "=== END ===\n\n"
        "=== CATALOG (use only these paths) ===\n"
        f"{catalog_block}\n"
        "=== END CATALOG ===\n\n"
    )
    if semantic_guide:
        user_prompt += (
            "=== SEMANTIC FIELD GUIDANCE ===\n"
            f"{semantic_guide}\n"
            "=== END ===\n\n"
        )
    user_prompt += (
        'List every distinct entity instance in this batch. Return JSON: {"nodes": [...]} with each node having path, ids, and parent only.'
    )
    return {"system": system_prompt, "user": user_prompt}


def get_fill_batch_prompt(
    *,
    markdown: str,
    path: str,
    spec: NodeSpec,
    descriptors: list[dict[str, Any]],
    projected_schema_json: str,
) -> dict[str, str]:
    """Build system and user prompts for one fill (Phase 2) batch."""
    n = len(descriptors)
    preview = [d.get("ids") or {} for d in descriptors[:5]]
    instances_preview = json.dumps(preview, indent=2, default=str)
    if n > 5:
        instances_preview += f"\n... and {n - 5} more"
    system_prompt = (
        "You are a precise extraction assistant. For each of the given node instances, "
        "fill all fields from the document according to the JSON schema. "
        'Return a single JSON object with key "items" containing an array of filled objects, one per instance in the same order. '
        'Return the filled objects in the exact order they were requested: the first item in "items" must correspond to the first instance identifier, the second to the second, and so on. '
        "Use the document to extract real values; do not invent data. "
        "Preserve identifier values (ids) in each object when the schema includes them."
    )
    user_prompt = (
        "=== DOCUMENT ===\n"
        f"{markdown[:120000]}\n"
        "=== END ===\n\n"
        f"=== NODE PATH: {path} ({spec.node_type}) ===\n"
        f"Instance identifiers to fill ({n} instances):\n"
        f"{instances_preview}\n\n"
        "=== SCHEMA (fill all fields per instance) ===\n"
        f"{projected_schema_json}\n"
        "=== END SCHEMA ===\n\n"
        'Return JSON: {"items": [<filled object 1>, <filled object 2>, ...]} with one object per instance in the exact order listed above.'
    )
    return {"system": system_prompt, "user": user_prompt}
