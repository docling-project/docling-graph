"""Dense extraction prompts: skeleton (Phase 1) and fill (Phase 2)."""

from __future__ import annotations

import json
from typing import Any

from ....utils.doclang_format import prompt_framing
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
    input_format: str = "markdown",
    coverage_retry: bool = False,
) -> dict[str, str]:
    """Build system and user prompts for one skeleton (Phase 1) batch.

    ``coverage_retry=True`` marks a second-pass batch over document sections
    that produced no entities on the first pass; the prompt says so explicitly
    and licenses an empty response, so the retry cannot pressure the model
    into inventing instances.
    """
    framing = prompt_framing(input_format)
    framing_block = f"{framing}\n\n" if framing else ""
    # Rule 2 mentions negative reference handles ONLY when an ALREADY EXTRACTED
    # list is actually present: on the first batch (and single-batch documents)
    # the extra clause has nothing to refer to, and on small models it was
    # observed to dilute the core "p must reference a handle in this response"
    # instruction badly enough that p was omitted entirely.
    rule2_cross_batch = (
        ' — or, when the parent was already extracted in an earlier batch, the NEGATIVE handle shown for it in the ALREADY EXTRACTED list (e.g. "p": -3). Re-emitting an already-extracted parent with the same ids also works; duplicates are merged automatically'
        if already_found
        else ""
    )
    # Rule 11 exists only for multi-level catalogs (some path nested below a
    # depth-1 parent): on flat catalogs every non-root node hangs off the root
    # and the sentence would be noise (rule-2-dilution lesson).
    has_nested_paths = any("." in p for p in allowed_paths)
    rule11_nested_parent = (
        "\n11. A nested entity's parent is the SPECIFIC entity instance whose section or "
        'article names it (e.g. items listed under a section titled with entity X belong to X): set "p" to that '
        "instance's handle (or negative handle), NEVER to the root and NEVER to a node made from a document or section title."
        if has_nested_paths
        else ""
    )
    system_prompt = (
        f"{framing_block}"
        "You are an expert extraction engine for document structure. "
        "Your task is to identify every distinct entity instance per catalog path. "
        "Do NOT extract any data fields, measurements, or property values. "
        "Scope boundary: Extract ONLY entities that are the primary subject, direct output, or original creation of the provided document. Do NOT extract external references, cited works, historical examples, or third-party entities that are mentioned only for context or background.\n\n"
        "When identifying entities from the schema, you must extract both localized and global objects: "
        "(1) Localized entities: items tied to distinct identifiers in the text (e.g. specific section headers, figures, tables, or line items). Create separate, distinct instances for each identifier found. "
        "(2) Global / shared entities: singleton items that apply broadly across the document or serve as shared references for other nodes (e.g. overarching policies, global configurations, or general methodologies). Extract these global entities even if they lack a specific localized identifier or label. "
        "Do not ignore an entity just because it lacks a distinct sub-label; if the schema defines it and the text describes it, it must be included in the skeleton. "
        "Each data row of a table is a separate entity instance; document-level metadata (titles, dates, totals, summary rows) is not.\n\n"
        "Rules:\n"
        '1. Use ONLY the catalog paths listed. Each node has exactly: "i" (its handle: a sequential integer starting at 1, unique in this response), "path", "ids" (identifier values from the document), "p" (the handle of its parent node in this response; omit or null for the root), and optionally "c" (the number N of the "--- CHUNK N ---" marker the entity was found under).\n'
        '2. Emit the root (path "", no parent) exactly once, first. Every other node\'s "p" must reference the handle of a node in this same response whose path is its parent path in the catalog'
        f"{rule2_cross_batch}.\n"
        '3. ids values are short labels copied verbatim from the document — a code, number, or name of at most a few words, never a sentence or description. Every ids entry MUST be a complete "field": "value" pair (e.g. {"name": "Gardenwork"}); NEVER emit a bare value without its field name. Use figure labels, table rows, section titles that name entities (e.g. FIG-4, Sample A, Protocol 1). Do not use generic section/chapter titles as identities for localized entities. For global/singleton entities (e.g. one protocol or one setup for the whole document), use a short descriptive id from the text or section (e.g. General Protocol) or ids={} if the schema allows; ensure at least one root and such singletons are still emitted.\n'
        '4. Always prefer the most specific proper name that the document gives an entity over a generic category word or a positional label. If a set of items each carry their own name, use those names as ids. For example, when a document lists offers named ESSENTIELLE, CONFORT and CONFORT PLUS, the ids must be "ESSENTIELLE", "CONFORT", "CONFORT PLUS" — NOT the generic word "Offer"/"Offre" nor positional labels like "Offer 1", "Offer 2". Only fall back to a generic or positional label when the document truly gives the entity no name of its own.\n'
        "5. For list-entity paths (e.g. studies[], experiments[]), emit one node per distinct instance.\n"
        '6. Output valid JSON only: {"nodes": [{"i": 1, "path": "", "ids": {}}, {"i": 2, "path": "...", "ids": {"...": "..."}, "p": 1, "c": 3}]}. No other fields.\n'
        "7. When identifying container nodes (nodes that have children with identity fields, e.g. Dataset with Curves), create separate instances if the contained data comes from distinct sources (e.g. different Figure numbers, different Tables) or represents distinct conditions. If child identifiers differ (e.g. Figure 2a vs Figure 7c), create separate parent instances so each logical group has its own container.\n"
        "8. Keep entities that play different roles distinct even when described together, and use an entity's actual name as its identity, never surrounding text such as address fragments.\n"
        "9. When several catalog paths could host an instance, choose the path whose description "
        "(see SEMANTIC FIELD GUIDANCE) matches the text; never place an instance at a path whose "
        "description does not fit it, and never invent instances for paths the text does not mention.\n"
        '10. Each node names exactly ONE entity. NEVER join two entity names into a single id (e.g. "Guarantee X, Offer Y" from a table row and column): a table cell linking two entities is a RELATIONSHIP between existing nodes, not a new entity — relationships are captured in a later phase. When a table cross-references entities, emit each row entity and each column entity once.'
        f"{rule11_nested_parent}"
    )
    user_prompt = f"[Batch {batch_index + 1}/{total_batches}]\n\n"
    if coverage_retry:
        user_prompt += (
            "NOTE: These document sections produced no entities on the first "
            "pass. Look again carefully and extract every catalog-path instance "
            "they contain; if they truly contain none, return "
            '{"nodes": []} — never invent instances.\n\n'
        )
    if already_found:
        user_prompt += (
            "=== ALREADY EXTRACTED (negative reference handles) — DO NOT RE-OUTPUT ===\n"
            f"{already_found}\n"
            "=== END ===\n\n"
            "The entities above are already captured; each line shows its reference handle "
            '"i" (a negative number). Do NOT include any of them in your output — emit ONLY '
            "entities that are NEW in this batch. When a NEW node's parent is one of the "
            'entities above, set the new node\'s "p" to that negative handle (e.g. "p": -3); '
            "no re-emission is needed. Re-emitting a listed entity wastes output budget and "
            "risks truncating the response.\n\n"
        )
    if global_context:
        user_prompt += f"=== DOCUMENT CONTEXT ===\n{global_context}\n=== END ===\n\n"
    user_prompt += (
        "=== BATCH DOCUMENT ===\n"
        f"{batch_markdown}\n"
        "=== END ===\n\n"
        "=== CATALOG (use only these paths) ===\n"
        f"{catalog_block}\n"
        "=== END CATALOG ===\n\n"
    )
    if semantic_guide:
        user_prompt += f"=== SEMANTIC FIELD GUIDANCE ===\n{semantic_guide}\n=== END ===\n\n"
    user_prompt += 'List every distinct entity instance in this batch. Return JSON: {"nodes": [...]} with each node having i, path, ids, p, and optionally c.'
    return {"system": system_prompt, "user": user_prompt}


def get_skeleton_reconciliation_prompt(
    instances_by_path: dict[str, list[dict[str, Any]]],
    candidate_groups: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Build prompts for the post-merge skeleton reconciliation pass.

    Pure id-space: the LLM sees only per-path instance identifier lists (no
    document) and returns groups of instances that are aliases of the same
    real-world entity at different granularities. ``candidate_groups`` are
    mechanical containment proposals (see resolvers.propose_containment_groups)
    the model must explicitly confirm or reject — they are suggestions, never
    pre-approved merges.
    """
    system_prompt = (
        "You deduplicate entity instance lists extracted from one document. "
        "For each path you receive the numbered identifier sets of the discovered instances. "
        "Group instances ONLY when they clearly refer to the SAME real-world entity — "
        "typically a generic or shorthand alias alongside its more specific form. "
        "NEVER group instances that differ by any parameter, quantity, concentration, condition, "
        "date, version, figure/table number, or index: those are distinct entities. "
        "Tier or variant names where one name extends the other with a qualifier word "
        "(e.g. 'CONFORT' vs 'CONFORT PLUS', 'Standard' vs 'Standard Pro') denote DIFFERENT "
        "offerings — never merge them; only merge a longer form that is merely a more "
        "descriptive way of writing the SAME entity. "
        "When in doubt, do not merge. Groups must stay within one path. "
        'For each group, "keep" is the number of the most specific instance and "merge" lists the '
        "numbers of its aliases. "
        'Return JSON only: {"merges": [{"path": "...", "keep": 0, "merge": [1, 2]}]}. '
        'Return {"merges": []} when nothing should be merged.'
    )
    blocks: list[str] = []
    for path, instances in instances_by_path.items():
        lines = [f"=== PATH {path} ==="]
        for idx, ids in enumerate(instances):
            lines.append(f"{idx}: {json.dumps(ids, ensure_ascii=False, default=str)}")
        blocks.append("\n".join(lines))
    user_prompt = "\n\n".join(blocks)
    if candidate_groups:

        def _ids_text(instances: list[dict[str, Any]], idx: Any) -> str:
            if isinstance(idx, int) and 0 <= idx < len(instances):
                return json.dumps(instances[idx], ensure_ascii=False, default=str)
            return "?"

        lines = [
            "=== CONTAINMENT CANDIDATES (mechanical substring matches — verify each) ===",
            "Include a candidate in your merges ONLY if the identifiers denote the same "
            "real-world entity; reject tier/variant pairs.",
        ]
        for group in candidate_groups:
            group_path = group.get("path")
            keep = group.get("keep")
            merge = group.get("merge") or []
            instances = instances_by_path.get(group_path, []) if isinstance(group_path, str) else []
            merged_txt = ", ".join(f"{m} ({_ids_text(instances, m)})" for m in merge)
            lines.append(
                f"- {group_path}: {merged_txt} may alias {keep} ({_ids_text(instances, keep)})"
            )
        user_prompt += "\n\n" + "\n".join(lines)
    user_prompt += '\n\nIdentify alias groups. Return JSON: {"merges": [...]} (empty list if none).'
    return {"system": system_prompt, "user": user_prompt}


def reconciliation_output_schema() -> dict[str, Any]:
    """JSON schema for the reconciliation pass output."""
    return {
        "type": "object",
        "properties": {
            "merges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "keep": {"type": "integer"},
                        "merge": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["path", "keep", "merge"],
                },
            }
        },
        "required": ["merges"],
    }


def get_fill_batch_prompt(
    *,
    markdown: str,
    path: str,
    spec: NodeSpec,
    descriptors: list[dict[str, Any]],
    projected_schema_json: str,
    input_format: str = "markdown",
    has_reference_fields: bool = False,
) -> dict[str, str]:
    """Build system and user prompts for one fill (Phase 2) batch.

    ``has_reference_fields=True`` adds the per-instance membership rule for
    id-only reference lists (such paths are filled one instance per call, so
    "THIS instance" is unambiguous).
    """
    n = len(descriptors)
    # Every instance id must be listed: the response is matched back to
    # descriptors by position, so the LLM needs the complete ordered list.
    instances_preview = json.dumps([d.get("ids") or {} for d in descriptors], indent=2, default=str)
    framing = prompt_framing(input_format)
    framing_block = f"{framing}\n\n" if framing else ""
    membership_rule = (
        " Reference-list fields (id-only lists naming related entities) must contain ONLY "
        "the memberships the document explicitly states for THIS specific instance; when the "
        "document does not state them for this instance, leave the list empty — never copy "
        "another instance's memberships or the union of a whole table."
        if has_reference_fields
        else ""
    )
    system_prompt = (
        f"{framing_block}"
        "You are a precise extraction assistant. For each of the given node instances, "
        "fill all fields from the document according to the JSON schema. "
        'Return a single JSON object with key "items" containing an array of filled objects, one per instance in the same order. '
        'Return the filled objects in the exact order they were requested: the first item in "items" must correspond to the first instance identifier, the second to the second, and so on. '
        "Use the document to extract real values; do not invent data. "
        "Preserve identifier values (ids) in each object when the schema includes them. "
        "Assign each value to the exact schema field it belongs to; never place fragments of one field into another (e.g. address parts into a name field). "
        "Copy numeric values digit-for-digit from the document; never compute, round, or aggregate them. "
        "Values from table summary rows (totals, subtotals) belong to document-level fields, never to row-level instances. "
        "Omit values that are not present in the document rather than guessing."
        f"{membership_rule}"
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
