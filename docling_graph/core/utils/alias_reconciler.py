"""
Graph-level alias reconciliation (runs for every extraction contract).

Documents routinely name the same entity two ways — a short label in a table
("Attentat") and a long section header where it is described in full
("Attentat et actes de terrorisme"). Identity-based node IDs are exact after
case/diacritic canonicalization, so the two mentions become two nodes and the
graph's hub splits: membership edges land on the short-name node while the
detail lives on the long-name one.

This module proposes same-class alias candidates deterministically
(canonical-containment with a digit-signature guard) and lets an id-space LLM
call confirm or reject each one. Confirmation is mandatory for merging:
containment alone cannot tell an alias from a product tier ("CONFORT PLUS" is
NOT "CONFORT"). Without an LLM callable the pass is propose-only — candidates
are logged, nothing is merged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import networkx as nx

from ...logging_utils import get_component_logger
from ..provenance.identity import PROVENANCE_NODE_ATTR, merge_compact_views
from .entity_name_normalizer import canonicalize_identity_for_dedup

logger = get_component_logger("AliasReconciler", __name__)

_DIGIT_RUNS = re.compile(r"\d+")

# A containment base shorter than this is too common to be a safe alias signal.
_MIN_CONTAINMENT_LEN = 4

# Framework-owned node attributes; never treated as content when choosing the
# attribute-richer survivor or copying attributes across.
_META_ATTRS = {"id", "label", "type", "__class__", PROVENANCE_NODE_ATTR, "merged_aliases"}


def digit_signature(text: str) -> tuple[str, ...]:
    """Ordered digit runs in an identifier ("LFP_20vol" -> ("20",))."""
    return tuple(_DIGIT_RUNS.findall(text))


def containment_groups(texts: list[str]) -> dict[int, list[int]]:
    """Map base index -> superset indexes for canonical-containment pairs.

    Guards (shared by the dense skeleton pass and the graph-level pass):
    equal digit signatures, a unique base per superset (two matching shorter
    ids are ambiguous -> skipped), bases at least _MIN_CONTAINMENT_LEN chars.
    Empty texts never participate.
    """
    signatures = [digit_signature(text) for text in texts]
    merge_by_keep: dict[int, list[int]] = {}
    for j, superset_text in enumerate(texts):
        if not superset_text:
            continue
        bases = [
            i
            for i, base_text in enumerate(texts)
            if i != j
            and base_text
            and len(base_text) >= _MIN_CONTAINMENT_LEN
            and len(base_text) < len(superset_text)
            and base_text in superset_text
            and signatures[i] == signatures[j]
        ]
        if len(bases) == 1:
            merge_by_keep.setdefault(bases[0], []).append(j)
    return merge_by_keep


def id_fields_by_class(models: list[Any]) -> dict[str, list[str]]:
    """Collect {class name: graph_id_fields} from every model reachable below
    ``models`` (entities and components alike; classes without id fields map
    to an empty list and are ignored by the reconciler)."""
    result: dict[str, list[str]] = {}
    seen: set[int] = set()

    def _visit(instance: Any) -> None:
        if id(instance) in seen or not hasattr(instance, "model_fields"):
            return
        seen.add(id(instance))
        config = getattr(instance, "model_config", {}) or {}
        raw = config.get("graph_id_fields", []) if isinstance(config, dict) else []
        result.setdefault(instance.__class__.__name__, [f for f in raw if isinstance(f, str)])
        for _name, value in instance:
            if hasattr(value, "model_fields"):
                _visit(value)
            elif isinstance(value, list):
                for item in value:
                    if hasattr(item, "model_fields"):
                        _visit(item)

    for model in models:
        _visit(model)
    return result


def _id_texts(node_data: dict[str, Any], id_fields: list[str]) -> tuple[str, str]:
    """(canonical text, human-readable display) of a node's identity values."""
    canonical = "".join(
        canonicalize_identity_for_dedup(field, node_data.get(field))
        for field in id_fields
        if node_data.get(field) is not None
    )
    display = " / ".join(
        str(node_data.get(field)) for field in id_fields if node_data.get(field) not in (None, "")
    )
    return canonical, display


def propose_alias_candidates(
    graph: nx.DiGraph,
    id_fields_map: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]]]:
    """Deterministic candidate proposal over graph nodes.

    Returns (node_ids_by_class, display_by_class, candidate_groups) where each
    group is {"class": cls, "keep": base_index, "merge": [superset_indexes]}
    with indexes into the per-class lists.
    """
    node_ids_by_class: dict[str, list[str]] = {}
    display_by_class: dict[str, list[str]] = {}
    texts_by_class: dict[str, list[str]] = {}
    for node_id, node_data in graph.nodes(data=True):
        cls = str(node_data.get("__class__") or "")
        fields = id_fields_map.get(cls) or []
        if not fields:
            continue
        canonical, display = _id_texts(node_data, fields)
        node_ids_by_class.setdefault(cls, []).append(node_id)
        display_by_class.setdefault(cls, []).append(display)
        texts_by_class.setdefault(cls, []).append(canonical)

    groups: list[dict[str, Any]] = []
    for cls, texts in texts_by_class.items():
        if len(texts) < 2:
            continue
        for keep, merge in sorted(containment_groups(texts).items()):
            groups.append({"class": cls, "keep": keep, "merge": merge})
    return node_ids_by_class, display_by_class, groups


def get_alias_reconciliation_prompt(
    display_by_class: dict[str, list[str]],
    candidate_groups: list[dict[str, Any]],
) -> dict[str, str]:
    """Id-space confirmation prompt (no document content), one call per graph."""
    system_prompt = (
        "You deduplicate entity lists extracted from one document into a knowledge graph. "
        "For each entity class you receive the numbered identifier of every node, plus "
        "mechanical containment candidates (one identifier contained in another). "
        "Confirm a candidate ONLY when both identifiers clearly denote the SAME real-world "
        "entity — typically a short table label alongside the full section title of the same "
        "thing. NEVER merge instances that differ by any parameter, quantity, date, version "
        "or index. Tier or variant names where one name extends the other with a qualifier "
        "word (e.g. 'CONFORT' vs 'CONFORT PLUS', 'Standard' vs 'Standard Pro') denote "
        "DIFFERENT offerings — never merge them. When in doubt, do not merge. "
        'Return JSON only: {"merges": [{"class": "...", "keep": 0, "merge": [1]}]} using the '
        'numbers shown; return {"merges": []} when nothing should be merged.'
    )
    classes_with_candidates = {group["class"] for group in candidate_groups}
    blocks: list[str] = []
    for cls in sorted(classes_with_candidates):
        lines = [f"=== CLASS {cls} ==="]
        for idx, display in enumerate(display_by_class.get(cls, [])):
            lines.append(f"{idx}: {display}")
        blocks.append("\n".join(lines))

    def _show(displays: list[str], idx: int) -> str:
        return repr(displays[idx]) if 0 <= idx < len(displays) else "?"

    candidate_lines = ["=== CONTAINMENT CANDIDATES (verify each; reject tier/variant pairs) ==="]
    for group in candidate_groups:
        displays = display_by_class.get(group["class"], [])
        merged_txt = ", ".join(f"{m} ({_show(displays, m)})" for m in group["merge"])
        candidate_lines.append(
            f"- {group['class']}: {merged_txt} may alias "
            f"{group['keep']} ({_show(displays, group['keep'])})"
        )
    user_prompt = (
        "\n\n".join(blocks)
        + "\n\n"
        + "\n".join(candidate_lines)
        + '\n\nConfirm true alias groups. Return JSON: {"merges": [...]} (empty list if none).'
    )
    return {"system": system_prompt, "user": user_prompt}


def alias_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "merges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string"},
                        "keep": {"type": "integer"},
                        "merge": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["class", "keep", "merge"],
                },
            }
        },
        "required": ["merges"],
    }


def _attr_richness(node_data: dict[str, Any]) -> int:
    """Count of meaningfully filled content attributes."""
    count = 0
    for key, value in node_data.items():
        if key in _META_ATTRS or value is None:
            continue
        if isinstance(value, (str, list, dict)) and not value:
            continue
        count += 1
    return count


def _merge_nodes(graph: nx.DiGraph, survivor: str, merged: str) -> None:
    """Fold ``merged`` into ``survivor``: fill empty attributes, union
    provenance, record the alias, redirect edges, remove the node."""
    survivor_data = graph.nodes[survivor]
    merged_data = graph.nodes[merged]
    for key, value in merged_data.items():
        if key in _META_ATTRS or value in (None, "", [], {}):
            continue
        if survivor_data.get(key) in (None, "", [], {}):
            survivor_data[key] = value
    merged_view = merge_compact_views(
        survivor_data.get(PROVENANCE_NODE_ATTR), merged_data.get(PROVENANCE_NODE_ATTR)
    )
    if merged_view is not None:
        survivor_data[PROVENANCE_NODE_ATTR] = merged_view
    aliases = survivor_data.setdefault("merged_aliases", [])
    alias_ref = {
        "id": merged,
        **{
            k: v
            for k, v in merged_data.items()
            if k not in _META_ATTRS and isinstance(v, str) and v
        },
    }
    aliases.append(alias_ref)
    for source, _target, edge_data in list(graph.in_edges(merged, data=True)):
        if source != survivor:
            graph.add_edge(source, survivor, **edge_data)
    for _source, target, edge_data in list(graph.out_edges(merged, data=True)):
        if target != survivor:
            graph.add_edge(survivor, target, **edge_data)
    graph.remove_node(merged)


def reconcile_graph_aliases(
    graph: nx.DiGraph,
    id_fields_map: dict[str, list[str]],
    llm_call_fn: Callable[..., Any] | None = None,
    context: str = "graph",
) -> dict[str, Any]:
    """Propose, confirm, and apply same-class alias merges on a built graph.

    Only pairs the deterministic proposer produced AND the LLM confirmed are
    merged (the intersection): the proposer's guards (digit signature, unique
    base) bound what the LLM may do, and the LLM's judgment protects tiers the
    guards cannot see. The attribute-richer node survives; the merged node's
    identity is recorded on the survivor under ``merged_aliases``.
    """
    stats = {"candidates": 0, "confirmed": 0, "merged": 0}
    node_ids_by_class, display_by_class, groups = propose_alias_candidates(graph, id_fields_map)
    stats["candidates"] = sum(len(g["merge"]) for g in groups)
    if not groups:
        return stats
    proposed_pairs = {(g["class"], g["keep"], m) for g in groups for m in g["merge"]}
    if llm_call_fn is None:
        preview = "; ".join(
            f"{g['class']}: {display_by_class[g['class']][g['keep']]!r} ~ "
            + ", ".join(repr(display_by_class[g["class"]][m]) for m in g["merge"])
            for g in groups[:5]
        )
        logger.info(
            "Alias reconciliation: %s candidate pair(s) proposed but no LLM available "
            "to confirm; nothing merged. Candidates: %s",
            stats["candidates"],
            preview,
        )
        return stats

    prompt = get_alias_reconciliation_prompt(display_by_class, groups)
    try:
        out = llm_call_fn(
            prompt=prompt,
            schema_json=json.dumps(alias_output_schema()),
            context=f"{context}_alias_reconcile",
        )
    except Exception as e:  # cleanup must never break conversion
        logger.warning("Alias reconciliation call failed: %s; nothing merged", e)
        return stats
    merges = out.get("merges") if isinstance(out, dict) else None
    if not isinstance(merges, list):
        return stats

    removed: set[str] = set()
    for group in merges:
        if not isinstance(group, dict):
            continue
        cls = group.get("class")
        keep_idx = group.get("keep")
        merge_idxs = group.get("merge")
        node_ids = node_ids_by_class.get(cls) if isinstance(cls, str) else None
        if node_ids is None or not isinstance(keep_idx, int) or not isinstance(merge_idxs, list):
            continue
        if not (0 <= keep_idx < len(node_ids)):
            continue
        for merge_idx in merge_idxs:
            if not isinstance(merge_idx, int) or not (0 <= merge_idx < len(node_ids)):
                continue
            # Guard rail: only pairs the deterministic proposer produced are
            # eligible (in either orientation) — a hallucinated pairing of two
            # unrelated instances is silently ignored.
            if (cls, keep_idx, merge_idx) not in proposed_pairs and (
                cls,
                merge_idx,
                keep_idx,
            ) not in proposed_pairs:
                continue
            stats["confirmed"] += 1
            node_a, node_b = node_ids[keep_idx], node_ids[merge_idx]
            if node_a in removed or node_b in removed or node_a == node_b:
                continue
            if node_a not in graph or node_b not in graph:
                continue
            richness_a = _attr_richness(graph.nodes[node_a])
            richness_b = _attr_richness(graph.nodes[node_b])
            survivor, merged = (node_a, node_b) if richness_a >= richness_b else (node_b, node_a)
            _merge_nodes(graph, survivor, merged)
            removed.add(merged)
            stats["merged"] += 1
            logger.info("Alias merge: %s absorbed into %s", merged, survivor)
    if stats["merged"]:
        logger.info(
            "Alias reconciliation: merged %s node(s) (%s candidate pair(s), %s confirmed)",
            stats["merged"],
            stats["candidates"],
            stats["confirmed"],
        )
    return stats
