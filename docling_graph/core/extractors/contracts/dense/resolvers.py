"""
Post-merge resolvers for dense skeleton nodes.

Two mechanisms live here:

- ``propose_containment_groups``: deterministic *proposal* of same-path alias
  groups (one canonical id contained in another). Proposals are confirmed or
  rejected by the reconciliation LLM call — containment alone cannot tell a
  refinement of the same entity ("nanoscaled LiFePO4" ~ "LiFePO4") from a
  distinct product tier ("CONFORT PLUS" is NOT "CONFORT"), so it never merges
  on its own.
- ``resolve_skeleton_nodes``: fuzzy same-path merge for OCR noise (casing,
  dropped accents, ligature splits). Aggressive dedupe mode only; the
  similarity threshold is internal — it is a property of how OCR noise looks,
  not something users should tune per run.

Fully autonomous: no imports from other contracts.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Callable

from docling_graph.core.utils.alias_reconciler import containment_groups
from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

logger = logging.getLogger(__name__)

# Similarity above which two same-path identifier strings are considered the
# same real-world entity (OCR noise: dropped accents, punctuation, spacing).
_FUZZY_MERGE_THRESHOLD = 0.9

_DIGIT_RUNS = re.compile(r"\d+")


def _digit_signature(text: str) -> tuple[str, ...]:
    """Ordered digit runs in an identifier ("LFP_20vol" -> ("20",))."""
    return tuple(_DIGIT_RUNS.findall(text))


def _concat_ids(node: dict[str, Any]) -> str:
    """Build a single string from node ids for similarity comparison."""
    ids = node.get("ids")
    if not isinstance(ids, dict):
        return ""
    return " | ".join(str(v).strip() for v in ids.values() if v is not None and str(v).strip())


def _fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_sort_ratio(a, b)) / 100.0
    except Exception:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parent_key(parent: dict[str, Any] | None, key_fn: Callable[[dict], Any]) -> Any | None:
    """Return the identity key for a parent dict, or None if no parent."""
    if parent is None or not isinstance(parent, dict):
        return None
    return key_fn({"path": parent.get("path"), "ids": parent.get("ids") or {}})


def _serialize_key(key: Any) -> str | None:
    """Serialize a (path, pairs) identity tuple into the ledger key format."""
    try:
        path, pairs = key
        if any(field == "__key" for field, _ in pairs):
            return None
        return f"{path}|" + ",".join(f"{field}={value}" for field, value in pairs)
    except Exception:
        return None


def _absorb_sources(keeper: dict[str, Any], merged: dict[str, Any], merged_key: Any) -> None:
    """Union the merged node's provenance bookkeeping into the keeper."""
    for list_key in ("_source_batch_indexes", "_source_chunk_ids"):
        target = keeper.setdefault(list_key, [])
        for value in merged.get(list_key) or []:
            if value not in target:
                target.append(value)
    serialized = _serialize_key(merged_key)
    if serialized is not None:
        merged_from = keeper.setdefault("_merged_from", [])
        if serialized not in merged_from:
            merged_from.append(serialized)


def _canonical_ids_text(ids: Any) -> str:
    """Concatenated canonical text of an instance's identifier values.

    Uses the same normalization dense dedup uses, so containment comparison is
    case/diacritic-insensitive. Returns "" when there is nothing usable.
    """
    if not isinstance(ids, dict):
        return ""
    return "".join(
        canonicalize_identity_for_dedup(str(field), value)
        for field, value in sorted(ids.items(), key=lambda kv: str(kv[0]))
        if value is not None
    )


def propose_containment_groups(
    instances_by_path: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Propose same-path alias groups where one canonical id contains another.

    PROPOSAL ONLY — nothing is merged here. A more specific descriptive id
    ("nanoscaled LiFePO4 (LFP)") often names the same real-world entity as its
    base ("LiFePO4 (LFP)"), which case/diacritic dedup and the fuzzy resolver
    both miss. But containment alone cannot distinguish that case from a
    distinct product tier ("CONFORT PLUS" vs "CONFORT"), so the groups are
    handed to the reconciliation LLM call, which confirms or rejects each one.

    Group indices refer to the per-path instance order of ``instances_by_path``
    (the same numbering the reconciliation prompt shows). Deterministic guards:

    - same catalog path only;
    - equal digit signatures (so "LFP_20vol"/"LFP_30vol" and "Article 5"/"6"
      are never even proposed);
    - a unique base: a superset containing *two* different shorter ids is
      ambiguous and skipped;
    - very short bases are ignored (substrings that short are too common to
      be a signal); see utils.alias_reconciler.containment_groups.

    ``keep`` points at the shorter base id — the canonical term and the most
    verbatim-friendly locator; the confirming LLM may still restructure.
    """
    groups: list[dict[str, Any]] = []
    for path, ids_list in instances_by_path.items():
        texts = [_canonical_ids_text(ids) for ids in ids_list]
        for keep, merge in sorted(containment_groups(texts).items()):
            groups.append({"path": path, "keep": keep, "merge": merge})
    return groups


def resolve_skeleton_nodes(
    skeleton_nodes: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Merge same-path skeleton nodes with near-identical identifier strings.

    Returns (resolved_nodes, stats). Performs a full parent remapping pass so
    no node is orphaned by a merge.
    """
    nodes = [dict(n) for n in skeleton_nodes]
    removed_indexes: set[int] = set()
    id_remap: dict[Any, dict[str, Any]] = {}  # dropped_key -> chosen node's ids
    merged_count = 0

    for i, left in enumerate(nodes):
        if i in removed_indexes:
            continue
        path = str(left.get("path") or "")
        left_key = key_fn(left)
        left_text = _concat_ids(left)
        if not left_text:
            continue
        for j in range(i + 1, len(nodes)):
            if j in removed_indexes:
                continue
            right = nodes[j]
            if str(right.get("path") or "") != path:
                continue
            right_key = key_fn(right)
            if left_key == right_key:
                continue
            right_text = _concat_ids(right)
            if not right_text:
                continue
            # Numbers are precise discriminators: "20vol" vs "30vol" or
            # "Article 5" vs "Article 6" are distinct entities no matter how
            # similar the surrounding text. A false merge destroys data; a
            # kept duplicate is merely redundant.
            if _digit_signature(left_text) != _digit_signature(right_text):
                continue
            if _fuzzy_similarity(left_text, right_text) < _FUZZY_MERGE_THRESHOLD:
                continue
            id_remap[right_key] = dict(left.get("ids") or {})
            _absorb_sources(left, right, right_key)
            removed_indexes.add(j)
            merged_count += 1

    kept_nodes = [n for idx, n in enumerate(nodes) if idx not in removed_indexes]

    if id_remap:
        for node in kept_nodes:
            parent = node.get("parent")
            if parent is None or not isinstance(parent, dict):
                continue
            pkey = _parent_key(parent, key_fn)
            if pkey is not None and pkey in id_remap:
                node["parent"] = {
                    "path": parent.get("path"),
                    "ids": id_remap[pkey],
                }

    stats = {"merged_count": merged_count}
    return kept_nodes, stats
