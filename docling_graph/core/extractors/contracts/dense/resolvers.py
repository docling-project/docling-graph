"""
Fuzzy post-merge resolver for dense skeleton nodes.

Used only by the "aggressive" dedupe mode: merges skeleton nodes on the same
path whose identifier strings are near-identical (OCR noise, casing, ligature
splits). The similarity threshold is internal — it is a property of how OCR
noise looks, not something users should tune per run.

Fully autonomous: no imports from other contracts.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Similarity above which two same-path identifier strings are considered the
# same real-world entity (OCR noise: dropped accents, punctuation, spacing).
_FUZZY_MERGE_THRESHOLD = 0.9

# A canonical id must be at least this long to be treated as a containment base;
# below it, substrings are too common to be a safe merge signal ("PVDF" is 4).
_MIN_CONTAINMENT_LEN = 4

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


def _canonical_text_from_key(key: Any) -> str:
    """Concatenated canonical id text from a (path, pairs) identity key.

    Uses the canonical values the key already carries (same normalization dense
    dedup uses), so containment comparison is case/diacritic-insensitive. Returns
    "" for positional-fallback keys (a synthetic "__key" field), which must never
    be treated as containment candidates.
    """
    try:
        _path, pairs = key
    except Exception:
        return ""
    if any(field == "__key" for field, _ in pairs):
        return ""
    return "".join(str(value) for _, value in pairs if value)


def merge_contained_skeleton_nodes(
    skeleton_nodes: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically merge same-path nodes whose canonical id is a superset.

    A more specific descriptive id ("nanoscaled LiFePO4 (LFP)") names the same
    real-world entity as its base ("LiFePO4 (LFP)"); likewise "Polyvinylidene
    difluoride (PVDF)" and "PVDF". Case/diacritic dedup and the fuzzy resolver
    both miss this — it is a substring, not near-equality. Safe enough to run in
    ``standard`` because it is bounded by three guards:

    - same catalog path only;
    - equal digit signatures (so "LFP_20vol"/"LFP_30vol" and "Article 5"/"6"
      never merge — a false merge destroys data, a kept duplicate is only noise);
    - a unique base: a superset that contains *two* different shorter ids is
      ambiguous and is left alone (mirrors the binder's fuzzy-containment guard).

    The shorter (base) id is kept — it is the canonical term and the most
    verbatim-friendly locator. Returns (resolved_nodes, stats) with a full parent
    remap so no child is orphaned by a merge.
    """
    nodes = [dict(n) for n in skeleton_nodes]
    keys = [key_fn(n) for n in nodes]
    texts = [_canonical_text_from_key(k) for k in keys]
    sigs = [_digit_signature(t) for t in texts]
    paths = [str(n.get("path") or "") for n in nodes]

    removed_indexes: set[int] = set()
    id_remap: dict[Any, dict[str, Any]] = {}
    merged_count = 0

    # For each candidate superset j, find the strictly-shorter same-path bases it
    # contains; merge only when exactly one base qualifies. removed_indexes only
    # ever gains the CURRENT j within its own iteration (never a smaller index
    # revisited later), so a distinct not-yet-removed check on entry is redundant.
    for j in range(len(nodes)):
        superset_text = texts[j]
        if not superset_text:
            continue
        bases = [
            i
            for i in range(len(nodes))
            if i != j
            and i not in removed_indexes
            and paths[i] == paths[j]
            and texts[i]
            and len(texts[i]) >= _MIN_CONTAINMENT_LEN
            and len(texts[i]) < len(superset_text)
            and texts[i] in superset_text
            and sigs[i] == sigs[j]
        ]
        if len(bases) != 1:
            continue
        keeper_idx = bases[0]
        id_remap[keys[j]] = dict(nodes[keeper_idx].get("ids") or {})
        _absorb_sources(nodes[keeper_idx], nodes[j], keys[j])
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
                node["parent"] = {"path": parent.get("path"), "ids": id_remap[pkey]}

    return kept_nodes, {"merged_count": merged_count}


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
