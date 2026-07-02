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
