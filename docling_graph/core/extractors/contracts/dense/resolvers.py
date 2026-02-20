"""
Optional post-merge resolvers for dense skeleton nodes (fuzzy/semantic dedup).

Fully autonomous: no imports from contracts.delta or contracts.staged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class DenseResolverConfig:
    """Settings for optional post-merge skeleton resolvers."""

    enabled: bool = False
    mode: str = "off"  # off | fuzzy | semantic | chain
    fuzzy_threshold: float = 0.8
    semantic_threshold: float = 0.8
    allow_merge_different_ids: bool = False


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


def _semantic_similarity(a: str, b: str) -> tuple[float, str | None]:
    if not a or not b:
        return 0.0, None
    try:
        import spacy
        nlp = spacy.blank("en")
        doc_a = nlp(a)
        doc_b = nlp(b)
        if not doc_a.vector_norm or not doc_b.vector_norm:
            raise ValueError("spaCy model has no vectors")
        return float(doc_a.similarity(doc_b)), None
    except Exception as exc:
        set_a = {t for t in a.lower().split() if t}
        set_b = {t for t in b.lower().split() if t}
        if not set_a or not set_b:
            return 0.0, f"semantic_fallback:{type(exc).__name__}"
        score = len(set_a & set_b) / len(set_a | set_b)
        return score, f"semantic_fallback:{type(exc).__name__}"


def _can_merge_with_ids(
    left: dict[str, Any],
    right: dict[str, Any],
    allow_merge_different_ids: bool,
) -> bool:
    """If both have non-empty distinct ids and allow_merge_different_ids is False, do not merge."""
    if allow_merge_different_ids:
        return True
    left_ids = left.get("ids") or {}
    right_ids = right.get("ids") or {}
    if not isinstance(left_ids, dict) or not isinstance(right_ids, dict):
        return True
    left_vals = tuple(sorted(str(v) for v in left_ids.values() if v is not None))
    right_vals = tuple(sorted(str(v) for v in right_ids.values() if v is not None))
    if not left_vals or not right_vals:
        return True
    return left_vals == right_vals


def _compute_merge_decision(
    left: dict[str, Any],
    right: dict[str, Any],
    config: DenseResolverConfig,
    mode: str,
) -> tuple[bool, float, str]:
    """Return (should_merge, score, resolver_kind)."""
    text_left = _concat_ids(left)
    text_right = _concat_ids(right)
    if not text_left or not text_right:
        return False, 0.0, ""

    if mode in ("fuzzy", "chain"):
        score = _fuzzy_similarity(text_left, text_right)
        if score >= float(config.fuzzy_threshold):
            return True, score, "fuzzy"

    if mode in ("semantic", "chain"):
        semantic_score, fallback = _semantic_similarity(text_left, text_right)
        if semantic_score >= float(config.semantic_threshold):
            return True, semantic_score, "semantic"
        if fallback and mode in ("semantic", "chain"):
            fuzzy_score = _fuzzy_similarity(text_left, text_right)
            if fuzzy_score >= float(config.fuzzy_threshold):
                return True, fuzzy_score, "fuzzy"

    return False, 0.0, ""


def _parent_key(parent: dict[str, Any] | None, key_fn: Callable[[dict], Any]) -> Any | None:
    """Return the identity key for a parent dict, or None if no parent."""
    if parent is None or not isinstance(parent, dict):
        return None
    return key_fn({"path": parent.get("path"), "ids": parent.get("ids") or {}})


def resolve_skeleton_nodes(
    skeleton_nodes: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    config: DenseResolverConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Optionally merge skeleton nodes that refer to the same entity (fuzzy/semantic).
    Returns (resolved_nodes, stats). Performs a full parent remapping pass so no node is orphaned.
    """
    mode = (config.mode or "off").lower()
    if not config.enabled or mode == "off":
        return skeleton_nodes, {"enabled": False, "mode": mode, "merged_count": 0}

    nodes = [dict(n) for n in skeleton_nodes]
    removed_indexes: set[int] = set()
    id_remap: dict[Any, dict[str, Any]] = {}  # dropped_key -> chosen node's ids
    merged_count = 0

    for i, left in enumerate(nodes):
        if i in removed_indexes:
            continue
        path = str(left.get("path") or "")
        left_key = key_fn(left)
        for j in range(i + 1, len(nodes)):
            if j in removed_indexes:
                continue
            right = nodes[j]
            if str(right.get("path") or "") != path:
                continue
            if not _can_merge_with_ids(left, right, config.allow_merge_different_ids):
                continue
            right_key = key_fn(right)
            if left_key == right_key:
                continue
            should_merge, _score, _kind = _compute_merge_decision(left, right, config, mode)
            if not should_merge:
                continue
            right_ids = right.get("ids") or {}
            if isinstance(right_ids, dict):
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

    stats = {
        "enabled": True,
        "mode": mode,
        "merged_count": merged_count,
    }
    return kept_nodes, stats
