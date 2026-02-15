"""Post-merge dedup resolvers for delta extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from .helpers import DedupPolicy, node_identity_key, same_identity_string

logger = logging.getLogger(__name__)


@dataclass
class DeltaResolverConfig:
    """Settings for optional post-merge entity resolvers."""

    enabled: bool = False
    mode: str = "off"  # off | fuzzy | semantic | chain
    fuzzy_threshold: float = 0.8
    semantic_threshold: float = 0.8
    properties: list[str] | None = None
    paths: list[str] | None = None
    allow_merge_different_ids: bool = (
        False  # If False, do not merge when both have non-empty distinct ids
    )


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _node_key(node: dict[str, Any], dedup_policy: dict[str, DedupPolicy]) -> Any:
    return node_identity_key(node, dedup_policy=dedup_policy)


def _relationship_endpoint_key(
    path: str,
    ids: dict[str, Any],
    dedup_policy: dict[str, DedupPolicy],
) -> Any:
    return node_identity_key(
        {"path": str(path or ""), "ids": ids, "properties": {}},
        dedup_policy=dedup_policy,
    )


def _concat_text(node: dict[str, Any], fields: list[str]) -> str:
    raw_props = node.get("properties")
    props: dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
    raw_ids = node.get("ids")
    ids: dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
    values: list[str] = []
    for field in fields:
        value = props.get(field)
        if not isinstance(value, str) or not value.strip():
            value = ids.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " | ".join(values)


def _fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz  # type: ignore

        return float(fuzz.token_sort_ratio(a, b)) / 100.0
    except Exception:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _semantic_similarity(a: str, b: str) -> tuple[float, str | None]:
    if not a or not b:
        return 0.0, None
    try:
        import spacy  # type: ignore

        nlp = spacy.blank("en")
        doc_a = nlp(a)
        doc_b = nlp(b)
        if not doc_a.vector_norm or not doc_b.vector_norm:
            raise ValueError("spaCy model has no vectors")
        return float(doc_a.similarity(doc_b)), None
    except Exception as exc:
        set_a = {token for token in a.lower().split() if token}
        set_b = {token for token in b.lower().split() if token}
        if not set_a or not set_b:
            return 0.0, f"semantic_fallback:{type(exc).__name__}"
        score = len(set_a & set_b) / len(set_a | set_b)
        return score, f"semantic_fallback:{type(exc).__name__}"


def _merge_nodes(primary: dict[str, Any], duplicate: dict[str, Any]) -> None:
    raw_primary = primary.get("properties")
    primary_props: dict[str, Any] = raw_primary if isinstance(raw_primary, dict) else {}
    raw_duplicate = duplicate.get("properties")
    duplicate_props: dict[str, Any] = raw_duplicate if isinstance(raw_duplicate, dict) else {}
    for key, value in duplicate_props.items():
        if key not in primary_props or _is_empty(primary_props.get(key)):
            primary_props[key] = value
    primary["properties"] = primary_props
    if not primary.get("parent") and duplicate.get("parent"):
        primary["parent"] = duplicate["parent"]


def _can_merge_with_ids(
    path: str,
    left: dict[str, Any],
    right: dict[str, Any],
    policy: DedupPolicy | None,
    config: DeltaResolverConfig | None = None,
) -> bool:
    """Decide whether resolver may consider merging this pair based on ids.

    When allow_merge_different_ids is True, always allow (content similarity decides).
    When False, do not merge when both nodes have non-empty, distinct identity strings,
    to avoid collapsing e.g. different experiment_id nodes (Fig-4 vs Fig-5).
    """
    if config is not None and getattr(config, "allow_merge_different_ids", False):
        return True
    left_ids = left.get("ids") if isinstance(left.get("ids"), dict) else {}
    right_ids = right.get("ids") if isinstance(right.get("ids"), dict) else {}
    if not left_ids or not right_ids:
        return True
    fields = (
        policy.identity_fields
        if policy is not None
        else tuple(sorted(set(left_ids) | set(right_ids)))
    )
    for field in fields:
        lval = left_ids.get(field)
        rval = right_ids.get(field)
        if lval is None or rval is None:
            continue
        if isinstance(lval, str) and isinstance(rval, str):
            if not same_identity_string(lval, rval):
                return False
            continue
        if str(lval) != str(rval):
            return False
    return True


def _compute_merge_decision(
    left: dict[str, Any],
    right: dict[str, Any],
    path: str,
    policy: DedupPolicy | None,
    config: DeltaResolverConfig,
    mode: str,
    fields: list[str],
) -> tuple[bool, float, str, str | None]:
    """Return (should_merge, score, resolver_kind, skipped_reason)."""
    text_left = _concat_text(left, fields)
    text_right = _concat_text(right, fields)
    if not text_left or not text_right:
        return False, 0.0, "", None

    score = 0.0
    resolver_kind = ""
    skipped_reason: str | None = None

    if policy and policy.identity_fields:
        left_ids = left.get("ids") if isinstance(left.get("ids"), dict) else {}
        right_ids = right.get("ids") if isinstance(right.get("ids"), dict) else {}
        if left_ids and right_ids:
            common = [
                f
                for f in policy.identity_fields
                if left_ids.get(f) is not None and right_ids.get(f) is not None
            ]
            if common and all(
                same_identity_string(str(left_ids[f]), str(right_ids[f])) for f in common
            ):
                return True, 1.0, "identity", None

    if mode in ("fuzzy", "chain"):
        score = _fuzzy_similarity(text_left, text_right)
        if score >= float(config.fuzzy_threshold):
            return True, score, "fuzzy", None

    if mode in ("semantic", "chain"):
        semantic_score, fallback_reason = _semantic_similarity(text_left, text_right)
        score = semantic_score
        resolver_kind = "semantic"
        if fallback_reason:
            skipped_reason = fallback_reason
        if score >= float(config.semantic_threshold):
            return True, score, "semantic", skipped_reason

    return False, score, resolver_kind, skipped_reason


def _apply_id_remap(
    relationships: list[dict[str, Any]],
    id_remap: dict[Any, dict[str, str]],
    dedup_policy: dict[str, DedupPolicy],
) -> None:
    """Mutate relationships in place, replacing source/target_ids using id_remap."""
    for rel in relationships:
        source_path = str(rel.get("source_path") or "")
        _src_ids = rel.get("source_ids")
        source_ids: dict[str, Any] = _src_ids if isinstance(_src_ids, dict) else {}
        source_key = _relationship_endpoint_key(source_path, source_ids, dedup_policy)
        if source_key in id_remap:
            rel["source_ids"] = dict(id_remap[source_key])
        target_path = str(rel.get("target_path") or "")
        _tgt_ids = rel.get("target_ids")
        target_ids: dict[str, Any] = _tgt_ids if isinstance(_tgt_ids, dict) else {}
        target_key = _relationship_endpoint_key(target_path, target_ids, dedup_policy)
        if target_key in id_remap:
            rel["target_ids"] = dict(id_remap[target_key])


def _is_root_parent(parent: Any) -> bool:
    if parent is None:
        return True
    if not isinstance(parent, dict):
        return False
    return str(parent.get("path") or "") == ""


def _parents_equivalent(left_parent: Any, right_parent: Any) -> bool:
    if left_parent == right_parent:
        return True
    return _is_root_parent(left_parent) and _is_root_parent(right_parent)


def resolve_post_merge_graph(
    merged_graph: dict[str, Any],
    *,
    dedup_policy: dict[str, DedupPolicy],
    config: DeltaResolverConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run optional post-merge resolvers and return updated graph + stats."""

    mode = (config.mode or "off").lower()
    if not config.enabled or mode == "off":
        return merged_graph, {"enabled": False, "mode": mode, "actions": [], "skipped_reasons": []}

    nodes = [dict(node) for node in (merged_graph.get("nodes") or []) if isinstance(node, dict)]
    relationships = [
        dict(rel) for rel in (merged_graph.get("relationships") or []) if isinstance(rel, dict)
    ]

    allowed_paths = set(config.paths or [])
    actions: list[dict[str, Any]] = []
    skipped_reasons: list[str] = []
    removed_indexes: set[int] = set()
    id_remap: dict[Any, dict[str, str]] = {}
    merge_tiers: dict[str, int] = {
        "resolver_fuzzy": 0,
        "resolver_semantic": 0,
        "resolver_identity": 0,
    }

    for i, left in enumerate(nodes):
        if i in removed_indexes:
            continue
        path = str(left.get("path") or "")
        if allowed_paths and path not in allowed_paths:
            continue
        for j in range(i + 1, len(nodes)):
            if j in removed_indexes:
                continue
            right = nodes[j]
            right_path = str(right.get("path") or "")
            if right_path != path:
                continue

            policy = dedup_policy.get(path)
            if not _can_merge_with_ids(path, left, right, policy, config):
                continue
            if not _parents_equivalent(left.get("parent"), right.get("parent")):
                continue

            fields = list(config.properties or [])
            if not fields and policy is not None:
                fields = list(policy.identity_fields or ())
            if not fields and policy is not None:
                fields = list(policy.allowed_match_fields)
            if not fields:
                fields = ["name", "title", "id", "code"]

            should_merge, score, resolver_kind, skipped_reason = _compute_merge_decision(
                left, right, path, policy, config, mode, fields
            )
            if skipped_reason:
                skipped_reasons.append(skipped_reason)
            if not should_merge:
                continue

            text_left = _concat_text(left, fields)
            text_right = _concat_text(right, fields)
            _merge_nodes(left, right)
            removed_indexes.add(j)
            left_ids = left.get("ids") if isinstance(left.get("ids"), dict) else {}
            right_ids = right.get("ids") if isinstance(right.get("ids"), dict) else {}
            if left_ids and right_ids:
                id_remap[_relationship_endpoint_key(path, right_ids, dedup_policy)] = {
                    str(k): str(v) for k, v in left_ids.items()
                }
            if resolver_kind == "fuzzy":
                merge_tiers["resolver_fuzzy"] += 1
            elif resolver_kind == "semantic":
                merge_tiers["resolver_semantic"] += 1
            elif resolver_kind == "identity":
                merge_tiers["resolver_identity"] += 1
            actions.append(
                {
                    "type": f"{resolver_kind}_merge",
                    "path": path,
                    "score": round(score, 4),
                    "left_text": text_left,
                    "right_text": text_right,
                }
            )

    if id_remap:
        _apply_id_remap(relationships, id_remap, dedup_policy)

    kept_nodes = [node for idx, node in enumerate(nodes) if idx not in removed_indexes]
    resolver_stats = {
        "enabled": True,
        "mode": mode,
        "merged_count": len(actions),
        "merge_tiers": merge_tiers,
        "actions": actions,
        "skipped_reasons": sorted(set(skipped_reasons)),
    }
    return {"nodes": kept_nodes, "relationships": relationships}, resolver_stats
