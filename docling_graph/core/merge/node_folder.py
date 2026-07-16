"""Per-field fold rules for same-identity nodes and edges (design §5.3/§5.7).

Generalizes the fill-empty policy of ``alias_reconciler._merge_nodes`` into an
explicit per-field strategy table. Meta attributes (``_META_ATTRS``) are never
treated as content: provenance and ``merged_from`` audit records are handled
by the caller (GraphMerger), not by the fold.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from ..utils.alias_reconciler import _META_ATTRS
from ..utils.description_merger import merge_descriptions
from .policy import MergePolicy


def _is_empty(value: Any) -> bool:
    """Empty per the fold contract: None or empty str/list/dict.

    Numeric zero and False are meaningful and never treated as empty (same
    semantics as ``graph_converter._is_empty_value``).
    """
    return value is None or (isinstance(value, str | list | dict) and len(value) == 0)


def _content_hash(item: Any) -> str:
    """Content hash for list-of-dict dedup (same idiom as ``_merge_entity_lists``)."""
    payload = json.dumps(item, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode()).hexdigest()[:16]


def _is_dict_list(value: list[Any]) -> bool:
    return bool(value) and all(isinstance(item, dict) for item in value)


def fold_node_attrs(
    survivor: dict[str, Any],
    incoming: Mapping[str, Any],
    policy: MergePolicy,
    dropped_source: str,
) -> list[dict[str, Any]]:
    """Fold ``incoming`` node attributes into ``survivor`` in place.

    Rules, per field (first match wins):
        1. Meta attribute -> skip (provenance/audit handled by the caller).
        2. Incoming empty -> skip (empty never clobbers).
        3. Survivor empty -> fill from incoming.
        4. Equal values -> no-op (checked before combine so folds stay idempotent).
        5. Both str and field in ``policy.combine_fields`` -> sentence-dedup merge.
        6. Both scalar lists -> order-preserving union.
        7. Both lists of dicts -> content-hash dedup, unmatched appended.
        8. Both non-empty, unequal -> conflict: survivor kept, record returned;
           ``keep-all`` additionally stores the suppressed value under the
           ``__conflicts__`` node attribute.

    Returns the conflict records produced by rule 8, each shaped
    ``{"node", "field", "kept", "dropped", "dropped_source"}``.
    """
    conflicts: list[dict[str, Any]] = []
    for key, value in incoming.items():
        if key in _META_ATTRS:
            continue
        if _is_empty(value):
            continue
        current = survivor.get(key)
        if _is_empty(current):
            survivor[key] = copy.deepcopy(value)
            continue
        if current == value:
            continue
        if key in policy.combine_fields and isinstance(current, str) and isinstance(value, str):
            survivor[key] = merge_descriptions(
                current, value, max_length=policy.description_max_length
            )
            continue
        if isinstance(current, list) and isinstance(value, list):
            if _is_dict_list(current) and _is_dict_list(value):
                seen = {_content_hash(item) for item in current}
                for item in value:
                    item_hash = _content_hash(item)
                    if item_hash not in seen:
                        seen.add(item_hash)
                        current.append(copy.deepcopy(item))
            else:
                for item in value:
                    if item not in current:
                        current.append(copy.deepcopy(item))
            continue
        conflicts.append(
            {
                "node": str(survivor.get("id") or ""),
                "field": key,
                "kept": current,
                "dropped": value,
                "dropped_source": dropped_source,
            }
        )
        if policy.conflicts == "keep-all":
            suppressed = survivor.setdefault("__conflicts__", [])
            record = {"field": key, "value": copy.deepcopy(value), "source": dropped_source}
            if record not in suppressed:
                suppressed.append(record)
    return conflicts


def fold_edge_attrs(
    survivor: dict[str, Any],
    incoming: Mapping[str, Any],
    policy: MergePolicy,
) -> None:
    """Same-label edge attr fold: fill-empty + scalar-list union only.

    Conflicting non-empty scalars keep the survivor value silently — edge
    attrs are decoration, not identity; label conflicts are handled (and
    recorded) by :func:`fold_edge`.
    """
    for key, value in incoming.items():
        if key in ("label", "also_labels") or _is_empty(value):
            continue
        current = survivor.get(key)
        if _is_empty(current):
            survivor[key] = copy.deepcopy(value)
        elif isinstance(current, list) and isinstance(value, list):
            for item in value:
                if item not in current:
                    current.append(copy.deepcopy(item))


def _extend_also_labels(
    survivor: dict[str, Any],
    labels: list[str],
    kept_label: str,
) -> None:
    """Retain losing edge labels under ``also_labels`` (created lazily)."""
    extra = [label for label in labels if label and label != kept_label and isinstance(label, str)]
    if not extra:
        return
    also = survivor.setdefault("also_labels", [])
    for label in extra:
        if label not in also:
            also.append(label)


def fold_edge(
    source: str,
    target: str,
    survivor: dict[str, Any],
    incoming: Mapping[str, Any],
    policy: MergePolicy,
    dropped_source: str,
) -> dict[str, Any] | None:
    """Union an incoming edge's attrs into the surviving ``(source, target)`` edge.

    Same label -> attr fold (returns None). Different label -> the survivor's
    edge is kept as-is, the losing label is retained under ``also_labels``,
    and a conflict record ``{"source", "kept_label", "dropped_labels",
    "target", "dropped_source"}`` is returned. Labels are never combined or
    invented. ``also_labels`` carried by the incoming edge (from a previous
    merge) is unioned in either way.
    """
    kept_label = str(survivor.get("label") or "")
    incoming_label = str(incoming.get("label") or "")
    carried = [str(label) for label in (incoming.get("also_labels") or [])]
    if incoming_label == kept_label:
        fold_edge_attrs(survivor, incoming, policy)
        _extend_also_labels(survivor, carried, kept_label)
        return None
    _extend_also_labels(survivor, [incoming_label, *carried], kept_label)
    return {
        "source": source,
        "kept_label": kept_label,
        "dropped_labels": [incoming_label],
        "target": target,
        "dropped_source": dropped_source,
    }
