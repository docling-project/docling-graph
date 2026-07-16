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
import unicodedata
from typing import Any, Mapping

from ..provenance.identity import PROVENANCE_NODE_ATTR
from ..utils.alias_reconciler import _META_ATTRS
from ..utils.description_merger import merge_descriptions
from .policy import MergePolicy

VARIANT_TYPE = "variant"
VARIANT_EDGE_LABEL = "HAS_CONFLICT_VARIANT"


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


def _equivalence_key(value: str) -> str:
    """Comparison shadow of a string: Unicode-NFKC-normalized, casefolded,
    whitespace-free. OCR renders the same source text with unstable spacing
    and width forms ('059/987 65 40' vs '059/9876540'), which must compare
    equal rather than conflict. Stored values are never rewritten."""
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def scalars_equivalent(current: Any, incoming: Any) -> bool:
    """Rule-4 equality: exact match, or strings equal modulo formatting noise
    (Unicode form, case, whitespace — see :func:`_equivalence_key`)."""
    if current == incoming:
        return True
    if isinstance(current, str) and isinstance(incoming, str):
        return _equivalence_key(current) == _equivalence_key(incoming)
    return False


def variant_node_id(base_id: str, stamp: str) -> str:
    """Deterministic id for a conflict-variant node.

    Derived from the canonical node's id plus the source stamp so that
    (a) re-keying can move variants in lockstep with their re-keyed base
    (``identity.rekey_graph``) and (b) a re-merge of a constituent input
    mints the identical id and skips instead of duplicating the variant.
    """
    suffix = f"in{stamp.removeprefix('input-')}" if stamp.startswith("input-") else stamp[:8]
    return f"{base_id}__var_{suffix}"


def build_conflict_variant(
    base_attrs: Mapping[str, Any],
    dropped_fields: Mapping[str, Any],
    stamp: str,
    provenance: Any = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Reify the suppressed side of rule-8 conflicts as a variant sub-node.

    Returns ``(variant_id, node_attrs, edge_attrs)``: a ``<Class>Variant``
    node (``type: variant``, ``__class__`` kept so consumers can interpret
    the fields) holding only the suppressed values of the conflicting
    fields, linked from the canonical node by a ``HAS_CONFLICT_VARIANT``
    edge. ``variant_of`` and ``variant_document_id`` are content-bearing:
    they keep the cleaner's content-hash dedup from ever re-fusing two
    variants that happen to carry equal values.
    """
    base_id = str(base_attrs.get("id") or "")
    cls = str(base_attrs.get("__class__") or "")
    variant_id = variant_node_id(base_id, stamp)
    attrs: dict[str, Any] = {
        "id": variant_id,
        "label": f"{cls}Variant" if cls else "ConflictVariant",
        "type": VARIANT_TYPE,
        "__class__": cls,
        **{field: copy.deepcopy(value) for field, value in dropped_fields.items()},
        "variant_of": base_id,
        "variant_document_id": stamp,
    }
    if provenance is not None:
        attrs[PROVENANCE_NODE_ATTR] = copy.deepcopy(provenance)
    edge_attrs = {
        "label": VARIANT_EDGE_LABEL,
        "document_id": stamp,
        "fields": sorted(dropped_fields),
    }
    return variant_id, attrs, edge_attrs


def conflicting_scalar_fields(
    survivor: Mapping[str, Any],
    incoming: Mapping[str, Any],
    policy: MergePolicy,
) -> list[str]:
    """Fields where folding ``incoming`` into ``survivor`` would trip rule 8.

    The non-mutating dry-run twin of :func:`fold_node_attrs`: a field
    conflicts when both values are non-empty and not equivalent (exact or
    formatting-insensitive string equality, :func:`scalars_equivalent`), the
    field is not sentence-combinable (``policy.combine_fields``), and the
    values are not lists (which union instead of conflicting). Meta
    attributes never count.
    """
    conflicts: list[str] = []
    for key, value in incoming.items():
        if key in _META_ATTRS or _is_empty(value):
            continue
        current = survivor.get(key)
        if _is_empty(current) or scalars_equivalent(current, value):
            continue
        if key in policy.combine_fields and isinstance(current, str) and isinstance(value, str):
            continue
        if isinstance(current, list) and isinstance(value, list):
            continue
        conflicts.append(key)
    return sorted(conflicts)


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
        4. Equivalent values (exact, or strings equal modulo Unicode form,
           case, and whitespace) -> no-op, survivor kept verbatim (checked
           before combine so folds stay idempotent).
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
        if scalars_equivalent(current, value):
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
