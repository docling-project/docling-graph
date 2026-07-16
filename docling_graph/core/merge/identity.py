"""Node identity across exported graphs: template walk + re-keying (design §5.4).

``recompute_node_id`` is a byte-for-byte replica of
``NodeIDRegistry._generate_fingerprint`` evaluated over node *attributes*
instead of a model instance, pinned by a parity test so the two
implementations can never drift silently. Re-keying guards against
normalizer drift across docling-graph versions and hand-edited exports.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator, Mapping, get_args

import networkx as nx
from pydantic import BaseModel

from ...exceptions import ConfigurationError
from ...logging_utils import get_component_logger
from ..provenance.identity import PROVENANCE_NODE_ATTR
from ..utils.alias_reconciler import _META_ATTRS
from ..utils.entity_name_normalizer import canonicalize_identity_for_dedup
from .node_folder import fold_edge, fold_node_attrs
from .policy import MergePolicy
from .provenance_merge import merge_node_views

logger = get_component_logger("MergeIdentity", __name__)

# ISO 8601 datetime with a 'T' separator, as datetime.isoformat() (and hence
# JSONExporter) writes it. Date-only values are excluded on purpose: they
# already agree between str(date) and date.isoformat().
_ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")


def _normalize_temporal(value: Any) -> Any:
    """Rewrite an exported ISO 8601 datetime string to its ``str(datetime)`` form.

    The registry canonicalizes ``str(datetime)`` ('2024-01-01 12:00:00') while
    exported attributes carry ``datetime.isoformat()`` ('2024-01-01T12:00:00');
    the 'T' survives canonicalization, so without this rewrite the two forms
    fingerprint to different node ids.
    """
    if isinstance(value, str) and _ISO_DATETIME.fullmatch(value):
        return value.replace("T", " ", 1)
    return value


def _iter_model_types(annotation: Any) -> Iterator[type[BaseModel]]:
    """Yield every BaseModel subclass reachable in a field annotation,
    unwrapping Optional/List/Union/Dict generics recursively."""
    if annotation is None:
        return
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            yield annotation
        return
    for arg in get_args(annotation):
        yield from _iter_model_types(arg)


def id_fields_by_template(template: type[BaseModel]) -> dict[str, list[str]]:
    """Collect {class name: graph_id_fields} for every model class reachable
    below ``template`` — the class-level analog of
    ``alias_reconciler.id_fields_by_class`` (which needs instances). Classes
    without id fields map to an empty list. Cycle-safe for recursive models."""
    result: dict[str, list[str]] = {}
    seen: set[type[BaseModel]] = set()

    def _visit(cls: type[BaseModel]) -> None:
        if cls in seen:
            return
        seen.add(cls)
        config = getattr(cls, "model_config", {}) or {}
        raw = config.get("graph_id_fields", []) if isinstance(config, dict) else []
        result.setdefault(cls.__name__, [f for f in raw if isinstance(f, str)])
        for field in cls.model_fields.values():
            for sub in _iter_model_types(field.annotation):
                _visit(sub)

    _visit(template)
    return result


def recompute_node_id(attrs: Mapping[str, Any], id_fields: list[str] | None) -> str:
    """Deterministic node id for exported node attributes.

    Replicates ``NodeIDRegistry._generate_fingerprint`` exactly: declared id
    fields are canonicalized with ``canonicalize_identity_for_dedup`` (list
    values become sorted, deduped canonical tuples); classes without id fields
    use the component branch (all truthy non-collection attributes, meta
    attributes excluded); ``__class__`` is mixed in; blake2b[:16] hex, prefixed
    with the class name. Exported ISO datetime strings are rewritten to their
    ``str(datetime)`` form first so they fingerprint like the registry's raw
    values.

    Known limitation (component branch only): the graph converter injects
    reserved ``id``/``label``/``type`` node attributes, so a model field
    literally named one of those is indistinguishable from the injected
    attribute here — the registry fingerprints the model's value while this
    function must skip the key as meta. No compensation is attempted: such
    templates already violate the reserved-attribute rule and produce broken
    graphs upstream.
    """
    class_name = str(attrs.get("__class__") or "")
    if not class_name:
        raise ConfigurationError(
            "Node has no __class__ attribute; cannot recompute its id",
            details={"node": str(attrs.get("id") or "")},
        )
    fingerprint_data: dict[str, Any] = {}
    if id_fields:
        for field in id_fields:
            if field in attrs:
                value = attrs[field]
                if isinstance(value, list):
                    canon = {
                        canonicalize_identity_for_dedup(field, _normalize_temporal(v))
                        for v in value
                    }
                    fingerprint_data[field] = tuple(sorted(canon))
                else:
                    fingerprint_data[field] = canonicalize_identity_for_dedup(
                        field, _normalize_temporal(value)
                    )
    else:
        # Component branch: entity-valued fields are None on the node and
        # embedded components are dicts, so both fall out exactly as they do
        # when the registry iterates the model instance.
        for field, value in attrs.items():
            if field in _META_ATTRS:
                continue
            if value and not isinstance(value, list | dict):
                fingerprint_data[field] = _normalize_temporal(value)
    # Deliberate divergence from the registry: skolem_document_id is the
    # content-bearing stamp _skolemize_root_collisions writes to keep
    # filename-stem-colliding roots apart. The registry never sees it (it does
    # not exist in model space), but without mixing it in a re-merge would
    # recompute the skolemized root back to its colliding base id and silently
    # re-fuse two distinct documents.
    skolem_document_id = attrs.get("skolem_document_id")
    if skolem_document_id:
        fingerprint_data["skolem_document_id"] = str(skolem_document_id)
    fingerprint_data["__class__"] = class_name
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
    fingerprint = hashlib.blake2b(fingerprint_str.encode()).hexdigest()[:16]
    return f"{class_name}_{fingerprint}"


def rekey_graph(
    graph: nx.DiGraph,
    id_fields_map: Mapping[str, list[str]],
    policy: MergePolicy,
    source_tag: str,
) -> tuple[nx.DiGraph, int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute every node id under the current normalizer.

    Fan-in (two old ids recomputing to one new id) folds through
    ``fold_node_attrs``/``fold_edge`` instead of ``nx.relabel_nodes``
    clobbering. Returns ``(rekeyed_graph, changed_id_count, field_conflicts,
    edge_label_conflicts)``.

    Raises:
        ConfigurationError: When a recomputed id collides across different
            ``__class__`` values (registry parity: corrupted input).
    """
    mapping: dict[str, str] = {}
    class_by_new_id: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        cls = str(attrs.get("__class__") or "")
        new_id = recompute_node_id(attrs, list(id_fields_map.get(cls) or []))
        existing_cls = class_by_new_id.get(new_id)
        if existing_cls is not None and existing_cls != cls:
            raise ConfigurationError(
                f"Node ID collision after re-keying: {new_id} maps to classes "
                f"{existing_cls} and {cls}",
                details={"node_id": str(node_id), "source": source_tag},
            )
        class_by_new_id[new_id] = cls
        mapping[str(node_id)] = new_id
    changed = sum(1 for old, new in mapping.items() if old != new)

    rekeyed = nx.DiGraph()
    rekeyed.graph.update(graph.graph)
    field_conflicts: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        new_id = mapping[str(node_id)]
        incoming = dict(attrs)
        incoming["id"] = new_id
        if new_id in rekeyed:
            survivor = rekeyed.nodes[new_id]
            field_conflicts.extend(fold_node_attrs(survivor, incoming, policy, source_tag))
            # merged_aliases is a meta attr (the fold skips it): union the
            # audit records so re-merged inputs keep their alias history.
            incoming_aliases = incoming.get("merged_aliases") or []
            if incoming_aliases:
                aliases = survivor.setdefault("merged_aliases", [])
                for entry in incoming_aliases:
                    if entry not in aliases:
                        aliases.append(dict(entry) if isinstance(entry, dict) else entry)
            merged_view = merge_node_views(
                survivor.get(PROVENANCE_NODE_ATTR), incoming.get(PROVENANCE_NODE_ATTR)
            )
            if merged_view is not None:
                survivor[PROVENANCE_NODE_ATTR] = merged_view
        else:
            rekeyed.add_node(new_id, **incoming)

    edge_conflicts: list[dict[str, Any]] = []
    for source, target, attrs in graph.edges(data=True):
        new_source, new_target = mapping[str(source)], mapping[str(target)]
        if rekeyed.has_edge(new_source, new_target):
            record = fold_edge(
                new_source,
                new_target,
                rekeyed.edges[new_source, new_target],
                attrs,
                policy,
                source_tag,
            )
            if record is not None:
                edge_conflicts.append(record)
        else:
            rekeyed.add_edge(new_source, new_target, **dict(attrs))

    if changed:
        logger.info(
            "Re-keyed %s of %s node id(s) in %s under the current normalizer",
            changed,
            graph.number_of_nodes(),
            source_tag,
        )
    return rekeyed, changed, field_conflicts, edge_conflicts
