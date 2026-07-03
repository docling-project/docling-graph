"""Bind provenance ledger entries to graph nodes after conversion (spec hook H9).

The binder walks the validated model tree along the same catalog paths the
dense contract used, computes each entity's canonical identity with the same
shared function the orchestrator recorded with, and resolves node IDs through
the same NodeIDRegistry instance the converter used — so binding can never
disagree with either side.

Resolution ladder (fail-empty, never fail-wrong):
    exact identity key -> unique same-path fuzzy containment -> unresolved.

Not imported from the package __init__ (and the dense catalog is imported
lazily) to keep ``docling_graph.core.provenance`` import-light: the dense
orchestrator imports that package at module load.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import networkx as nx
from pydantic import BaseModel

from .anchor_scan import locate_values
from .identity import (
    PROVENANCE_NODE_ATTR,
    canonical_id_text,
    compact_view,
    identity_key,
    merge_compact_views,
)
from .models import NodeProvenance, ProvenanceLedger, SourceAnchor

logger = logging.getLogger(__name__)


def _relative_path(parent_path: str, child_path: str) -> str:
    """Catalog path of child relative to parent (e.g. 'a[].b[]' under 'a[]' -> 'b[]')."""
    if not parent_path:
        return child_path
    if child_path.startswith(parent_path):
        return child_path[len(parent_path) :].lstrip(".")
    return child_path


def _instances_at(instance: BaseModel, relative_path: str) -> Iterator[BaseModel]:
    """Yield model instances reached from ``instance`` along a relative catalog path.

    Handles component hops: a child entity path may traverse embedded
    non-entity models (e.g. 'metadata.authors[]' from the root entity).
    """
    current: list[BaseModel] = [instance]
    for segment in relative_path.split("."):
        if not segment:
            continue
        is_list = segment.endswith("[]")
        field_name = segment[:-2] if is_list else segment
        next_level: list[BaseModel] = []
        for obj in current:
            value = getattr(obj, field_name, None)
            if value is None:
                continue
            if is_list:
                if isinstance(value, list):
                    next_level.extend(v for v in value if isinstance(v, BaseModel))
            elif isinstance(value, BaseModel):
                next_level.append(value)
        current = next_level
        if not current:
            return
    yield from current


def _fuzzy_same_path_lookup(
    path: str,
    ids: dict[str, Any],
    fuzzy_index: dict[str, list[tuple[str, NodeProvenance]]],
) -> NodeProvenance | None:
    """Unique canonical-containment match within the same catalog path, else None."""
    ref_text = canonical_id_text(ids)
    if not ref_text or len(ref_text) < 3:
        return None
    matches: list[NodeProvenance] = []
    for candidate_text, entry in fuzzy_index.get(path, []):
        if candidate_text and (ref_text in candidate_text or candidate_text in ref_text):
            matches.append(entry)
            if len(matches) > 1:
                return None
    return matches[0] if len(matches) == 1 else None


def bind_provenance(
    *,
    graph: nx.DiGraph,
    models: list[BaseModel],
    ledger: ProvenanceLedger,
    registry: Any,
    template: type[BaseModel],
    include_spans: bool = False,
) -> dict[str, int]:
    """Annotate graph nodes with compact provenance views; returns bind stats.

    Every entity node reached from the extracted models gets exactly one of:
    a resolved compact view, or ``{"status": "unresolved"}`` — never a wrong
    attribution and never silence.
    """
    from ..extractors.contracts.dense.catalog import build_node_catalog

    catalog = build_node_catalog(template)
    spec_by_path = {s.path: s for s in catalog.nodes}
    children_by_parent: dict[str, list[Any]] = {}
    for spec in catalog.nodes:
        if spec.path:
            children_by_parent.setdefault(spec.parent_path, []).append(spec)

    fuzzy_index: dict[str, list[tuple[str, NodeProvenance]]] = {}
    for entry in ledger.nodes.values():
        text = canonical_id_text(entry.ids)
        if text:
            fuzzy_index.setdefault(entry.catalog_path, []).append((text, entry))

    # Verbatim location uses each node's FINAL identifier values against the
    # stored chunk text — this is the precise locator and is what makes the
    # skeleton-vs-fill id mismatch (issue: all-unresolved dense) a non-problem.
    chunk_texts = {cid: rec.text for cid, rec in ledger.chunks.items() if rec.text}

    stats = {
        "nodes_seen": 0,
        "bound_verbatim": 0,
        "bound_observed": 0,
        "bound_document": 0,
        "unresolved": 0,
    }
    added_verbatim = 0
    visited: set[int] = set()
    # A ledger without per-node skeleton entries (direct contract) falls back to
    # document scope when a node cannot be located; a node-level (dense) ledger
    # falls back to unresolved.
    document_fallback = not ledger.node_level
    document_id = ledger.document.document_id if ledger.document is not None else ""

    def _bind_instance(instance: BaseModel, path: str) -> None:
        nonlocal added_verbatim
        spec = spec_by_path.get(path)
        if spec is None or spec.kind != "entity":
            return
        node_id = registry.get_node_id(instance)
        if node_id not in graph.nodes:
            return
        stats["nodes_seen"] += 1
        node_data = graph.nodes[node_id]
        ids = {
            f: getattr(instance, f, None)
            for f in spec.id_fields
            if getattr(instance, f, None) is not None
        }
        key = identity_key(path, ids, spec.id_fields)
        entry = ledger.nodes.get(key) if key else None
        if entry is None:
            entry = _fuzzy_same_path_lookup(path, ids, fuzzy_index)

        # A document-scoped entry (the root) is intentionally whole-document —
        # never try to pinpoint it (its id often appears in many chunks).
        if entry is not None and "scope:document" in entry.notes:
            _assign(node_data, compact_view(entry, ledger, include_spans=include_spans))
            stats["bound_observed"] += 1
            return

        # Precise location: does the node's real identifier appear verbatim?
        located = locate_values([str(v) for v in ids.values()], chunk_texts)
        located = [(c, s) for c, s in located if c in ledger.chunks]
        if located:
            if entry is None:
                # Direct contract (or unmatched dense node): synthesize an entry
                # so the ledger records the grounding too.
                entry = NodeProvenance(
                    identity_key=key or f"{path}#located{len(ledger.nodes)}",
                    catalog_path=path,
                    node_type=spec.node_type,
                    ids={str(k): str(v) for k, v in ids.items()},
                )
                ledger.nodes[entry.identity_key] = entry
            existing = {(a.chunk_id, a.span) for a in entry.anchors if a.kind == "verbatim"}
            for chunk_id, span in located:
                if (chunk_id, span) not in existing:
                    entry.anchors.append(
                        SourceAnchor(chunk_id=chunk_id, kind="verbatim", span=span)
                    )
                    existing.add((chunk_id, span))
                    added_verbatim += 1
            _assign(node_data, compact_view(entry, ledger, include_spans=include_spans))
            stats["bound_verbatim"] += 1
            return

        if entry is not None:
            _assign(node_data, compact_view(entry, ledger, include_spans=include_spans))
            stats["bound_observed"] += 1
        elif document_fallback:
            if PROVENANCE_NODE_ATTR not in node_data:
                node_data[PROVENANCE_NODE_ATTR] = {
                    "document_id": document_id,
                    "scope": "document",
                }
                stats["bound_document"] += 1
        elif PROVENANCE_NODE_ATTR not in node_data:
            node_data[PROVENANCE_NODE_ATTR] = {"status": "unresolved"}
            stats["unresolved"] += 1

    def _assign(node_data: dict[str, Any], view: dict[str, Any]) -> None:
        existing = node_data.get(PROVENANCE_NODE_ATTR)
        if existing and existing.get("status") != "unresolved":
            view = merge_compact_views(existing, view) or view
        node_data[PROVENANCE_NODE_ATTR] = view

    def _visit(instance: BaseModel, path: str) -> None:
        if id(instance) in visited:
            return
        visited.add(id(instance))
        _bind_instance(instance, path)
        for child_spec in children_by_parent.get(path, []):
            relative = _relative_path(path, child_spec.path)
            for child in _instances_at(instance, relative):
                _visit(child, child_spec.path)

    for model in models:
        if isinstance(model, BaseModel):
            _visit(model, "")

    if added_verbatim:
        ledger.resolution = "span"
    resolved = stats["bound_verbatim"] + stats["bound_observed"] + stats["bound_document"]
    logger.info(
        "Provenance binding: %s/%s node(s) grounded "
        "(%s verbatim, %s observed, %s document, %s unresolved)",
        resolved,
        stats["nodes_seen"],
        stats["bound_verbatim"],
        stats["bound_observed"],
        stats["bound_document"],
        stats["unresolved"],
    )
    ledger.bind_stats = dict(stats)
    return stats
