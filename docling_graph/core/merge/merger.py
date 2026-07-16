"""GraphMerger — deterministic union of exported knowledge graphs.

Node identity is the exported node ID (a content hash), so cross-graph entity
alignment reduces to key equality. Folding reuses the fill-empty policy the
in-run pipeline already lives by; ambiguous aliases are proposed into the
report, never auto-merged (a human decisions file replaces the LLM confirmer).
Same inputs always produce byte-identical output; merge(A, A) == A; the fold
is left-associative under input order and deliberately NOT commutative
(conflict winners depend on input order).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, TypeAlias

import networkx as nx
from pydantic import BaseModel, Field

from ...exceptions import ConfigurationError
from ...logging_utils import get_component_logger
from ..converters.graph_converter import _provenance_weight
from ..importers.graph_json import load_graph_input
from ..provenance.identity import PROVENANCE_NODE_ATTR
from ..provenance.models import ProvenanceLedger, template_schema_hash
from ..utils.alias_reconciler import (
    _META_ATTRS,
    _attr_richness,
    propose_alias_candidates,
    reconcile_graph_aliases,
)
from ..utils.graph_cleaner import GraphCleaner, validate_graph_structure
from ..utils.stats_calculator import calculate_graph_stats
from .identity import id_fields_by_template, rekey_graph
from .node_folder import (
    VARIANT_TYPE,
    build_conflict_variant,
    conflicting_scalar_fields,
    fold_edge,
    fold_node_attrs,
)
from .policy import MergePolicy
from .provenance_merge import merge_node_views, write_ledger_sidecars

logger = get_component_logger("GraphMerger", __name__)

MergeInputItem: TypeAlias = str | Path | nx.DiGraph

IdentitySource = Literal["template", "v2_export", "ledger", "node_ids"]


class MergeSource(BaseModel):
    """One merge input, as loaded (after duplicate absorption)."""

    index: int
    source: str
    document_id: str = ""
    template_name: str = ""
    template_schema_hash: str = ""
    node_count: int = 0
    edge_count: int = 0
    has_ledger: bool = False


class MergeReport(BaseModel):
    """Full audit record of one merge run (serialized as merge_report.json).

    Deliberately timestamp-free so re-running the same merge produces
    byte-identical artifacts.
    """

    sources: list[MergeSource] = Field(default_factory=list)
    duplicates_absorbed: list[str] = Field(default_factory=list)
    identity_source: IdentitySource = "node_ids"
    rekeyed: bool = False
    rekeyed_changed: int = 0
    nodes_folded: int = 0
    field_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    conflict_variants: int = 0
    """Variant sub-nodes created under ``conflicts: variants`` — one per
    (folded node, conflicting source) pair, holding the suppressed values."""
    edge_label_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    root_skolemized: list[dict[str, Any]] = Field(default_factory=list)
    cross_document_splits: list[dict[str, Any]] = Field(default_factory=list)
    """Same-ID collisions split instead of folded: the inputs share no root
    (so they are not re-extractions of one document) and folding would have
    overwritten conflicting scalar values (weak local identities such as line
    numbers minting equal IDs for unrelated instances). A proven conflict is
    contagious within its (document pair, class) group (``reason`` is
    ``field-conflict`` or ``same-class-conflict`` with ``triggered_by``)."""
    alias_candidates: list[dict[str, Any]] = Field(default_factory=list)
    alias_stats: dict[str, int] = Field(default_factory=dict)
    ignored_alias_decisions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    dry_run: bool = False

    @property
    def alias_merged(self) -> int:
        return int(self.alias_stats.get("merged", 0))


@dataclass
class _LoadedInput:
    """One resolved merge input."""

    index: int
    graph: nx.DiGraph
    ledger: ProvenanceLedger | None
    source: str
    graph_path: Path | None

    @property
    def document_id(self) -> str:
        if self.ledger is not None and self.ledger.document is not None:
            return self.ledger.document.document_id
        return ""


def _advisory_similarity(keep_display: str, merge_displays: list[str]) -> float | None:
    """Advisory rapidfuzz score for alias triage (never merges anything).

    Degrades to None when rapidfuzz is unavailable.
    """
    if not keep_display or not merge_displays:
        return None
    try:
        from rapidfuzz import fuzz
    except Exception:
        return None
    best = max(float(fuzz.token_sort_ratio(keep_display, d)) for d in merge_displays)
    return round(best / 100.0, 4)


class GraphMerger:
    """Deterministic multi-graph merge (see module docstring and design doc)."""

    def __init__(
        self,
        inputs: Sequence[MergeInputItem],
        template: type[BaseModel] | str | None = None,
        policy: MergePolicy | None = None,
    ) -> None:
        self.policy = policy or MergePolicy()
        self.template = self._resolve_template(template)
        self.report = MergeReport(dry_run=self.policy.dry_run)
        self.inputs = self._load_inputs(inputs)

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_template(
        template: type[BaseModel] | str | None,
    ) -> type[BaseModel] | None:
        if template is None:
            return None
        if isinstance(template, str):
            # Lazy import: core must not pull the pipeline package at load time.
            from ...pipeline.stages import TemplateLoadingStage

            return TemplateLoadingStage._load_from_string(template)
        if isinstance(template, type) and issubclass(template, BaseModel):
            return template
        raise ConfigurationError(
            "Invalid template type: expected a BaseModel subclass or dotted path",
            details={"type": type(template).__name__},
        )

    def _load_inputs(self, inputs: Sequence[MergeInputItem]) -> list[_LoadedInput]:
        if not inputs:
            raise ConfigurationError("merge requires at least one input graph")
        loaded: list[_LoadedInput] = []
        seen: dict[str, str] = {}
        for position, raw in enumerate(inputs):
            if isinstance(raw, nx.DiGraph):
                # Deep copy: folding mutates attribute values (list unions), and
                # the caller's graph must survive the merge untouched.
                graph = copy.deepcopy(raw)
                ledger, graph_path = None, None
                source = f"graph-object-{position}"
                dedup_key = f"object:{id(raw)}"
            else:
                input_path = Path(raw)
                graph, ledger, graph_path = load_graph_input(input_path)
                source = str(input_path)
                document_id = (
                    ledger.document.document_id
                    if ledger is not None and ledger.document is not None
                    else ""
                )
                dedup_key = f"doc:{document_id}" if document_id else f"path:{graph_path.resolve()}"
            if dedup_key in seen:
                message = (
                    f"Duplicate input absorbed: {source} carries the same "
                    f"{'document' if dedup_key.startswith('doc:') else 'graph'} "
                    f"as {seen[dedup_key]}"
                )
                logger.warning(message)
                self.report.duplicates_absorbed.append(source)
                self.report.warnings.append(message)
                continue
            seen[dedup_key] = source
            item = _LoadedInput(
                index=len(loaded),
                graph=graph,
                ledger=ledger,
                source=source,
                graph_path=graph_path,
            )
            loaded.append(item)
            document = ledger.document if ledger is not None else None
            self.report.sources.append(
                MergeSource(
                    index=item.index,
                    source=source,
                    document_id=item.document_id,
                    template_name=(
                        document.template_name
                        if document is not None and document.template_name
                        else str(graph.graph.get("template_name") or "")
                    ),
                    template_schema_hash=(
                        document.template_schema_hash
                        if document is not None and document.template_schema_hash
                        else str(graph.graph.get("template_schema_hash") or "")
                    ),
                    node_count=graph.number_of_nodes(),
                    edge_count=graph.number_of_edges(),
                    has_ledger=ledger is not None,
                )
            )
        return loaded

    # ------------------------------------------------------------------ merge

    def merge(self) -> tuple[nx.DiGraph, MergeReport]:
        """Run the full merge plan and return (merged graph, report)."""
        self._check_template_compatibility()
        id_fields_map, identity_source = self._resolve_identity()
        self.report.identity_source = identity_source

        rekey = self.policy.rekey
        if rekey is None:
            rekey = identity_source != "node_ids"
        elif rekey and identity_source == "node_ids":
            raise ConfigurationError(
                "Re-keying requires an id-fields source: pass --template, or merge "
                "format-v2 exports / dense-contract ledgers"
            )
        if rekey:
            changed_total = 0
            for item in self.inputs:
                item.graph, changed, field_conflicts, edge_conflicts = rekey_graph(
                    item.graph, id_fields_map, self.policy, item.source
                )
                changed_total += changed
                self.report.field_conflicts.extend(field_conflicts)
                self.report.edge_label_conflicts.extend(edge_conflicts)
            self.report.rekeyed = True
            self.report.rekeyed_changed = changed_total

        self._skolemize_root_collisions(id_fields_map)
        self._split_conflicting_collisions()
        graph = self._union_fold()
        GraphCleaner(verbose=True).clean_graph(graph)
        self._alias_pass(graph, id_fields_map)
        validate_graph_structure(graph, raise_on_error=True)

        metadata = calculate_graph_stats(graph, len(self.inputs))
        self.report.node_count = metadata.node_count
        self.report.edge_count = metadata.edge_count
        self.report.node_types = dict(metadata.node_types)
        self.report.edge_types = dict(metadata.edge_types)
        self._stamp_graph_metadata(graph, id_fields_map)
        logger.info(
            "Merged %s input(s): %s nodes, %s edges (%s folded, %s field conflict(s))",
            len(self.inputs),
            metadata.node_count,
            metadata.edge_count,
            self.report.nodes_folded,
            len(self.report.field_conflicts),
        )
        return graph, self.report

    def _check_template_compatibility(self) -> None:
        """Schema-hash gate: same-template inputs merge safely by construction."""
        hashes: dict[str, list[str]] = {}
        missing: list[str] = []
        for source_info in self.report.sources:
            schema_hash = source_info.template_schema_hash
            if schema_hash:
                hashes.setdefault(schema_hash, []).append(source_info.source)
            else:
                missing.append(source_info.source)
        if self.template is not None:
            # An explicit --template joins the gate like an input: re-keying
            # and alias proposal under a schema the inputs were not extracted
            # with is exactly the mismatch this check exists to catch.
            hashes.setdefault(template_schema_hash(self.template), []).append(
                f"--template {self.template.__name__}"
            )
        if len(hashes) > 1:
            details = {schema_hash[:12]: sources for schema_hash, sources in hashes.items()}
            if self.policy.strict_template_check:
                raise ConfigurationError(
                    "Inputs were extracted with different template schemas; refusing to "
                    "merge. Use --no-strict-template-check to merge anyway (same-named "
                    "classes from different templates will then merge by node ID).",
                    details={"schema_hashes": details},
                )
            message = (
                "Template schemas differ across inputs; merging anyway — same-named "
                f"classes from different templates merge by node ID: {details}"
            )
            logger.warning(message)
            self.report.warnings.append(message)
        if missing and len(self.inputs) > 1:
            message = (
                "Template compatibility check skipped for inputs without a schema hash: "
                + ", ".join(missing)
            )
            logger.warning(message)
            self.report.warnings.append(message)

    def _resolve_identity(self) -> tuple[dict[str, list[str]], IdentitySource]:
        """Resolve {class: graph_id_fields} in priority order (design §5.4):
        template walk > format-v2 embedded map > dense-ledger ids keys."""
        if self.template is not None:
            return id_fields_by_template(self.template), "template"
        v2_map: dict[str, list[str]] = {}
        found_v2 = False
        for item in self.inputs:
            raw = item.graph.graph.get("id_fields_map")
            if isinstance(raw, dict) and raw:
                found_v2 = True
                for cls, fields in raw.items():
                    if cls not in v2_map and isinstance(fields, list):
                        v2_map[str(cls)] = [f for f in fields if isinstance(f, str)]
        if found_v2:
            return v2_map, "v2_export"
        ledger_map: dict[str, set[str]] = {}
        for item in self.inputs:
            if item.ledger is None or not item.ledger.nodes:
                continue
            for entry in item.ledger.nodes.values():
                if entry.node_type and entry.ids:
                    ledger_map.setdefault(entry.node_type, set()).update(entry.ids.keys())
        if ledger_map:
            return {cls: sorted(fields) for cls, fields in ledger_map.items()}, "ledger"
        return {}, "node_ids"

    def _skolemize_root_collisions(self, id_fields_map: dict[str, list[str]]) -> None:
        """Split filename-stem root collisions (design §5.8).

        Only when ledgers prove distinct documents (document_ids differ), root
        node IDs match, AND the shared identity value equals a source-file
        stem — the provably filename-derived case ``repair_root_identity`` can
        produce. Content-derived identities are never touched. Without
        ledgers, colliding roots merge with a loud warning naming both
        sources.
        """
        first_owner: dict[str, _LoadedInput] = {}
        for item in self.inputs:
            renames: dict[str, str] = {}
            for node_id in list(item.graph.nodes):
                owner = first_owner.get(str(node_id))
                if owner is None:
                    continue
                if item.graph.in_degree(node_id) != 0 or owner.graph.in_degree(node_id) != 0:
                    continue
                if item.ledger is None or owner.ledger is None:
                    message = (
                        f"Root node {node_id} appears in both {owner.source} and "
                        f"{item.source}; no provenance ledgers to verify distinct "
                        "documents — merging them"
                    )
                    logger.warning(message)
                    self.report.warnings.append(message)
                    continue
                if item.document_id == owner.document_id:
                    continue
                attrs = item.graph.nodes[node_id]
                cls = str(attrs.get("__class__") or "")
                fields = id_fields_map.get(cls) or []
                stems = {
                    Path(ledger.document.source).stem
                    for ledger in (item.ledger, owner.ledger)
                    if ledger.document is not None and ledger.document.source
                }
                if fields:
                    stem_value = next(
                        (
                            attrs.get(field)
                            for field in fields
                            if isinstance(attrs.get(field), str) and attrs.get(field) in stems
                        ),
                        None,
                    )
                    if stem_value is None:
                        # Content-derived identity: a legitimate cross-document
                        # fold, never skolemized.
                        continue
                else:
                    # No declared id fields (v1 export without --template,
                    # direct-contract ledger): fall back to scanning the root's
                    # scalar string attributes for a filename-stem match.
                    stem_value = next(
                        (
                            value
                            for field, value in attrs.items()
                            if field not in _META_ATTRS
                            and isinstance(value, str)
                            and value in stems
                        ),
                        None,
                    )
                    if stem_value is None:
                        message = (
                            f"Root node {node_id} appears in both {owner.source} and "
                            f"{item.source} (distinct documents per their ledgers), but "
                            "its class declares no id fields and no attribute matches a "
                            "source filename stem — cannot rule out cross-document "
                            "fusion; merging them"
                        )
                        logger.warning(message)
                        self.report.warnings.append(message)
                        continue
                new_id = f"{node_id}__doc_{item.document_id[:8]}"
                renames[str(node_id)] = new_id
                # Content-bearing stamp: without it, two byte-identical roots
                # would be re-fused by GraphCleaner's content-hash dedup right
                # after the split.
                attrs["skolem_document_id"] = item.document_id
                self.report.root_skolemized.append(
                    {
                        "original_id": str(node_id),
                        "skolemized_id": new_id,
                        "identity_value": stem_value,
                        "document_id": item.document_id,
                        "source": item.source,
                        "collided_with": owner.source,
                    }
                )
                logger.warning(
                    "Root skolemized: %s in %s shares a filename-stem identity (%r) "
                    "with %s — renamed to %s",
                    node_id,
                    item.source,
                    stem_value,
                    owner.source,
                    new_id,
                )
            if renames:
                nx.relabel_nodes(item.graph, renames, copy=False)
                for new_id in renames.values():
                    item.graph.nodes[new_id]["id"] = new_id
            for node_id in item.graph.nodes:
                first_owner.setdefault(str(node_id), item)

    @staticmethod
    def _root_scope(graph: nx.DiGraph, node_id: str, roots: set[str]) -> set[str]:
        """The root node IDs a node hangs under (itself, when it is a root)."""
        scope = {str(ancestor) for ancestor in nx.ancestors(graph, node_id)}
        scope.add(str(node_id))
        return scope & roots

    def _split_conflicting_collisions(self) -> None:
        """Veto cross-document folds that would overwrite conflicting content.

        Node IDs are content hashes of identity fields, but some identities
        are only locally unique (line numbers, positions, step indexes): two
        unrelated documents can mint the same ID for different instances, and
        folding them corrupts both (chimera descriptions, clobbered amounts,
        children shared across roots). The guard is structural + content-based:

        - the two occurrences hang under **no common root node ID** (per-node
          ancestry, so multi-document containers like re-merged outputs stay
          honest) — occurrences sharing a root are re-extractions of the same
          logical document (e.g. a JPG and a DOCX of one invoice), where
          same-ID children are the same instance and extraction noise
          legitimately folds under keep-first; and
        - folding would trip at least one rule-8 scalar conflict
          (:func:`conflicting_scalar_fields`) — fill-empty-compatible
          occurrences still fold, so shared entities (a Party appearing in
          two invoices) keep folding by identity.

        One proven conflict is contagious within its (document pair, class)
        group: a scalar conflict on any collision of class C between inputs
        A and B is proof that C's identity fields under-determine instances
        across these documents, so every C-collision between A and B splits —
        line 2 of two unrelated invoices is a different instance even when
        its extracted values happen to agree, and folding it would share one
        child across both roots. Other classes are untouched: a compatible
        Party still folds even while the LineItems split.

        A both-roots collision has itself as its scope on both sides, so the
        shared-scope carve-out defers it to the root-collision policy above.
        The later input's node is renamed ``<id>__doc_<document_id[:8]>``
        (``__src_<index>`` without a ledger) and stamped with
        ``skolem_document_id`` so re-keying on a future re-merge does not
        silently re-fuse the pair (same mechanism as root skolemization).
        Every split is recorded on ``report.cross_document_splits``.
        """
        roots_by_input = [
            {str(node) for node in item.graph.nodes if item.graph.in_degree(node) == 0}
            for item in self.inputs
        ]
        first_owner: dict[str, _LoadedInput] = {}
        for item in self.inputs:
            collisions: list[tuple[str, _LoadedInput, list[str], str]] = []
            for node_id in item.graph.nodes:
                key = str(node_id)
                owner = first_owner.get(key)
                if owner is None:
                    continue
                owner_scope = self._root_scope(owner.graph, key, roots_by_input[owner.index])
                item_scope = self._root_scope(item.graph, key, roots_by_input[item.index])
                if owner_scope & item_scope:
                    continue
                conflicts = conflicting_scalar_fields(
                    owner.graph.nodes[key], item.graph.nodes[key], self.policy
                )
                cls = str(item.graph.nodes[key].get("__class__") or "")
                collisions.append((key, owner, conflicts, cls))
            # Conflict contagion: the first conflicting collision of each
            # (owner input, class) group condemns the whole group.
            trigger_by_group: dict[tuple[int, str], str] = {}
            for key, owner, conflicts, cls in collisions:
                if conflicts and cls:
                    trigger_by_group.setdefault((owner.index, cls), key)
            renames: dict[str, str] = {}
            for key, owner, conflicts, cls in collisions:
                trigger = trigger_by_group.get((owner.index, cls)) if cls else None
                if not conflicts and trigger is None:
                    continue
                attrs = item.graph.nodes[key]
                if item.document_id:
                    stamp, new_id = item.document_id, f"{key}__doc_{item.document_id[:8]}"
                else:
                    stamp, new_id = f"input-{item.index}", f"{key}__src_{item.index}"
                attrs["skolem_document_id"] = stamp
                renames[key] = new_id
                record: dict[str, Any] = {
                    "original_id": key,
                    "split_id": new_id,
                    "class": cls,
                    "conflicting_fields": conflicts,
                    "reason": "field-conflict" if conflicts else "same-class-conflict",
                    "document_id": item.document_id,
                    "source": item.source,
                    "collided_with": owner.source,
                }
                if not conflicts:
                    record["triggered_by"] = trigger
                self.report.cross_document_splits.append(record)
                if conflicts:
                    logger.warning(
                        "Cross-document split: %s in %s collides with %s but the inputs "
                        "share no root and %d field(s) conflict (%s) — folded IDs from "
                        "unrelated documents describe different instances; renamed to %s",
                        key,
                        item.source,
                        owner.source,
                        len(conflicts),
                        ", ".join(conflicts),
                        new_id,
                    )
                else:
                    logger.warning(
                        "Cross-document split (same-class conflict): %s in %s collides "
                        "with %s without field conflicts, but same-class collision %s "
                        "from the same document pair does conflict — %s identities "
                        "under-determine instances across these documents; renamed to %s",
                        key,
                        item.source,
                        owner.source,
                        trigger,
                        cls,
                        new_id,
                    )
            if renames:
                nx.relabel_nodes(item.graph, renames, copy=False)
                roots = roots_by_input[item.index]
                for old_id, new_id in renames.items():
                    item.graph.nodes[new_id]["id"] = new_id
                    if old_id in roots:
                        roots.discard(old_id)
                        roots.add(new_id)
            for node_id in item.graph.nodes:
                first_owner.setdefault(str(node_id), item)

    def _source_tag(self, position: int) -> str:
        return self.inputs[position].source

    def _union_fold(self) -> nx.DiGraph:
        """Accumulate the union: fold same-ID nodes (§5.3), union edges (§5.7),
        wrap cross-document provenance (§5.5), record merged_from. Under
        ``conflicts: variants``, each fold's suppressed values are reified as
        variant sub-nodes after the union (:func:`build_conflict_variant`)."""
        node_order: list[str] = []
        node_occurrences: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        edge_order: list[tuple[str, str]] = []
        edge_occurrences: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        for position, item in enumerate(self.inputs):
            for node_id, attrs in item.graph.nodes(data=True):
                key = str(node_id)
                if key not in node_occurrences:
                    node_order.append(key)
                    node_occurrences[key] = []
                node_occurrences[key].append((position, attrs))
            for source, target, attrs in item.graph.edges(data=True):
                edge_key = (str(source), str(target))
                if edge_key not in edge_occurrences:
                    edge_order.append(edge_key)
                    edge_occurrences[edge_key] = []
                edge_occurrences[edge_key].append((position, attrs))

        merged = nx.DiGraph()
        folded = 0
        variant_specs: list[tuple[str, int, dict[str, Any], Any]] = []
        for node_id in node_order:
            occurrences = node_occurrences[node_id]
            if len(occurrences) == 1:
                merged.add_node(node_id, **occurrences[0][1])
                continue
            folded += len(occurrences) - 1
            ordered = occurrences
            if self.policy.precedence == "richest":
                # Total order: richness desc, wrapped-aware provenance weight
                # desc, input index asc as the stable tiebreak.
                ordered = sorted(
                    occurrences,
                    key=lambda occ: (
                        -_attr_richness(occ[1]),
                        -_provenance_weight(occ[1]),
                        occ[0],
                    ),
                )
            base = dict(ordered[0][1])
            provenance = base.get(PROVENANCE_NODE_ATTR)
            suppressed = list(base.get("__conflicts__") or [])
            merged_from = [dict(entry) for entry in (base.get("merged_from") or [])]
            merged_aliases = list(base.get("merged_aliases") or [])
            for position, attrs in ordered[1:]:
                records = fold_node_attrs(base, attrs, self.policy, self._source_tag(position))
                self.report.field_conflicts.extend(records)
                if (
                    self.policy.conflicts == "variants"
                    and records
                    # A variant never spawns variants of its own: a colliding
                    # variant pair folds keep-first like any other node.
                    and str(base.get("type") or "") != VARIANT_TYPE
                ):
                    dropped = {r["field"]: r["dropped"] for r in records}
                    variant_specs.append(
                        (node_id, position, dropped, attrs.get(PROVENANCE_NODE_ATTR))
                    )
                provenance = merge_node_views(provenance, attrs.get(PROVENANCE_NODE_ATTR))
                # Audit trails from previous merges survive re-merging.
                for entry in attrs.get("merged_from") or []:
                    if entry not in merged_from:
                        merged_from.append(dict(entry))
                for entry in attrs.get("merged_aliases") or []:
                    if entry not in merged_aliases:
                        merged_aliases.append(dict(entry) if isinstance(entry, dict) else entry)
                for entry in attrs.get("__conflicts__") or []:
                    if entry not in suppressed:
                        suppressed.append(entry)
            if provenance is not None:
                base[PROVENANCE_NODE_ATTR] = provenance
            if suppressed:
                base["__conflicts__"] = suppressed
            if merged_aliases:
                base["merged_aliases"] = merged_aliases
            for position, _attrs in ordered:
                item = self.inputs[position]
                entry = {"document_id": item.document_id, "source": item.source}
                if entry not in merged_from:
                    merged_from.append(entry)
            base["merged_from"] = merged_from
            merged.add_node(node_id, **base)

        for edge_key in edge_order:
            source, target = edge_key
            occurrences = edge_occurrences[edge_key]
            base = dict(occurrences[0][1])
            for position, attrs in occurrences[1:]:
                record = fold_edge(
                    source, target, base, attrs, self.policy, self._source_tag(position)
                )
                if record is not None:
                    self.report.edge_label_conflicts.append(record)
            merged.add_edge(source, target, **base)

        for base_id, position, dropped, provenance in variant_specs:
            item = self.inputs[position]
            stamp = item.document_id or f"input-{position}"
            variant_id, attrs, edge_attrs = build_conflict_variant(
                merged.nodes[base_id], dropped, stamp, provenance
            )
            if variant_id in merged:
                # Re-merge of an export that already carries this variant: the
                # deterministic id makes the re-detected conflict a no-op.
                continue
            merged.add_node(variant_id, **attrs)
            merged.add_edge(base_id, variant_id, **edge_attrs)
            self.report.conflict_variants += 1

        self.report.nodes_folded = folded
        return merged

    # ------------------------------------------------------------------ alias

    def _alias_pass(self, graph: nx.DiGraph, id_fields_map: dict[str, list[str]]) -> None:
        """Propose alias candidates into the report; apply human decisions only."""
        if not any(id_fields_map.values()):
            message = (
                "Alias reconciliation skipped: no id-fields source available "
                "(pass --template, or merge format-v2 exports)"
            )
            logger.info(message)
            self.report.warnings.append(message)
            return
        node_ids_by_class, display_by_class, groups = propose_alias_candidates(graph, id_fields_map)
        self.report.alias_candidates = self._candidate_stubs(
            graph, id_fields_map, node_ids_by_class, display_by_class, groups
        )
        decisions_fn: Callable[..., Any] | None = None
        confirmed_pairs: list[dict[str, str]] = []
        if self.policy.alias_decisions is not None:
            decisions_fn = self._build_decisions_fn(node_ids_by_class, groups, confirmed_pairs)
        nodes_before = set(graph.nodes)
        self.report.alias_stats = reconcile_graph_aliases(
            graph, id_fields_map, llm_call_fn=decisions_fn, context="merge"
        )
        # Human confirmations the reconciler's own guards vetoed (e.g. sibling
        # co-occurrence) must not disappear silently: an applied merge always
        # removes one node of its pair, so a fully surviving pair was vetoed.
        removed = nodes_before - set(graph.nodes)
        for pair in confirmed_pairs:
            if pair["keep_id"] in removed or pair["merge_id"] in removed:
                continue
            self.report.ignored_alias_decisions.append(
                {
                    "class": pair["class"],
                    "keep_id": pair["keep_id"],
                    "merge_ids": [pair["merge_id"]],
                    "reason": "vetoed by reconciliation guards (e.g. sibling co-occurrence)",
                }
            )

    def _candidate_stubs(
        self,
        graph: nx.DiGraph,
        id_fields_map: dict[str, list[str]],
        node_ids_by_class: dict[str, list[str]],
        display_by_class: dict[str, list[str]],
        groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ready-to-edit decision stubs (flip "confirm" and re-run with
        --alias-decisions). Similarity scores are advisory triage only;
        ``field_conflicts`` lists the content fields a confirmed merge would
        contradict (identity fields excluded — the displays already show
        them) — many conflicts usually mean distinct instances that happen
        to share an identifier, not aliases."""
        stubs: list[dict[str, Any]] = []
        for group in groups:
            cls = str(group["class"])
            node_ids = node_ids_by_class.get(cls, [])
            displays = display_by_class.get(cls, [])

            def _display(index: int, displays: list[str] = displays) -> str:
                return displays[index] if 0 <= index < len(displays) else ""

            keep = int(group["keep"])
            merge_indexes = [int(m) for m in group["merge"]]
            keep_display = _display(keep)
            merge_displays = [_display(m) for m in merge_indexes]
            merge_ids = [node_ids[m] for m in merge_indexes]
            identity_fields = set(id_fields_map.get(cls) or [])
            field_conflicts = sorted(
                {
                    field
                    for merge_id in merge_ids
                    for field in conflicting_scalar_fields(
                        graph.nodes[node_ids[keep]], graph.nodes[merge_id], self.policy
                    )
                }
                - identity_fields
            )
            stubs.append(
                {
                    "class": cls,
                    "keep_id": node_ids[keep],
                    "keep_display": keep_display,
                    "merge_ids": merge_ids,
                    "merge_displays": merge_displays,
                    "similarity": _advisory_similarity(keep_display, merge_displays),
                    "field_conflicts": field_conflicts,
                    "confirm": False,
                }
            )
        return stubs

    def _build_decisions_fn(
        self,
        node_ids_by_class: dict[str, list[str]],
        groups: list[dict[str, Any]],
        confirmed_pairs: list[dict[str, str]],
    ) -> Callable[..., Any]:
        """Human decisions file -> llm_call_fn closure (index translation).

        Decision stubs carry node IDs; ``reconcile_graph_aliases`` consumes
        per-class index groups. Confirmed node-ID groups are translated into
        the current propose run's index space; ids absent from the run and
        unproposed pairs are reported as ignored (the reconciler's own guards
        would drop them silently either way). Every pair handed to the
        reconciler is also appended to ``confirmed_pairs`` so the caller can
        diff confirmations against outcomes.
        """
        decisions = self._load_decisions(self.policy.alias_decisions)
        index_by_class = {
            cls: {node_id: index for index, node_id in enumerate(node_ids)}
            for cls, node_ids in node_ids_by_class.items()
        }
        proposed_pairs = {
            (str(g["class"]), int(g["keep"]), int(m)) for g in groups for m in g["merge"]
        }
        merges: list[dict[str, Any]] = []
        for decision in decisions:
            if not isinstance(decision, dict) or not decision.get("confirm"):
                continue
            cls = str(decision.get("class") or "")
            keep_id = str(decision.get("keep_id") or "")
            merge_ids = [str(m) for m in (decision.get("merge_ids") or [])]
            index_of = index_by_class.get(cls) or {}
            keep = index_of.get(keep_id)
            if keep is None:
                self.report.ignored_alias_decisions.append(
                    {
                        "class": cls,
                        "keep_id": keep_id,
                        "merge_ids": merge_ids,
                        "reason": "keep_id not present in the merged graph",
                    }
                )
                continue
            confirmed: list[int] = []
            for merge_id in merge_ids:
                merge_index = index_of.get(merge_id)
                if merge_index is None:
                    reason = "merge id not present in the merged graph"
                elif (cls, keep, merge_index) not in proposed_pairs and (
                    cls,
                    merge_index,
                    keep,
                ) not in proposed_pairs:
                    reason = "pair was not proposed in this run"
                else:
                    confirmed.append(merge_index)
                    confirmed_pairs.append({"class": cls, "keep_id": keep_id, "merge_id": merge_id})
                    continue
                self.report.ignored_alias_decisions.append(
                    {
                        "class": cls,
                        "keep_id": keep_id,
                        "merge_ids": [merge_id],
                        "reason": reason,
                    }
                )
            if confirmed:
                merges.append({"class": cls, "keep": keep, "merge": confirmed})

        def decisions_fn(
            prompt: Any = None, schema_json: Any = None, context: Any = None, **_: Any
        ) -> dict[str, Any]:
            return {"merges": merges}

        return decisions_fn

    @staticmethod
    def _load_decisions(path: Path | None) -> list[dict[str, Any]]:
        """Read a decisions file: a merge report (alias_candidates key) or a bare list."""
        if path is None:
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigurationError(
                f"Cannot read alias decisions file: {path}",
                details={"path": str(path)},
                cause=e,
            ) from e
        if isinstance(data, dict):
            data = data.get("alias_candidates")
        if not isinstance(data, list):
            raise ConfigurationError(
                "Alias decisions must be a JSON list of candidate stubs or a merge "
                "report with an 'alias_candidates' key",
                details={"path": str(path)},
            )
        return [d for d in data if isinstance(d, dict)]

    # ----------------------------------------------------------------- output

    def _stamp_graph_metadata(self, graph: nx.DiGraph, id_fields_map: dict[str, list[str]]) -> None:
        """Fresh graph-level metadata (§5.9); the merged export stays
        format-v2 self-describing so it can itself be re-merged."""
        template_names = {s.template_name for s in self.report.sources if s.template_name}
        schema_hashes = {
            s.template_schema_hash for s in self.report.sources if s.template_schema_hash
        }
        if self.template is not None:
            graph.graph["template_name"] = self.template.__name__
            # The template drove identity for this merge; its hash (already
            # gate-checked against the inputs) is the authoritative stamp.
            graph.graph["template_schema_hash"] = template_schema_hash(self.template)
        else:
            graph.graph["template_name"] = (
                next(iter(template_names)) if len(template_names) == 1 else ""
            )
            graph.graph["template_schema_hash"] = (
                next(iter(schema_hashes)) if len(schema_hashes) == 1 else ""
            )
        graph.graph["format"] = "docling-graph/v2"
        graph.graph["id_fields_map"] = {cls: list(fields) for cls, fields in id_fields_map.items()}
        graph.graph["merge"] = {
            "sources": [s.model_dump(mode="json") for s in self.report.sources],
            "identity_source": self.report.identity_source,
            "nodes_folded": self.report.nodes_folded,
            "field_conflicts": self.report.field_conflicts,
            "conflict_variants": self.report.conflict_variants,
            "edge_label_conflicts": self.report.edge_label_conflicts,
            "cross_document_splits": self.report.cross_document_splits,
            "alias_candidates": self.report.alias_candidates,
            "alias_merged": self.report.alias_merged,
            "rekeyed": self.report.rekeyed,
            "rekeyed_changed": self.report.rekeyed_changed,
        }

    def write_provenance_sidecars(self, provenance_dir: Path) -> Path:
        """Write each input's ledger verbatim + manifest.json (design §5.5)."""
        entries: list[tuple[dict[str, Any], ProvenanceLedger | None]] = []
        for item in self.inputs:
            document = item.ledger.document if item.ledger is not None else None
            source_info = self.report.sources[item.index]
            entries.append(
                (
                    {
                        "index": item.index,
                        "document_id": item.document_id,
                        "source": item.source,
                        "template_name": source_info.template_name,
                        "template_schema_hash": source_info.template_schema_hash,
                        "converted_at": (
                            document.converted_at.isoformat() if document is not None else None
                        ),
                        "graph": str(item.graph_path) if item.graph_path is not None else None,
                    },
                    item.ledger,
                )
            )
        return write_ledger_sidecars(entries, provenance_dir)


def merge_graphs(
    inputs: Sequence[MergeInputItem],
    template: type[BaseModel] | str | None = None,
    policy: MergePolicy | None = None,
) -> tuple[nx.DiGraph, MergeReport]:
    """Merge exported knowledge graphs deterministically.

    Args:
        inputs: Run directories, graph.json paths, or in-memory ``nx.DiGraph``
            objects (in precedence order — the first graph is the base).
        template: Optional extraction template (class or ``module.Class``
            dotted path); enables re-keying and alias proposal.
        policy: Merge policy; defaults to :class:`MergePolicy`.

    Returns:
        Tuple of (merged graph, merge report).
    """
    merger = GraphMerger(inputs, template=template, policy=policy)
    return merger.merge()
