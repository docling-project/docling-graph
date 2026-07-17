"""
Handles conversion of Pydantic models to NetworkX graph structure.

This module provides the GraphConverter class for converting Pydantic models
into directed graphs with nodes and edges, including features like stable node
IDs, edge metadata, bidirectional edges, and automatic cleanup.

Key Concepts:
- Entities (is_entity=True or default): Become separate nodes with edges
- Components (is_entity=False): Embedded as dictionaries in parent nodes
- Entities nested inside components still become nodes, linked by an edge
  from the nearest entity ancestor (consistent with the dense catalog walk)
- Duplicate instances of the same entity enrich the first node's missing
  attributes instead of being silently dropped
"""

from typing import Any, List, Mapping, Optional, Set

import networkx as nx
from pydantic import BaseModel

from ...exceptions import GraphError
from ...logging_utils import get_component_logger
from ..provenance.identity import PROVENANCE_NODE_ATTR, iter_provenance_views
from ..provenance.models import template_schema_hash
from ..utils.alias_reconciler import _attr_richness, id_fields_by_class, reconcile_graph_aliases
from ..utils.entity_name_normalizer import canonicalize_identity_for_dedup
from ..utils.graph_cleaner import GraphCleaner, validate_graph_structure
from ..utils.stats_calculator import calculate_graph_stats
from .config import GraphConfig
from .models import Edge, GraphMetadata
from .node_id_registry import NodeIDRegistry

logger = get_component_logger("GraphConverter", __name__)


def get_model_config_value(model: BaseModel, key: str, default: Any) -> Any:
    """
    Safely get configuration value from Pydantic model's model_config.

    Handles both dict-like and object-like config access patterns.

    Args:
        model: Pydantic model instance
        key: Configuration key to retrieve
        default: Default value if key not found

    Returns:
        Configuration value or default

    Examples:
        >>> is_entity = get_model_config_value(model, "is_entity", True)
        >>> id_fields = get_model_config_value(model, "graph_id_fields", [])
    """
    config = model.model_config
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _empty_identity_node_ids(graph: nx.DiGraph, id_fields_map: dict[str, list[str]]) -> list[str]:
    """Node ids whose class declares identity fields that are all empty on the node."""

    def _filled(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    out: list[str] = []
    for node_id, attrs in graph.nodes(data=True):
        fields = id_fields_map.get(str(attrs.get("__class__") or ""))
        if fields and not any(_filled(attrs.get(f)) for f in fields):
            out.append(str(node_id))
    return out


def _is_empty_value(value: Any) -> bool:
    """True for values that carry no data (None or empty str/list/dict).

    Numeric zero and False are meaningful and count as non-empty.
    """
    if value is None:
        return True
    if isinstance(value, str | list | dict):
        return len(value) == 0
    return False


def _collect_cardinality_bounds(model_instances: List[BaseModel]) -> dict[str, int]:
    """{class name: graph_max_instances} for every model class reachable below
    the instances. The bound is a template-declared safety rail against
    discovery spam (e.g. hundreds of financial-table rows promoted to a
    segments[] class whose docstring documents 3-6 real instances); classes
    without the key are unbounded."""
    bounds: dict[str, int] = {}
    seen: set[int] = set()

    def _visit(instance: BaseModel) -> None:
        if id(instance) in seen:
            return
        seen.add(id(instance))
        raw = get_model_config_value(instance, "graph_max_instances", None)
        cls = instance.__class__.__name__
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
            bounds[cls] = raw
        elif raw is not None and cls not in bounds:
            logger.warning("Ignoring invalid graph_max_instances=%r on %s", raw, cls)
        for _name, value in instance:
            if isinstance(value, BaseModel):
                _visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, BaseModel):
                        _visit(item)

    for model in model_instances:
        _visit(model)
    return bounds


def _provenance_weight(node_data: dict[str, Any]) -> int:
    """Distinct-chunk support recorded by the grounding binder (0 when off).

    Wrapped multi-document views (produced by cross-document graph merging)
    sum the chunk support of every source, so re-merged nodes never score 0.
    """
    prov = node_data.get(PROVENANCE_NODE_ATTR)
    if not isinstance(prov, dict):
        return 0
    weight = 0
    for view in iter_provenance_views(prov):
        chunks = view.get("chunks")
        weight += len(chunks) if isinstance(chunks, list) else 0
        omitted = view.get("chunks_omitted")
        if isinstance(omitted, int) and omitted > 0:
            weight += omitted
    return weight


class GraphConverter:
    """Converts Pydantic models to NetworkX graphs with enhanced features.

    This converter supports:
    - Deterministic node ID generation via NodeIDRegistry
    - Automatic graph cleanup (phantom nodes, duplicates)
    - Stable node IDs across batch extractions
    - Bidirectional edges
    - Full validation

    This converter is stateless and thread-safe. All conversion state is managed
    through method parameters rather than instance variables.
    """

    def __init__(
        self,
        config: GraphConfig | None = None,
        add_reverse_edges: bool = False,
        validate_graph: bool = True,
        registry: NodeIDRegistry | None = None,
        auto_cleanup: bool = True,
        alias_llm_fn: Any | None = None,
        enforce_cardinality_bounds: bool = True,
    ) -> None:
        """
        Initialize the graph converter.

        Args:
            config: Graph configuration (optional)
            add_reverse_edges: Create bidirectional edges (default: False)
            validate_graph: Validate graph structure (default: True)
            registry: NodeIDRegistry for deterministic node IDs across batches.
                If None, creates a new registry per conversion (works for single-batch).
                Pass a shared registry for cross-batch consistency.
            auto_cleanup: Automatically cleanup graph after conversion,
                removing phantom nodes, duplicates, orphaned edges (default: True)
            alias_llm_fn: Optional id-space LLM callable
                ``fn(prompt=..., schema_json=..., context=...)`` used to CONFIRM
                deterministic same-class alias candidates (short table label vs
                full section title). Without it the alias pass is propose-only:
                candidates are logged, nothing is merged.
            enforce_cardinality_bounds: Enforce template-declared
                ``graph_max_instances`` class bounds after cleanup, demoting the
                least-supported surplus instances (default: True). Templates
                without the marker are unaffected either way.
        """
        self.config = config or GraphConfig()
        # Use parameter value directly (don't use 'or' which would make False use config default)
        self.add_reverse_edges = add_reverse_edges
        self.validate_graph = validate_graph

        # Initialize registry (use provided or create new)
        self.registry = registry or NodeIDRegistry()

        # Initialize cleaner for automatic cleanup
        self.auto_cleanup = auto_cleanup
        self.cleaner = GraphCleaner(verbose=True) if auto_cleanup else None
        self.alias_llm_fn = alias_llm_fn
        self.enforce_cardinality_bounds = enforce_cardinality_bounds

    def pydantic_list_to_graph(
        self,
        model_instances: List[BaseModel],
        provenance_binder: Any | None = None,
    ) -> tuple[nx.DiGraph, GraphMetadata]:
        """
        Convert list of Pydantic models to a NetworkX graph.

        Process:
        1. Pre-register all models for deterministic node IDs
        2. Create nodes from models
        3. Create edges between entities
        4. Bind provenance (if a binder is provided) — before cleanup, so
           duplicate-node merging can union provenance instead of losing it
        5. Apply automatic cleanup (if enabled)
        6. Validate graph structure
        7. Calculate statistics

        Args:
            model_instances: List of Pydantic model instances to convert
            provenance_binder: Optional callable ``(graph, model_instances) -> None``
                that annotates nodes with provenance. The converter stays
                agnostic of the provenance module; the pipeline stage injects
                a closure.

        Returns:
            Tuple of (graph, metadata)
        """
        if not model_instances:
            raise GraphError(
                "Cannot create graph from empty model list",
                details={"reason": "no_models_extracted"},
            )

        # Pre-register all models to ensure consistent node IDs across batches
        logger.info("Pre-registering models for deterministic node IDs...")
        self.registry.register_batch(model_instances)

        # Create fresh graph for this conversion
        graph = nx.DiGraph()
        visited_ids: Set[str] = set()

        # First pass: create nodes
        for model in model_instances:
            self._create_nodes_pass(model, graph, visited_ids)

        # Second pass: create edges
        edges_to_add: List[Edge] = []
        for model in model_instances:
            edges = self._create_edges_pass(model, visited_ids)
            edges_to_add.extend(edges)

        # Add edges to graph
        edge_list = [(e.source, e.target, {"label": e.label, **e.properties}) for e in edges_to_add]

        if self.add_reverse_edges:
            reverse_edge_list = [
                (
                    e.target,
                    e.source,
                    {"label": f"reverse_{e.label}", **e.properties},
                )
                for e in edges_to_add
            ]
            edge_list.extend(reverse_edge_list)

        graph.add_edges_from(edge_list)

        # Bind provenance before cleanup: the cleaner's duplicate merge unions
        # node provenance, which requires the attribute to already be present.
        if provenance_binder is not None:
            try:
                provenance_binder(graph, model_instances)
            except Exception as e:
                # Grounding must never break graph conversion.
                logger.warning("Provenance binding failed: %s", e)

        id_fields_map = id_fields_by_class(model_instances)

        # Format-v2 self-describing export: embed the identity contract so a
        # later `docling-graph merge` can re-key nodes and propose aliases from
        # the export alone, without the original template. The schema hash uses
        # the same derivation as DocumentOrigin.template_schema_hash so the two
        # values always agree.
        root_template = type(model_instances[0])
        try:
            schema_hash = template_schema_hash(root_template)
        except Exception:
            schema_hash = ""
        graph.graph["format"] = "docling-graph/v2"
        graph.graph["template_name"] = root_template.__name__
        graph.graph["template_schema_hash"] = schema_hash
        graph.graph["id_fields_map"] = id_fields_map

        # Auto-cleanup if enabled
        if self.auto_cleanup and self.cleaner:
            logger.info("Running automatic graph cleanup...")
            graph = self.cleaner.clean_graph(graph)
            # Alias reconciliation (default-on with cleanup): merge same-class
            # nodes whose identifiers are containment aliases (table label vs
            # section title), subject to LLM confirmation via alias_llm_fn.
            alias_stats = reconcile_graph_aliases(
                graph,
                id_fields_map,
                llm_call_fn=self.alias_llm_fn,
            )
            if alias_stats.get("candidates"):
                graph.graph["alias_reconciliation"] = alias_stats

        # Closed-catalog reference enforcement (template-declared): drop
        # reference edges whose target exists ONLY through closed-catalog
        # reference fields — hallucinated members of a fixed catalog.
        self._enforce_closed_catalogs(graph)

        # Template-declared per-class cardinality bounds: demote surplus
        # instances of spam-prone classes, keeping the best-supported ones.
        # Runs after alias reconciliation so the filled/provenance ranking
        # signals reflect the final merged state.
        if self.enforce_cardinality_bounds:
            bounds = _collect_cardinality_bounds(model_instances)
            if bounds:
                root_classes = {m.__class__.__name__ for m in model_instances}
                self._enforce_cardinality_bounds(graph, bounds, root_classes, id_fields_map)

        # Integrity: a node whose class declares identity fields but carries no
        # identity value can never be matched, deduplicated, or evaluated —
        # surface it loudly instead of letting it reach the export silently.
        empty_identity = _empty_identity_node_ids(graph, id_fields_map)
        if empty_identity:
            graph.graph["empty_identity_nodes"] = empty_identity
            logger.warning(
                "Integrity: %s exported node(s) have empty identity fields: %s",
                len(empty_identity),
                ", ".join(empty_identity[:5]) + ("..." if len(empty_identity) > 5 else ""),
            )

        # Validate
        if self.validate_graph:
            try:
                validate_graph_structure(graph, raise_on_error=True)
                logger.info("Graph structure validated successfully")
            except ValueError as e:
                logger.error("Validation failed: %s", e)
                raise

        # Calculate statistics
        registry_stats = self.registry.get_stats()
        logger.info(
            "Final graph: %s nodes, %s edges (registry: %s entities across %s classes)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            registry_stats["total_entities"],
            len(registry_stats["classes"]),
        )

        metadata = calculate_graph_stats(graph, len(model_instances))
        return graph, metadata

    def _enforce_cardinality_bounds(
        self,
        graph: nx.DiGraph,
        bounds: dict[str, int],
        root_classes: Set[str],
        id_fields_map: dict[str, list[str]],
    ) -> None:
        """Demote instances of a bounded class past its ``graph_max_instances``.

        Instances are ranked best-first by filled-attribute count, then
        distinct-chunk provenance support, then in-degree from non-root nodes,
        then canonical identity as the stable tiebreak. Filled-first is
        deliberate and load-bearing: provenance-chunk-first buries true
        instances under alias-merged junk ("Total"/"Other" rows accumulate
        hundreds of chunks while a true segment can carry one). Demoted nodes
        are removed with their incident edges and recorded under
        ``graph.graph["demoted_nodes"]`` for audit.
        """
        demoted: list[dict[str, Any]] = []
        for cls in sorted(bounds):
            bound = bounds[cls]
            scored: list[tuple[Any, ...]] = []
            for node_id, data in graph.nodes(data=True):
                if str(data.get("__class__") or "") != cls:
                    continue
                ext_in = sum(
                    1
                    for source, _ in graph.in_edges(node_id)
                    if str(graph.nodes[source].get("__class__") or "") not in root_classes
                )
                identity = "".join(
                    canonicalize_identity_for_dedup(field, data.get(field))
                    for field in id_fields_map.get(cls, [])
                    if data.get(field) is not None
                )
                scored.append(
                    (
                        node_id,
                        data,
                        _attr_richness(data),
                        _provenance_weight(data),
                        ext_in,
                        identity,
                    )
                )
            if len(scored) <= bound:
                continue
            scored.sort(key=lambda s: (-s[2], -s[3], -s[4], s[5], str(s[0])))
            for node_id, data, filled, chunks, ext_in, _identity in scored[bound:]:
                demoted.append(
                    {
                        "id": str(node_id),
                        "class": cls,
                        "identity": {f: data.get(f) for f in id_fields_map.get(cls, [])},
                        "filled": filled,
                        "chunks": chunks,
                        "ext_in": ext_in,
                        "reason": "cardinality_bound",
                    }
                )
                graph.remove_node(node_id)
            logger.warning(
                "Cardinality bound: demoted %s of %s %s node(s) past graph_max_instances=%s",
                len(scored) - bound,
                len(scored),
                cls,
                bound,
            )
        if demoted:
            graph.graph["demoted_nodes"] = demoted

    def _enforce_closed_catalogs(self, graph: nx.DiGraph) -> None:
        """Drop reference edges to targets instantiated ONLY by closed-catalog
        reference fields (every in-edge carries the marker), removing targets
        that end up fully disconnected. A target that also exists through any
        other edge keeps everything — the catalog member is real, the marked
        edge just references it. Guard: enforcement requires at least one
        independently anchored member of the target class in the graph; when
        EVERY member is closed-catalog-only, the canonical catalog was not
        extracted at all and dropping would wipe the class — skip and warn
        instead. The transient edge marker is stripped before export either way.
        """

        def _marked(edge_data: dict[str, Any]) -> bool:
            # Label-scoped: a stale marker left by nx.DiGraph attr-merging of a
            # re-added (source, target) pair no longer matches the surviving
            # label and must not count.
            marker = edge_data.get("_closed_catalog")
            return bool(marker) and marker == edge_data.get("label")

        total_by_class: dict[str, int] = {}
        candidates_by_class: dict[str, list[str]] = {}
        any_marker = False
        for node_id, data in graph.nodes(data=True):
            cls = str(data.get("__class__") or "")
            total_by_class[cls] = total_by_class.get(cls, 0) + 1
            in_edges = list(graph.in_edges(node_id, data=True))
            if not in_edges:
                continue
            if any(_marked(d) for _, _, d in in_edges):
                any_marker = True
                if all(_marked(d) for _, _, d in in_edges):
                    candidates_by_class.setdefault(cls, []).append(node_id)
        if any_marker:
            drops: dict[str, int] = {}
            removed_nodes = 0
            for cls, candidates in candidates_by_class.items():
                if len(candidates) >= total_by_class.get(cls, 0):
                    logger.warning(
                        "Closed-catalog guard skipped for %s: all %s node(s) are "
                        "closed-catalog-only — the canonical catalog was not "
                        "extracted, refusing to wipe the class",
                        cls,
                        total_by_class.get(cls, 0),
                    )
                    continue
                for node_id in candidates:
                    for source, _, data in list(graph.in_edges(node_id, data=True)):
                        label = str(data.get("label") or "")
                        drops[label] = drops.get(label, 0) + 1
                        graph.remove_edge(source, node_id)
                    if graph.degree(node_id) == 0:
                        graph.remove_node(node_id)
                        removed_nodes += 1
            if drops:
                graph.graph["closed_catalog_drops"] = drops
                logger.warning(
                    "Closed catalog: dropped %s reference edge(s) to unanchored targets "
                    "(%s node(s) removed): %s",
                    sum(drops.values()),
                    removed_nodes,
                    drops,
                )
        for _, _, data in graph.edges(data=True):
            data.pop("_closed_catalog", None)

    def _create_nodes_pass(
        self,
        model: BaseModel,
        graph: nx.DiGraph,
        visited_ids: Set[str],
        processed_instances: Set[int] | None = None,
    ) -> None:
        """
        Recursively create nodes from model and nested entities.

        Entities (is_entity=True): Create separate nodes with edges
        Components (is_entity=False): Embed as dictionaries in parent nodes,
        but their subtrees are still traversed — an entity nested inside a
        component becomes its own node (mirroring the dense catalog, which
        walks through components and parents nested entities to the nearest
        entity ancestor).

        Duplicate instances of the same entity (same node ID reached via
        different paths) enrich the existing node: missing attributes are
        filled from the new instance (first non-empty value wins), and the
        duplicate's children are still traversed so children present only on
        the later instance are not lost.
        """
        # Guard recursion by object identity so re-encountering the same
        # instance (shared object, cyclic reference) terminates. Distinct
        # duplicate instances have distinct ids and are each processed.
        if processed_instances is None:
            processed_instances = set()
        if id(model) in processed_instances:
            return
        processed_instances.add(id(model))

        # Check if this model should be an entity (respect is_entity=False)
        is_entity = get_model_config_value(model, "is_entity", True)

        if not is_entity:
            # No node for the component itself (it embeds in its parent), but
            # entities nested below it still need nodes.
            self._walk_nested_models(model, graph, visited_ids, processed_instances)
            return

        # Get node ID from registry
        node_id = self._get_node_id(model)

        if node_id in visited_ids:
            # Same entity reached again via another path: fill attributes the
            # first instance left empty, then keep walking the children.
            self._enrich_existing_node(model, graph, node_id)
            self._walk_nested_models(model, graph, visited_ids, processed_instances)
            return

        visited_ids.add(node_id)

        # Prepare node attributes
        node_attrs: dict[str, Any] = {
            "id": node_id,
            "label": model.__class__.__name__,
            "type": "entity",
            "__class__": model.__class__.__name__,
        }

        # Add all fields from model
        for field_name, field_value in model:
            if isinstance(field_value, BaseModel):
                # Check if nested model is an entity or component
                is_nested_entity = get_model_config_value(field_value, "is_entity", True)

                if is_nested_entity:
                    # Entity: set to None (will be linked via edge)
                    node_attrs[field_name] = None
                    self._create_nodes_pass(field_value, graph, visited_ids, processed_instances)
                else:
                    # Component: embed as dictionary (entity fields inside it
                    # nulled — they become nodes + edges), then traverse its
                    # subtree for nested entities.
                    node_attrs[field_name] = self._component_to_attrs(field_value)
                    self._create_nodes_pass(field_value, graph, visited_ids, processed_instances)

            elif isinstance(field_value, list):
                # Handle empty lists and lists with content
                if field_value and isinstance(field_value[0], BaseModel):
                    # Non-empty list of BaseModel instances
                    # Check if list contains entities or components
                    is_list_entity = get_model_config_value(field_value[0], "is_entity", True)

                    if is_list_entity:
                        # List of entities: set to None (will be linked via edges)
                        node_attrs[field_name] = None
                        for item in field_value:
                            self._create_nodes_pass(item, graph, visited_ids, processed_instances)
                    else:
                        # List of components: embed as list of dictionaries and
                        # traverse each subtree for nested entities.
                        node_attrs[field_name] = [
                            self._component_to_attrs(item)
                            for item in field_value
                            if isinstance(item, BaseModel)
                        ]
                        for item in field_value:
                            if isinstance(item, BaseModel):
                                self._create_nodes_pass(
                                    item, graph, visited_ids, processed_instances
                                )
                else:
                    # Empty list or list of primitives - preserve as-is
                    node_attrs[field_name] = field_value
            else:
                node_attrs[field_name] = field_value

        graph.add_node(node_id, **node_attrs)

    def _walk_nested_models(
        self,
        model: BaseModel,
        graph: nx.DiGraph,
        visited_ids: Set[str],
        processed_instances: Set[int],
    ) -> None:
        """Recurse into every nested BaseModel field so entities below this
        model (whether it is a component or a duplicate entity instance) still
        get their nodes created."""
        for _field_name, field_value in model:
            if isinstance(field_value, BaseModel):
                self._create_nodes_pass(field_value, graph, visited_ids, processed_instances)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        self._create_nodes_pass(item, graph, visited_ids, processed_instances)

    def _component_to_attrs(self, component: BaseModel) -> dict[str, Any]:
        """
        Serialize a component for embedding in its parent entity's node.

        Like model_dump(), except entity-typed values are nulled: those become
        separate nodes linked by an edge from the nearest entity ancestor, so
        embedding their data too would duplicate it.
        """
        out: dict[str, Any] = {}
        for field_name, field_value in component:
            if isinstance(field_value, BaseModel):
                if get_model_config_value(field_value, "is_entity", True):
                    out[field_name] = None
                else:
                    out[field_name] = self._component_to_attrs(field_value)
            elif isinstance(field_value, list) and any(
                isinstance(item, BaseModel) for item in field_value
            ):
                first_model = next(item for item in field_value if isinstance(item, BaseModel))
                if get_model_config_value(first_model, "is_entity", True):
                    out[field_name] = None
                else:
                    out[field_name] = [
                        self._component_to_attrs(item)
                        for item in field_value
                        if isinstance(item, BaseModel)
                    ]
            else:
                out[field_name] = field_value
        return out

    def _enrich_existing_node(self, model: BaseModel, graph: nx.DiGraph, node_id: str) -> None:
        """
        Fill missing attributes on an already-created node from a duplicate
        instance of the same entity. First non-empty value wins — existing
        non-empty attributes are never overwritten, so conflicting duplicates
        keep the first-seen value while empty slots recover data that would
        otherwise be silently dropped.
        """
        if node_id not in graph:
            return
        existing = graph.nodes[node_id]
        for field_name, field_value in model:
            if isinstance(field_value, BaseModel):
                if get_model_config_value(field_value, "is_entity", True):
                    continue  # entity fields stay None (linked via edges)
                new_value: Any = self._component_to_attrs(field_value)
            elif isinstance(field_value, list) and field_value:
                if isinstance(field_value[0], BaseModel):
                    if get_model_config_value(field_value[0], "is_entity", True):
                        continue
                    new_value = [
                        self._component_to_attrs(item)
                        for item in field_value
                        if isinstance(item, BaseModel)
                    ]
                else:
                    new_value = field_value
            else:
                new_value = field_value

            if _is_empty_value(new_value):
                continue
            if _is_empty_value(existing.get(field_name)):
                existing[field_name] = new_value

    def _create_edges_pass(
        self,
        model: BaseModel,
        visited_ids: Set[str],
    ) -> List[Edge]:
        """
        Recursively create edges from model relationships.

        Only entities are edge sources (components don't have node IDs), but
        component subtrees are traversed: an entity nested inside a component
        gets an edge from the nearest entity ancestor, matching how the dense
        catalog parents such entities.
        """
        edges: List[Edge] = []

        # Check if this model is an entity (components don't have node IDs)
        is_entity = get_model_config_value(model, "is_entity", True)
        if not is_entity:
            # A component with no entity ancestor (top-level component) has no
            # edge source; nested components are handled via the ancestor's
            # _edges_through_component walk.
            return edges

        source_id = self._get_node_id(model)

        # Process all fields
        for field_name, field_value in model:
            # Check for explicit edge label in field metadata
            edge_label = self._get_edge_label(model, field_name)
            edge_props = self._edge_properties(model, field_name, edge_label)

            if isinstance(field_value, BaseModel):
                is_nested_entity = get_model_config_value(field_value, "is_entity", True)

                if is_nested_entity:
                    target_id = self._get_node_id(field_value)
                    edges.append(
                        Edge(
                            source=source_id,
                            target=target_id,
                            label=edge_label or field_name,
                            properties=edge_props,
                        )
                    )
                    # Recursively process nested entity
                    edges.extend(self._create_edges_pass(field_value, visited_ids))
                else:
                    # Component: no edge to it, but entities below it link
                    # from this entity (the nearest entity ancestor).
                    edges.extend(self._edges_through_component(source_id, field_value, visited_ids))

            elif isinstance(field_value, list) and field_value:
                if isinstance(field_value[0], BaseModel):
                    # Only create edges for lists of entities
                    is_list_entity = get_model_config_value(field_value[0], "is_entity", True)

                    if is_list_entity:
                        for item in field_value:
                            target_id = self._get_node_id(item)
                            edges.append(
                                Edge(
                                    source=source_id,
                                    target=target_id,
                                    label=edge_label or field_name,
                                    properties=edge_props,
                                )
                            )
                            # Recursively process nested entity
                            edges.extend(self._create_edges_pass(item, visited_ids))
                    else:
                        for item in field_value:
                            if isinstance(item, BaseModel):
                                edges.extend(
                                    self._edges_through_component(source_id, item, visited_ids)
                                )

        return edges

    def _edges_through_component(
        self,
        source_id: str,
        component: BaseModel,
        visited_ids: Set[str],
    ) -> List[Edge]:
        """
        Walk a component subtree creating edges from the nearest entity
        ancestor (source_id) to any entities nested below the component.
        Nested components are walked through with the same source.
        """
        edges: List[Edge] = []
        for field_name, field_value in component:
            edge_label = self._get_edge_label(component, field_name)
            edge_props = self._edge_properties(component, field_name, edge_label)

            if isinstance(field_value, BaseModel):
                if get_model_config_value(field_value, "is_entity", True):
                    target_id = self._get_node_id(field_value)
                    edges.append(
                        Edge(
                            source=source_id,
                            target=target_id,
                            label=edge_label or field_name,
                            properties=edge_props,
                        )
                    )
                    edges.extend(self._create_edges_pass(field_value, visited_ids))
                else:
                    edges.extend(self._edges_through_component(source_id, field_value, visited_ids))

            elif isinstance(field_value, list) and field_value:
                for item in field_value:
                    if not isinstance(item, BaseModel):
                        continue
                    if get_model_config_value(item, "is_entity", True):
                        target_id = self._get_node_id(item)
                        edges.append(
                            Edge(
                                source=source_id,
                                target=target_id,
                                label=edge_label or field_name,
                                properties=edge_props,
                            )
                        )
                        edges.extend(self._create_edges_pass(item, visited_ids))
                    else:
                        edges.extend(self._edges_through_component(source_id, item, visited_ids))

        return edges

    def _get_node_id(self, model: BaseModel) -> str:
        """Get deterministic node ID from registry."""
        return self.registry.get_node_id(model)

    def _get_edge_label(self, model: BaseModel, field_name: str) -> str | None:
        """
        Extract edge label from field metadata if available.

        Looks for json_schema_extra['edge_label'] in field info.
        """
        field_info = type(model).model_fields.get(field_name)
        if field_info and isinstance(field_info.json_schema_extra, Mapping):
            value = field_info.json_schema_extra.get("edge_label")
            if isinstance(value, str):
                return value
        return None

    def _edge_properties(
        self, model: BaseModel, field_name: str, edge_label: str | None
    ) -> dict[str, Any]:
        """Edge properties derived from field metadata.

        ``reference_closed_catalog`` fields stamp a transient ``_closed_catalog``
        marker holding the edge's label, consumed (and stripped) by
        ``_enforce_closed_catalogs``. The marker is label-scoped because
        nx.DiGraph MERGES attribute dicts when the same (source, target) pair is
        added under a second label — a boolean marker would contaminate the
        surviving edge; a label-scoped one only counts when it still matches.
        """
        field_info = type(model).model_fields.get(field_name)
        if field_info and isinstance(field_info.json_schema_extra, Mapping):
            if field_info.json_schema_extra.get("reference_closed_catalog"):
                return {"_closed_catalog": edge_label or field_name}
        return {}

    def set_registry(self, registry: NodeIDRegistry) -> None:
        """Update the registry (for sharing across multiple conversions)."""
        self.registry = registry
        logger.info("Registry updated with %s entities", registry.get_stats()["total_entities"])
