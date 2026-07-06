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

from ...logging_utils import get_component_logger
from ..utils.alias_reconciler import id_fields_by_class, reconcile_graph_aliases
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


def _is_empty_value(value: Any) -> bool:
    """True for values that carry no data (None or empty str/list/dict).

    Numeric zero and False are meaningful and count as non-empty.
    """
    if value is None:
        return True
    if isinstance(value, str | list | dict):
        return len(value) == 0
    return False


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
            raise ValueError("Cannot create graph from empty model list")

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

        # Auto-cleanup if enabled
        if self.auto_cleanup and self.cleaner:
            logger.info("Running automatic graph cleanup...")
            graph = self.cleaner.clean_graph(graph)
            # Alias reconciliation (default-on with cleanup): merge same-class
            # nodes whose identifiers are containment aliases (table label vs
            # section title), subject to LLM confirmation via alias_llm_fn.
            alias_stats = reconcile_graph_aliases(
                graph,
                id_fields_by_class(model_instances),
                llm_call_fn=self.alias_llm_fn,
            )
            if alias_stats.get("candidates"):
                graph.graph["alias_reconciliation"] = alias_stats

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

            if isinstance(field_value, BaseModel):
                is_nested_entity = get_model_config_value(field_value, "is_entity", True)

                if is_nested_entity:
                    target_id = self._get_node_id(field_value)
                    edges.append(
                        Edge(
                            source=source_id,
                            target=target_id,
                            label=edge_label or field_name,
                            properties={},
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
                                    properties={},
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

            if isinstance(field_value, BaseModel):
                if get_model_config_value(field_value, "is_entity", True):
                    target_id = self._get_node_id(field_value)
                    edges.append(
                        Edge(
                            source=source_id,
                            target=target_id,
                            label=edge_label or field_name,
                            properties={},
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
                                properties={},
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

    def set_registry(self, registry: NodeIDRegistry) -> None:
        """Update the registry (for sharing across multiple conversions)."""
        self.registry = registry
        logger.info("Registry updated with %s entities", registry.get_stats()["total_entities"])
