"""
Handles conversion of Pydantic models to NetworkX graph structure.

This module provides the GraphConverter class for converting Pydantic models
into directed graphs with nodes and edges, including features like stable node
IDs, edge metadata, bidirectional edges, and automatic cleanup.
"""

from typing import List, Optional, Set

import networkx as nx
from pydantic import BaseModel
from rich import print as rich_print

from ..utils.graph_cleaner import GraphCleaner, validate_graph_structure
from ..utils.stats_calculator import calculate_graph_stats
from .config import GraphConfig
from .id_registry import NodeIDRegistry
from .models import Edge, GraphMetadata


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
        config: Optional[GraphConfig] = None,
        add_reverse_edges: bool = False,
        validate_graph: bool = True,
        registry: Optional[NodeIDRegistry] = None,
        auto_cleanup: bool = True,
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
        """
        self.config = config or GraphConfig()
        self.add_reverse_edges = add_reverse_edges or self.config.add_reverse_edges
        self.validate_graph = validate_graph or self.config.validate_graph

        # Initialize registry (use provided or create new)
        self.registry = registry or NodeIDRegistry()

        # Initialize cleaner for automatic cleanup
        self.auto_cleanup = auto_cleanup
        self.cleaner = GraphCleaner(verbose=True) if auto_cleanup else None

    def pydantic_list_to_graph(
        self,
        model_instances: List[BaseModel],
    ) -> tuple[nx.DiGraph, GraphMetadata]:
        """
        Convert list of Pydantic models to a NetworkX graph.

        Process:
        1. Pre-register all models for deterministic node IDs
        2. Create nodes from models
        3. Create edges between entities
        4. Apply automatic cleanup (if enabled)
        5. Validate graph structure
        6. Calculate statistics

        Args:
            model_instances: List of Pydantic model instances to convert

        Returns:
            Tuple of (graph, metadata)
        """
        if not model_instances:
            raise ValueError("Cannot create graph from empty model list")

        # Pre-register all models to ensure consistent node IDs across batches
        rich_print(
            "[blue][GraphConverter][/blue] Pre-registering models for deterministic node IDs..."
        )
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

        # Auto-cleanup if enabled
        if self.auto_cleanup and self.cleaner:
            rich_print("[blue][GraphConverter][/blue] Running automatic graph cleanup...")
            graph = self.cleaner.clean_graph(graph)

        # Validate
        if self.validate_graph:
            try:
                validate_graph_structure(graph, raise_on_error=True)
                rich_print("[green][GraphConverter][/green] Graph structure validated successfully")
            except ValueError as e:
                rich_print(f"[red][GraphConverter][/red] Validation failed: {e}")
                raise

        # Calculate statistics
        registry_stats = self.registry.get_stats()
        rich_print(
            f"[blue][GraphConverter][/blue] Final graph: "
            f"[cyan]{graph.number_of_nodes()}[/cyan] nodes, "
            f"[yellow]{graph.number_of_edges()}[/yellow] edges\n"
            f"  Registry: {registry_stats['total_entities']} entities across "
            f"{len(registry_stats['classes'])} classes"
        )

        metadata = calculate_graph_stats(graph, len(model_instances))
        return graph, metadata

    def _create_nodes_pass(
        self,
        model: BaseModel,
        graph: nx.DiGraph,
        visited_ids: Set[str],
    ) -> None:
        """Recursively create nodes from model and nested entities."""
        # Get node ID from registry
        node_id = self._get_node_id(model)

        if node_id in visited_ids:
            return

        visited_ids.add(node_id)

        # Prepare node attributes
        node_attrs = {
            "id": node_id,
            "label": model.__class__.__name__,
            "type": "entity",
            "__class__": model.__class__.__name__,
        }

        # Add all fields from model
        for field_name, field_value in model:
            if isinstance(field_value, BaseModel):
                node_attrs[field_name] = None  # Reference, not value
                self._create_nodes_pass(field_value, graph, visited_ids)
            elif isinstance(field_value, list) and field_value:
                if isinstance(field_value[0], BaseModel):
                    # List of nested entities
                    node_attrs[field_name] = None
                    for item in field_value:
                        self._create_nodes_pass(item, graph, visited_ids)
                else:
                    # List of primitives
                    node_attrs[field_name] = field_value
            else:
                node_attrs[field_name] = field_value

        graph.add_node(node_id, **node_attrs)

    def _create_edges_pass(
        self,
        model: BaseModel,
        visited_ids: Set[str],
    ) -> List[Edge]:
        """Recursively create edges from model relationships."""
        edges: List[Edge] = []
        source_id = self._get_node_id(model)

        # Process all fields
        for field_name, field_value in model:
            # Check for explicit edge label in field metadata
            edge_label = self._get_edge_label(model, field_name)

            if isinstance(field_value, BaseModel):
                target_id = self._get_node_id(field_value)
                edges.append(
                    Edge(
                        source=source_id,
                        target=target_id,
                        label=edge_label or field_name,
                        properties={},
                    )
                )

                # Recursively process nested model
                edges.extend(self._create_edges_pass(field_value, visited_ids))

            elif isinstance(field_value, list) and field_value:
                if isinstance(field_value[0], BaseModel):
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

                        # Recursively process nested model
                        edges.extend(self._create_edges_pass(item, visited_ids))

        return edges

    def _get_node_id(self, model: BaseModel) -> str:
        """Get deterministic node ID from registry."""
        return self.registry.get_node_id(model)

    def _get_edge_label(self, model: BaseModel, field_name: str) -> Optional[str]:
        """
        Extract edge label from field metadata if available.

        Looks for json_schema_extra['edge_label'] in field info.
        """
        field_info = model.model_fields.get(field_name)
        if field_info and field_info.json_schema_extra:
            return field_info.json_schema_extra.get("edge_label")
        return None

    def set_registry(self, registry: NodeIDRegistry) -> None:
        """Update the registry (for sharing across multiple conversions)."""
        self.registry = registry
        rich_print(
            f"[blue][GraphConverter][/blue] Registry updated with "
            f"{registry.get_stats()['total_entities']} entities"
        )
