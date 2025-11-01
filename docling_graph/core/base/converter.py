"""
Handles conversion of Pydantic models to NetworkX graph structure.

This module provides the GraphConverter class for converting Pydantic models
into directed graphs with nodes and edges, including features like stable node
IDs, edge metadata, and bidirectional edges.
"""

import hashlib
import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, List, Optional, Set, Type

import networkx as nx
from pydantic import BaseModel
from rich import print

from ..utils.graph_stats import calculate_graph_stats
from .config import GraphConfig
from .models import Edge, GraphMetadata


class GraphConverter:
    """Converts Pydantic models to NetworkX graphs with enhanced features.

    This converter is stateless and thread-safe. All conversion state is managed
    through method parameters rather than instance variables.
    """

    def __init__(
        self,
        config: Optional[GraphConfig] = None,
        add_reverse_edges: bool = False,
        validate_graph: bool = True,
    ):
        """Initialize the graph converter.

        Args:
            config: Graph configuration. Uses defaults if None.
            add_reverse_edges: If True, creates bidirectional edges.
        """
        self.config = config or GraphConfig()
        self.add_reverse_edges = add_reverse_edges or self.config.add_reverse_edges
        self.validate_graph = validate_graph or self.config.validate_graph

    def pydantic_list_to_graph(
        self, model_instances: List[BaseModel]
    ) -> tuple[nx.DiGraph, GraphMetadata]:
        """Convert list of Pydantic models to a NetworkX graph.

        Args:
            model_instances: List of Pydantic model instances to convert.

        Returns:
            Tuple of (NetworkX graph, graph metadata).

        Raises:
            ValueError: If model_instances is empty.
        """
        if not model_instances:
            raise ValueError("Cannot create graph from empty model list")

        # Create fresh graph and visited set for this conversion (stateless)
        graph = nx.DiGraph()
        visited_ids: Set[str] = set()

        # First pass: create nodes and collect edges
        edges_to_add: List[Edge] = []
        for model in model_instances:
            self._create_nodes_pass(model, graph, visited_ids)

        # Second pass: create edges
        for model in model_instances:
            edges = self._create_edges_pass(model, visited_ids)
            edges_to_add.extend(edges)

        # Add edges to graph in bulk
        edge_list = [(e.source, e.target, {"label": e.label, **e.properties}) for e in edges_to_add]

        if self.add_reverse_edges:
            reverse_edge_list = [
                (e.target, e.source, {"label": f"reverse_{e.label}", **e.properties})
                for e in edges_to_add
            ]
            edge_list.extend(reverse_edge_list)

        # Add all edges in one go
        graph.add_edges_from(edge_list)

        # Validate Graph Structure
        if self.validate_graph:
            self._validate_graph(graph)

        # Calculate metadata
        metadata = calculate_graph_stats(graph, len(model_instances))
        return graph, metadata

    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize a value for node attributes.

        Args:
            value: Value to serialize.

        Returns:
            Serialized value safe for NetworkX attributes.
        """
        # Handle date/time objects
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, time):
            return value.isoformat()
        if isinstance(value, timedelta):
            return str(value)

        # Handle numeric types
        if isinstance(value, Decimal):
            return float(value)

        # Handle Pydantic models
        if isinstance(value, BaseModel):
            return value.model_dump()

        # Handle collections
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Handle strings (truncate if too long)
        if isinstance(value, str):
            if len(value) > self.config.MAX_STRING_LENGTH:
                print(
                    f"[yellow][!] Truncating string from {len(value)} "
                    f"to {self.config.MAX_STRING_LENGTH} chars[/yellow]"
                )
                return value[: self.config.MAX_STRING_LENGTH] + self.config.TRUNCATE_SUFFIX
            return value

        # Default: convert complex objects to string safely
        if not isinstance(value, (int, float, bool, type(None))):
            try:
                return str(value)
            except Exception as e:
                print(
                    f"[red][GraphConverter] Serialization failed for object of type[/red]"
                    f"{type(value).__name__}: {e}"
                )
                return "<unserializable>"

        return value

    def _get_node_id(self, model_instance: BaseModel) -> str:
        """Generate stable node ID for a model instance.

        Uses graph_id_fields from model config if available, otherwise
        generates hash from model content. Always includes a hash suffix
        for consistency and to handle complex field types.

        Args:
            model_instance: Pydantic model instance.

        Returns:
            Unique node ID string in format: ClassName_hash
        """
        model_config = model_instance.model_config

        # Check for graph_id_fields in model config
        if hasattr(model_config, "get"):
            id_fields = model_config.get("graph_id_fields", [])
        else:
            id_fields = getattr(model_config, "graph_id_fields", [])

        # Determine what to hash
        if id_fields:
            # Use specified fields to create content for hashing
            id_content = {}
            for field in id_fields:
                if not hasattr(model_instance, field):
                    raise ValueError(f"graph_id_fields references non-existent field: {field}")
                value = getattr(model_instance, field)
                # Handle lists - convert to sorted tuple for consistent hashing
                if isinstance(value, list):
                    # Sort if items are comparable, otherwise use as-is
                    try:
                        value = tuple(sorted(value))
                    except TypeError:
                        value = tuple(value)
                id_content[field] = value

            # Create JSON string from ID fields only
            content_str = json.dumps(id_content, sort_keys=True, default=str)
        else:
            # Use entire model for hashing
            content_str = json.dumps(model_instance.model_dump(), sort_keys=True, default=str)

        # Generate hash from content
        hash_value = hashlib.blake2b(content_str.encode(), digest_size=32).hexdigest()

        # Return ID in consistent format: ClassName_hash
        return (
            f"{model_instance.__class__.__name__}_{hash_value[: self.config.NODE_ID_HASH_LENGTH]}"
        )

    def _create_nodes_pass(
        self, model_instance: BaseModel, graph: nx.DiGraph, visited_ids: Set[str]
    ) -> None:
        """First pass: create nodes for model and nested models.

        Args:
            model_instance: Pydantic model instance to process.
            graph: NetworkX graph to add nodes to.
            visited_ids: Set of already visited node IDs.
        """
        node_id = self._get_node_id(model_instance)

        if node_id in visited_ids:
            return

        visited_ids.add(node_id)

        # Create node with serialized attributes
        node_data = {
            "label": model_instance.__class__.__name__,
            "type": "entity",
        }

        for field_name, field_value in model_instance.model_dump().items():
            # Check if it's a list of BaseModel instances (edges/relationships)
            if isinstance(field_value, list):
                # Check if list contains BaseModel instances
                if field_value and all(isinstance(item, dict) for item in field_value):
                    # Skip - this is likely a nested model that will become an edge
                    continue
                else:
                    # It's a list of simple types (strings, numbers, etc.) - include it
                    node_data[field_name] = self._serialize_value(field_value)
            # Skip dicts and BaseModel instances (they become edges)
            elif isinstance(field_value, (dict, BaseModel)):
                continue
            else:
                # Simple field - add it
                node_data[field_name] = self._serialize_value(field_value)

        graph.add_node(node_id, **node_data)

        # Recursively process nested models
        for field_name, field_value in model_instance:
            if isinstance(field_value, BaseModel):
                self._create_nodes_pass(field_value, graph, visited_ids)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        self._create_nodes_pass(item, graph, visited_ids)

    def _create_edges_pass(self, model_instance: BaseModel, visited_ids: Set[str]) -> List[Edge]:
        """Second pass: create edges for relationships.

        Args:
            model_instance: Pydantic model instance to process.
            visited_ids: Set of visited node IDs (for validation).

        Returns:
            List of Edge objects representing relationships.
        """
        edges: List[Edge] = []
        source_id = self._get_node_id(model_instance)

        for field_name, field_value in model_instance:
            # Handle single nested model
            if isinstance(field_value, BaseModel):
                target_id = self._get_node_id(field_value)

                # Edge Node validation
                if target_id not in visited_ids:
                    raise ValueError(f"Target node {target_id} not found for edge from {source_id}")

                edges.append(
                    Edge(source=source_id, target=target_id, label=field_name, properties={})
                )
                # Recursively get edges from nested model
                edges.extend(self._create_edges_pass(field_value, visited_ids))

            # Handle list of nested models
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        target_id = self._get_node_id(item)
                        edges.append(
                            Edge(
                                source=source_id, target=target_id, label=field_name, properties={}
                            )
                        )
                        edges.extend(self._create_edges_pass(item, visited_ids))

        return edges

    def _add_edge_to_graph(self, edge: Edge, graph: nx.DiGraph) -> None:
        """Add an edge to the graph with its properties.

        Args:
            edge: Edge object to add to graph.
            graph: NetworkX graph to add edge to.
        """
        if graph.has_edge(edge.source, edge.target):
            # Merge properties
            existing = graph[edge.source][edge.target]
            existing.update(edge.properties)

        graph.add_edge(edge.source, edge.target, label=edge.label, **edge.properties)

    def _validate_graph(self, graph: nx.DiGraph) -> None:
        """
        Validate graph structure after creation.

        Ensures all edge endpoints reference existing nodes in the graph.
        Uses rich console for formatted error output.

        Args:
            graph: NetworkX directed graph to validate.

        Raises:
            ValueError: If graph contains edges pointing to non-existent nodes.
        """
        nodes = set(graph.nodes())
        invalid_edges = []

        # Check all edges have valid source and target nodes
        for source, target in graph.edges():
            if source not in nodes:
                invalid_edges.append((source, target, "source"))
            if target not in nodes:
                invalid_edges.append((source, target, "target"))

        # Report errors with rich formatting if found
        if invalid_edges:
            print("[red][GraphConverter] Graph validation failed![/red]")

            # Show first 5 invalid edges
            for source, target, issue_type in invalid_edges[:5]:
                print(
                    f"[red]  • Invalid edge: {source} -> {target} "
                    f"({issue_type} node not found)[/red]"
                )

            # Indicate if there are more
            if len(invalid_edges) > 5:
                print(f"[red]  • ... and {len(invalid_edges) - 5} more invalid edges[/red]")

            raise ValueError(f"Graph validation failed: {len(invalid_edges)} invalid edge(s) found")
