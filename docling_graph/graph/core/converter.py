"""
Handles conversion of Pydantic models to NetworkX graph structure.

This module provides the GraphConverter class for converting Pydantic models
into directed graphs with nodes and edges, including features like stable node
IDs, edge metadata, and bidirectional edges.
"""

import networkx as nx
import hashlib
import json

from datetime import date, datetime, time, timedelta
from pydantic import BaseModel
from decimal import Decimal

from typing import List, Any, Set, Optional, Type
from .models import Edge, GraphMetadata
from .config import GraphConfig

from ..utils.graph_stats import calculate_graph_stats


class GraphConverter:
    """Converts Pydantic models to NetworkX graphs with enhanced features."""

    def __init__(
        self, 
        config: Optional[GraphConfig] = None,
        add_reverse_edges: bool = False
    ):
        """Initialize the graph converter.

        Args:
            config: Graph configuration. Uses defaults if None.
            add_reverse_edges: If True, creates bidirectional edges.
        """
        self.config = config or GraphConfig()
        self.graph = nx.DiGraph()
        self._visited_ids: Set[str] = set()
        self.add_reverse_edges = add_reverse_edges or self.config.add_reverse_edges

    def pydantic_list_to_graph(
        self, 
        model_instances: List[BaseModel]
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

        self.graph = nx.DiGraph()
        self._visited_ids = set()

        # First pass: create nodes and collect edges
        edges_to_add: List[Edge] = []

        for model in model_instances:
            self._create_nodes_pass(model)

        # Second pass: create edges
        for model in model_instances:
            edges = self._create_edges_pass(model)
            edges_to_add.extend(edges)

        # Add edges to graph
        for edge in edges_to_add:
            self._add_edge_to_graph(edge)

            if self.add_reverse_edges:
                reverse_edge = Edge(
                    source=edge.target,
                    target=edge.source,
                    label=f"reverse_{edge.label}",
                    properties=edge.properties
                )
                self._add_edge_to_graph(reverse_edge)

        # Calculate metadata
        metadata = calculate_graph_stats(self.graph, len(model_instances))

        return self.graph, metadata

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
        if isinstance(value, str) and len(value) > self.config.MAX_STRING_LENGTH:
            return value[:self.config.MAX_STRING_LENGTH] + self.config.TRUNCATE_SUFFIX

        # Default: convert to string for complex objects
        if not isinstance(value, (str, int, float, bool, type(None))):
            return str(value)

        return value

    def _get_node_id(self, model_instance: BaseModel) -> str:
        """Generate stable node ID for a model instance.

        Uses graph_id_fields from model config if available, otherwise
        generates hash from model content.

        Args:
            model_instance: Pydantic model instance.

        Returns:
            Unique node ID string.
        """
        model_config = model_instance.model_config

        # Check for graph_id_fields in model config
        if hasattr(model_config, 'get'):
            id_fields = model_config.get('graph_id_fields', [])
        else:
            id_fields = getattr(model_config, 'graph_id_fields', [])

        if id_fields:
            # Use specified fields for ID
            id_parts = [str(getattr(model_instance, field, '')) for field in id_fields]
            return '_'.join(id_parts)

        # Generate hash-based ID
        content = json.dumps(
            model_instance.model_dump(), 
            sort_keys=True, 
            default=str
        )
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return f"{model_instance.__class__.__name__}_{hash_value[:self.config.NODE_ID_HASH_LENGTH]}"

    def _create_nodes_pass(self, model_instance: BaseModel) -> None:
        """First pass: create nodes for model and nested models.

        Args:
            model_instance: Pydantic model instance to process.
        """
        node_id = self._get_node_id(model_instance)

        if node_id in self._visited_ids:
            return

        self._visited_ids.add(node_id)

        # Create node with serialized attributes
        node_data = {
            'label': model_instance.__class__.__name__,
            'type': 'entity',
        }

        for field_name, field_value in model_instance.model_dump().items():
            if not isinstance(field_value, (list, dict, BaseModel)):
                node_data[field_name] = self._serialize_value(field_value)

        self.graph.add_node(node_id, **node_data)

        # Recursively process nested models
        for field_name, field_value in model_instance:
            if isinstance(field_value, BaseModel):
                self._create_nodes_pass(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        self._create_nodes_pass(item)

    def _create_edges_pass(self, model_instance: BaseModel) -> List[Edge]:
        """Second pass: create edges for relationships.

        Args:
            model_instance: Pydantic model instance to process.

        Returns:
            List of Edge objects representing relationships.
        """
        edges: List[Edge] = []
        source_id = self._get_node_id(model_instance)

        for field_name, field_value in model_instance:
            # Handle single nested model
            if isinstance(field_value, BaseModel):
                target_id = self._get_node_id(field_value)
                edges.append(Edge(
                    source=source_id,
                    target=target_id,
                    label=field_name,
                    properties={}
                ))
                # Recursively get edges from nested model
                edges.extend(self._create_edges_pass(field_value))

            # Handle list of nested models
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, BaseModel):
                        target_id = self._get_node_id(item)
                        edges.append(Edge(
                            source=source_id,
                            target=target_id,
                            label=field_name,
                            properties={}
                        ))
                        edges.extend(self._create_edges_pass(item))

        return edges

    def _add_edge_to_graph(self, edge: Edge) -> None:
        """Add an edge to the graph with its properties.

        Args:
            edge: Edge object to add to graph.
        """
        edge_data = {
            'label': edge.label,
            **edge.properties
        }
        self.graph.add_edge(edge.source, edge.target, **edge_data)
