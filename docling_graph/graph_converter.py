"""
Handles the conversion of Pydantic models to a NetworkX graph structure.

This module defines the GraphConverter class, which provides methods to
convert Pydantic models into a directed graph with nodes and edges,
including enhanced features like stable node IDs, edge metadata,
and bidirectional edges.
"""

import networkx as nx
import hashlib
import json

from datetime import date, datetime, time, timedelta
from pydantic import BaseModel
from decimal import Decimal

from typing import List, Any, Set, Optional
from .graph_models import Edge

class GraphConverter:
    """
    Converts Pydantic models to NetworkX graphs with enhanced features.
    """

    def __init__(self, add_reverse_edges: bool = False):
        """
        Initialize the graph converter.
        
        Args:
            add_reverse_edges: If True, creates bidirectional edges for easier querying
        """
        self.graph = nx.DiGraph()
        self._visited_ids: Set[str] = set()
        self.add_reverse_edges = add_reverse_edges

    def _serialize_value(self, value: Any) -> Any:
        """
        Safely serialize a value for node attributes.
        """
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        elif isinstance(value, time):
            return value.isoformat()
        elif isinstance(value, timedelta):
            return value.total_seconds()
        elif isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, dict)):
            return value
        else:
            return str(value)

    def _get_hashable_attributes(self, model_instance: BaseModel) -> dict:
        """
        Extract hashable attributes from a model, excluding edge fields.
        """
        attr_dict = {}
        for field_name in model_instance.model_fields:
            value = getattr(model_instance, field_name)
            
            if self._is_field_an_edge(value):
                continue
            
            try:
                attr_dict[field_name] = json.dumps(
                    self._serialize_value(value), 
                    default=str, 
                    sort_keys=True
                )
            except (TypeError, ValueError):
                attr_dict[field_name] = str(value)
        
        return attr_dict

    def _get_node_id(self, model_instance: BaseModel) -> str:
        """
        Generate a unique, stable node ID for a Pydantic model instance.
        """
        node_label = model_instance.__class__.__name__
        config = getattr(model_instance, 'model_config', {})
        id_fields = config.get('graph_id_fields', [])

        if id_fields:
            id_parts = []
            for field in id_fields:
                try:
                    value = getattr(model_instance, field)
                    id_parts.append(str(value) if value is not None else '')
                except AttributeError:
                    raise ValueError(
                        f"{node_label} defines graph_id_fields={id_fields} "
                        f"but field '{field}' does not exist on the model."
                    )
            
            if not any(id_parts):
                raise ValueError(
                    f"{node_label} has graph_id_fields={id_fields} but all values are empty. "
                    f"Cannot generate stable node ID. Instance: {model_instance}"
                )
            
            id_string = ":".join(id_parts)
        else:
            attr_dict = self._get_hashable_attributes(model_instance)
            id_string = json.dumps(attr_dict, sort_keys=True)

        hash_id = hashlib.md5(id_string.encode()).hexdigest()
        return f"{node_label}_{hash_id[:12]}"

    def _is_field_an_edge(self, value: Any) -> bool:
        """Check if a field value represents an edge."""
        if isinstance(value, BaseModel) and not isinstance(value, Edge):
            return True
        if isinstance(value, Edge):
            return True
        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, (BaseModel, Edge)):
                return True
        return False

    def _create_nodes_pass(self, model_instance: BaseModel):
        """First pass: Create all nodes in the graph."""
        node_id = self._get_node_id(model_instance)
        
        if node_id in self._visited_ids:
            return
        self._visited_ids.add(node_id)

        node_attrs = {"label": model_instance.__class__.__name__}
        
        for field_name in model_instance.model_fields:
            value = getattr(model_instance, field_name)
            
            if self._is_field_an_edge(value):
                continue
            
            node_attrs[field_name] = self._serialize_value(value)

        self.graph.add_node(node_id, **node_attrs)

        # Recursively process nested models
        for field_name in model_instance.model_fields:
            value = getattr(model_instance, field_name)
            
            if isinstance(value, Edge):
                self._create_nodes_pass(value.target)
            elif isinstance(value, BaseModel) and not isinstance(value, Edge):
                self._create_nodes_pass(value)
            elif isinstance(value, list) and value:
                first_item = value[0]
                if isinstance(first_item, Edge):
                    for edge_item in value:
                        self._create_nodes_pass(edge_item.target)
                elif isinstance(first_item, BaseModel):
                    for item in value:
                        self._create_nodes_pass(item)

    def _create_edges_pass(self, model_instance: BaseModel):
        """Second pass: Create all edges between nodes."""
        current_node_id = self._get_node_id(model_instance)

        for field_name in model_instance.model_fields:
            value = getattr(model_instance, field_name)

            # Case 1: Single Edge[T] instance
            if isinstance(value, Edge):
                target_node_id = self._get_node_id(value.target)
                if self.graph.has_node(target_node_id):
                    edge_attrs = {
                        "label": value.label,
                        "cardinality": "one-to-one"
                    }
                    if value.weight is not None:
                        edge_attrs["weight"] = value.weight
                    if value.context is not None:
                        edge_attrs["context"] = value.context
                    
                    self.graph.add_edge(current_node_id, target_node_id, **edge_attrs)
                    
                    if self.add_reverse_edges:
                        reverse_attrs = edge_attrs.copy()
                        reverse_attrs["label"] = f"inverse_{value.label}"
                        self.graph.add_edge(target_node_id, current_node_id, **reverse_attrs)
                
                self._create_edges_pass(value.target)

            # Case 2: Single BaseModel instance
            elif isinstance(value, BaseModel) and not isinstance(value, Edge):
                target_node_id = self._get_node_id(value)
                if self.graph.has_node(target_node_id):
                    edge_attrs = {
                        "label": field_name,
                        "cardinality": "one-to-one"
                    }
                    self.graph.add_edge(current_node_id, target_node_id, **edge_attrs)
                    
                    if self.add_reverse_edges:
                        reverse_attrs = edge_attrs.copy()
                        reverse_attrs["label"] = f"inverse_{field_name}"
                        self.graph.add_edge(target_node_id, current_node_id, **reverse_attrs)
                
                self._create_edges_pass(value)

            # Case 3: List of models
            elif isinstance(value, list):
                if not value:  # Handle empty lists
                    continue
                
                first_item = value[0]
                
                # List of Edge[T] instances
                if isinstance(first_item, Edge):
                    for idx, edge_item in enumerate(value):
                        target_node_id = self._get_node_id(edge_item.target)
                        if self.graph.has_node(target_node_id):
                            edge_attrs = {
                                "label": edge_item.label,
                                "index": idx,
                                "cardinality": "one-to-many"
                            }
                            if edge_item.weight is not None:
                                edge_attrs["weight"] = edge_item.weight
                            if edge_item.context is not None:
                                edge_attrs["context"] = edge_item.context
                            
                            self.graph.add_edge(current_node_id, target_node_id, **edge_attrs)
                            
                            if self.add_reverse_edges:
                                reverse_attrs = edge_attrs.copy()
                                reverse_attrs["label"] = f"inverse_{edge_item.label}"
                                self.graph.add_edge(target_node_id, current_node_id, **reverse_attrs)
                        
                        self._create_edges_pass(edge_item.target)
                
                # List of regular BaseModel instances
                elif isinstance(first_item, BaseModel):
                    for idx, item in enumerate(value):
                        target_node_id = self._get_node_id(item)
                        if self.graph.has_node(target_node_id):
                            edge_attrs = {
                                "label": field_name,
                                "index": idx,
                                "cardinality": "one-to-many"
                            }
                            self.graph.add_edge(current_node_id, target_node_id, **edge_attrs)
                            
                            if self.add_reverse_edges:
                                reverse_attrs = edge_attrs.copy()
                                reverse_attrs["label"] = f"inverse_{field_name}"
                                self.graph.add_edge(target_node_id, current_node_id, **reverse_attrs)
                        
                        self._create_edges_pass(item)

    def pydantic_list_to_graph(self, models: List[BaseModel]) -> nx.DiGraph:
        """
        Convert a list of Pydantic models to a single NetworkX graph.
        
        Returns:
            NetworkX DiGraph (never returns None)
        """
        if not models:
            raise ValueError("Cannot create graph from empty model list")
        
        self.graph.clear()
        self._visited_ids.clear()

        # Add graph metadata
        self.graph.graph['created_at'] = datetime.now().isoformat()
        self.graph.graph['model_types'] = list(set(m.__class__.__name__ for m in models))
        self.graph.graph['num_root_entities'] = len(models)
        self.graph.graph['has_reverse_edges'] = self.add_reverse_edges

        # Pass 1: Create nodes
        for root_model in models:
            try:
                self._create_nodes_pass(root_model)
            except Exception as e:
                raise ValueError(
                    f"Error creating nodes for {root_model.__class__.__name__}: {e}"
                ) from e

        # Pass 2: Create edges
        for root_model in models:
            try:
                self._create_edges_pass(root_model)
            except Exception as e:
                raise ValueError(
                    f"Error creating edges for {root_model.__class__.__name__}: {e}"
                ) from e

        # Add statistics
        self.graph.graph['num_nodes'] = self.graph.number_of_nodes()
        self.graph.graph['num_edges'] = self.graph.number_of_edges()
        
        if self.graph.number_of_nodes() > 0:
            degrees = [d for _, d in self.graph.degree()]
            self.graph.graph['avg_degree'] = sum(degrees) / len(degrees)
        else:
            self.graph.graph['avg_degree'] = 0

        return self.graph

    def to_json_serializable(self) -> dict:
        """Export the graph to JSON format."""
        graph_data = nx.node_link_data(self.graph)
        
        return {
            "metadata": {
                "created_at": self.graph.graph.get('created_at'),
                "model_types": self.graph.graph.get('model_types', []),
                "num_root_entities": self.graph.graph.get('num_root_entities', 0),
                "num_nodes": self.graph.graph.get('num_nodes', 0),
                "num_edges": self.graph.graph.get('num_edges', 0),
                "avg_degree": self.graph.graph.get('avg_degree', 0),
                "has_reverse_edges": self.graph.graph.get('has_reverse_edges', False)
            },
            "nodes": [
                {
                    "id": n["id"],
                    "type": n.get("label", "Unknown"),
                    "properties": {k: v for k, v in n.items() if k not in ["id", "label"]}
                }
                for n in graph_data["nodes"]
            ],
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "label": e.get("label", "related_to"),
                    "properties": {k: v for k, v in e.items() if k not in ["source", "target", "label"]}
                }
                for e in graph_data["links"]
            ]
        }
