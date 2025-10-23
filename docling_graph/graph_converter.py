"""
Handles the conversion of Pydantic models to a NetworkX graph structure.
"""
from pydantic import BaseModel
import networkx as nx
import hashlib
from typing import Any, Dict, List, Tuple

class GraphConverter:
    """
    Converts Pydantic models into a NetworkX graph, handling both direct and name-based relationships.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.name_to_id_map: Dict[Tuple[str, str], str] = {}
        self._visited_ids = set()

    def _get_node_id(self, model_instance: BaseModel) -> str:
        """Creates a deterministic, content-based ID for a node."""
        node_label = model_instance.__class__.__name__
        
        # Define key fields for creating a unique signature for each entity type
        id_fields = {
            "InsurancePlan": ["name"],
            "Guarantee": ["name"],
            "HomeInsurance": ["product_name"],
            "Person": ["name"],
            "Organization": ["name"],
            "Address": ["street", "city", "postal_code"],
            "Invoice": ["bill_no"],
        }.get(node_label, [])

        # Fallback: use all string/number fields if no specific key is defined
        if not id_fields:
            id_fields = [
                f for f, info in model_instance.model_fields.items()
                if info.annotation in (str, int, float)
            ]

        # Use the first available field as a "name" for the map, or hash all
        id_string = ""
        for field_name in id_fields:
            value = getattr(model_instance, field_name, None)
            if value:
                id_string += str(value)
        
        if not id_string:
            # If no key fields, hash the whole (serializable) model
            try:
                id_string = model_instance.model_dump_json()
            except Exception:
                id_string = str(model_instance) # Last resort

        hash_id = hashlib.md5(id_string.encode()).hexdigest()
        return f"{node_label}_{hash_id[:10]}"

    def _get_node_name_for_map(self, model_instance: BaseModel) -> str:
        """Gets the 'name' of a node for name-based edge mapping."""
        # This is the field other models will use to refer to this one
        if hasattr(model_instance, "name"):
            return getattr(model_instance, "name")
        if hasattr(model_instance, "bill_no"):
            return getattr(model_instance, "bill_no")
        # Fallback
        return str(model_instance)

    def _create_nodes_and_map(self, model_instance: BaseModel):
        """Pass 1: Recursively create nodes and populate the name_to_id_map."""
        node_id = self._get_node_id(model_instance)
        
        if node_id in self._visited_ids:
            return  # This exact entity has been processed
        self._visited_ids.add(node_id)

        node_label = model_instance.__class__.__name__
        
        # Add to name map
        node_name = self._get_node_name_for_map(model_instance)
        if node_name and (node_name, node_label) not in self.name_to_id_map:
            self.name_to_id_map[(node_name, node_label)] = node_id

        # Create node in graph
        node_attrs = {"id": node_id, "label": node_label}
        for field_name, field_info in model_instance.model_fields.items():
            value = getattr(model_instance, field_name)
            
            # Skip fields that define edges
            if field_info.json_schema_extra and 'edge_label' in field_info.json_schema_extra:
                continue

            if isinstance(value, (str, int, float, bool)):
                node_attrs[field_name] = value
            elif isinstance(value, list) and all(isinstance(i, (str, int, float, bool)) for i in value):
                node_attrs[field_name] = ", ".join(map(str, value)) # Simple list to string

        self.graph.add_node(node_id, **node_attrs)

        # Recurse
        for field_name, field_info in model_instance.model_fields.items():
            value = getattr(model_instance, field_name)
            if isinstance(value, BaseModel):
                self._create_nodes_and_map(value)
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                for item in value:
                    self._create_nodes_and_map(item)

    def _create_edges(self, model_instance: BaseModel):
        """Pass 2: Recursively create edges for an already-processed node."""
        current_node_id = self._get_node_id(model_instance)
        
        for field_name, field_info in model_instance.model_fields.items():
            if not (field_info.json_schema_extra and 'edge_label' in field_info.json_schema_extra):
                continue
                
            edge_label = field_info.json_schema_extra['edge_label']
            value = getattr(model_instance, field_name)
            
            items_to_link = []
            if isinstance(value, BaseModel):
                items_to_link.append(value)
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                items_to_link.extend(value)

            # Case 1: Linking to other Pydantic models (direct object link)
            for item in items_to_link:
                target_node_id = self._get_node_id(item)
                if self.graph.has_node(target_node_id):
                    self.graph.add_edge(current_node_id, target_node_id, label=edge_label)
            
            # Case 2: Linking by name (e.g., plans linking to guarantees)
            if isinstance(value, list) and value and isinstance(value[0], str):
                target_label = field_info.json_schema_extra.get('target_label')
                if target_label:
                    for item_name in value:
                        if (item_name, target_label) in self.name_to_id_map:
                            target_node_id = self.name_to_id_map[(item_name, target_label)]
                            self.graph.add_edge(current_node_id, target_node_id, label=edge_label)
        
        # Recurse
        for field_name in model_instance.model_fields:
            value = getattr(model_instance, field_name)
            if isinstance(value, BaseModel):
                self._create_edges(value)
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                for item in value:
                    self._create_edges(item)

    def pydantic_list_to_graph(self, models: List[BaseModel]) -> nx.DiGraph:
        """
        Converts a LIST of Pydantic models to a single NetworkX graph.
        This is the new main entry point.
        """
        # Clear previous state
        self.graph.clear()
        self.name_to_id_map.clear()
        self._visited_ids.clear()

        # Pass 1: Create all nodes and the name-to-ID map
        for root_model in models:
            self._create_nodes_and_map(root_model)
        
        # Pass 2: Create all edges using the populated graph and map
        for root_model in models:
            self._create_edges(root_model)
        
        return self.graph

    def to_json_serializable(self) -> dict:
        """Exports the graph to a backend-agnostic, JSON-friendly format."""
        graph_data = nx.node_link_data(self.graph)
        nodes = [{"id": n["id"], "type": n.get("label", ""), "properties": {k:v for k,v in n.items() if k not in ["id", "label"]}} for n in graph_data["nodes"]]
        edges = [{"id": f"e{i}", "source": l["source"], "target": l["target"], "label": l.get("label", "")} for i, l in enumerate(graph_data["links"])]
        return {"nodes": nodes, "edges": edges}
