"""
Module for exporting networkx graphs to various formats.
"""

import networkx as nx
import pandas as pd
import re

from pathlib import Path
from typing import Any

def escape_cypher_string(value: Any) -> str:
    """Escapes strings for use in Cypher queries."""
    if not isinstance(value, str):
        value = str(value)
    # Escape quotes and backslashes
    return value.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")

def to_cypher(graph: nx.Graph, output_path: Path):
    """
    Exports a networkx graph to a Cypher script.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Create nodes
        f.write("// --- Create Nodes ---\n")
        f.write("CREATE\n")
        node_statements = []
        
        # Use a dictionary to ensure sanitized node variables are unique
        node_vars = {}
        i = 0

        for node, data in graph.nodes(data=True):
            # Use 'label' for node label, default to 'Node'
            labels = data.get("label", "Node")
            
            # Sanitize node ID for use as Cypher variable
            node_var_base = re.sub(r'[^a-zA-Z0-9_]', '_', str(node))
            if not node_var_base or node_var_base[0].isdigit():
                node_var_base = 'n' + node_var_base

            # Ensure uniqueness
            node_var = node_var_base
            while node_var in node_vars:
                node_var = f"{node_var_base}_{i}"
                i += 1
            node_vars[node] = node_var

            # Prepare properties
            properties = {"id": node} # Always include original ID as a property
            for k, v in data.items():
                if k != "label" and v is not None: # Ensure value is not None
                    properties[k] = v
            
            # Build property string
            prop_string = ", ".join(
                [f'{k}: "{escape_cypher_string(v)}"' for k, v in properties.items()]
            )
            
            node_statements.append(f"  ({node_vars[node]}:`{labels}` {{{prop_string}}})")
        
        f.write(",\n".join(node_statements) + ";\n")


        # Create relationships
        f.write("\n// --- Create Relationships ---\n")
        for source, target, data in graph.edges(data=True):
            rel_type = data.get("label", "RELATED_TO")
            
            # Prepare properties
            properties = {}
            for k, v in data.items():
                if k != "label" and v is not None: # Ensure value is not None
                    properties[k] = v
            
            # Build property string
            if properties:
                prop_string = " {" + ", ".join(
                    [f'{k}: "{escape_cypher_string(v)}"' for k, v in properties.items()]
                ) + "}"
            else:
                prop_string = ""

            f.write(f"MATCH (a {{id: \"{escape_cypher_string(source)}\"}}), (b {{id: \"{escape_cypher_string(target)}\"}}) CREATE (a)-[:`{rel_type}`{prop_string}]->(b);\n")

def to_csv(graph: nx.Graph, output_dir: Path):
    """
    Exports a networkx graph to nodes and relationships CSV files.
    """
    
    graph_dir = Path(output_dir, "graph")
    graph_dir.mkdir(parents=True, exist_ok=True)

    # --- Export Nodes ---
    nodes_data = []
    for node, data in graph.nodes(data=True):
        # Base node info
        node_info = {
            "id:ID": node, # :ID hints to Neo4j admin import
            "label:LABEL": data.get("label", "Node") # :LABEL hints to Neo4j admin import
        }
        
        # Add all other properties, skipping 'label'
        node_info.update({k: v for k, v in data.items() if k != "label"})
        nodes_data.append(node_info)
        
    nodes_df = pd.DataFrame(nodes_data)
    
    # Check if DataFrame is not empty before reordering
    if not nodes_df.empty:
        # Reorder columns to put id and label first
        cols = ["id:ID", "label:LABEL"] + [c for c in nodes_df.columns if c not in ["id:ID", "label:LABEL"]]
        nodes_df = nodes_df[cols]
    
    nodes_df.to_csv(graph_dir / "nodes.csv", index=False, encoding="utf-8")

    # --- Export Relationships ---
    edges_data = []
    for source, target, data in graph.edges(data=True):
        # Base relationship info
        edge_info = {
            ":START_ID": source,
            ":END_ID": target,
            ":TYPE": data.get("label", "RELATED_TO")
        }
        
        # Add all other properties, skipping 'label'
        edge_info.update({k: v for k, v in data.items() if k != "label"})
        edges_data.append(edge_info)
        
    edges_df = pd.DataFrame(edges_data)
    
    # Check if DataFrame is not empty before reordering
    if not edges_df.empty:
        # Reorder columns to put start, end, and type first
        cols = [":START_ID", ":END_ID", ":TYPE"] + [c for c in edges_df.columns if c not in [":START_ID", ":END_ID", ":TYPE"]]
        edges_df = edges_df[cols]

    edges_df.to_csv(graph_dir / "relationships.csv", index=False, encoding="utf-8")

