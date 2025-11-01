"""
Cytoscape visualizer for interactive graph visualization in the browser.
"""

import json
import os
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from rich import print


class InteractiveVisualizer:
    """Visualize graphs using Cytoscape in the browser."""

    def __init__(self) -> None:
        """Initialize Cytoscape visualizer."""

    def load_csv(self, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load graph data from CSV files.

        Args:
            path: Directory containing nodes.csv and edges.csv

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        nodes_path = path / "nodes.csv"
        edges_path = path / "edges.csv"

        print(f"Loading nodes from {nodes_path}...")
        nodes_df = pd.read_csv(nodes_path)

        print(f"Loading edges from {edges_path}...")
        edges_df = pd.read_csv(edges_path)

        return nodes_df, edges_df

    def load_json(self, path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load graph data from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Tuple of (nodes_df, edges_df)
        """
        print(f"Loading graph from {path}...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)

        return nodes_df, edges_df

    def _extract_labels_for_count(self, row: pd.Series) -> list[str]:
        # 1) plural 'labels' column (array or delimited string)
        if "labels" in row and self._is_valid_value(row["labels"]):
            v = row["labels"]
            if isinstance(v, (list, tuple, set)):
                return [str(x) for x in v if str(x)]
            if isinstance(v, str):
                # split common delimiters
                if "|" in v:
                    return [p.strip() for p in v.split("|") if p.strip()]
                if "," in v:
                    return [p.strip() for p in v.split(",") if p.strip()]
                # Neo4j-like ":Research:Entity"
                if ":" in v and not v.strip().startswith("http"):
                    return [p for p in v.split(":") if p]
                return [v]
        # 2) single label-like columns
        for col in ["label", "node_label", "node_type", "type", "category", "class", "kind"]:
            if col in row and self._is_valid_value(row[col]):
                return [str(row[col])]
        return ["Unknown"]

    def _compute_node_type_counts(self, nodes_df: pd.DataFrame) -> dict:
        counts = Counter()
        for _, r in nodes_df.iterrows():
            for lab in self._extract_labels_for_count(r):
                counts[lab] += 1
        return dict(counts)

    def _is_valid_value(self, value) -> bool:
        """
        Check if a value is valid (not NaN, None, or empty).
        Handles scalars, arrays, and lists properly.
        """
        # Handle None
        if value is None:
            return False

        # Handle numpy arrays and lists
        if isinstance(value, (list, np.ndarray)):
            return len(value) > 0

        # Handle pandas NA types
        if pd.isna(value):
            return False

        # Handle empty strings
        if isinstance(value, str) and value.strip() == "":
            return False

        return True

    def _serialize_value(self, value):
        """
        Convert a value to a JSON-serializable format.
        Handles numpy types, lists, and other complex types.
        """
        # Handle None and NaN
        if value is None or (not isinstance(value, (list, np.ndarray)) and pd.isna(value)):
            return None

        # Handle numpy types
        if hasattr(value, "item"):
            return value.item()

        # Handle lists and arrays
        if isinstance(value, (list, np.ndarray)):
            return [self._serialize_value(v) for v in value]

        # Handle dicts
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Return as-is for basic types
        return value

    def prepare_data_for_cytoscape(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> dict:
        """
        Prepare dataframes for Cytoscape visualization.

        Returns a dict with 'nodes' and 'edges' in Cytoscape format.
        """
        # Prepare nodes
        nodes_list = []
        for _, row in nodes_df.iterrows():
            node_data = {}
            for key, value in row.items():
                if self._is_valid_value(value):
                    serialized = self._serialize_value(value)
                    if serialized is not None:
                        node_data[key] = serialized
            if "id" not in node_data:
                node_data["id"] = str(row.name)
            else:
                node_data["id"] = str(node_data["id"])
            # Preserve existing display label fallback for UI text
            if "label" not in node_data:
                node_data["label"] = node_data.get("name", node_data.get("type", node_data["id"]))
            nodes_list.append({"data": node_data})

        # Prepare edges
        edges_list = []
        for idx, row in edges_df.iterrows():
            edge_data = {}
            for key, value in row.items():
                if self._is_valid_value(value):
                    serialized = self._serialize_value(value)
                    if serialized is not None:
                        edge_data[key] = serialized
            if "source" not in edge_data or "target" not in edge_data:
                raise ValueError("Edges dataframe must have 'source' and 'target' columns")
            edge_data["source"] = str(edge_data["source"])
            edge_data["target"] = str(edge_data["target"])
            if "id" not in edge_data:
                edge_data["id"] = f"{edge_data['source']}-{edge_data['target']}-{idx}"
            edges_list.append({"data": edge_data})

        meta = {
            "node_types": self._compute_node_type_counts(nodes_df),
            "node_count": len(nodes_list),
            "edge_count": len(edges_list),
        }

        return {"nodes": nodes_list, "edges": edges_list, "meta": meta}

    def display_cytoscape_graph(
        self,
        path: Path,
        format: Literal["csv", "json"] = "csv",
        output_path: Optional[Path] = None,
        open_browser: bool = True,
    ) -> Path:
        """Load graph data from file and visualize with Cytoscape in the browser."""
        # Load data
        if format == "csv":
            nodes_df, edges_df = self.load_csv(path)
        elif format == "json":
            nodes_df, edges_df = self.load_json(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self._prepare_and_visualize(nodes_df, edges_df, output_path, open_browser)

    def save_cytoscape_graph(
        self, graph: nx.DiGraph, output_path: Path, open_browser: bool = False, **kwargs
    ) -> Path:
        """Visualize a NetworkX graph using Cytoscape."""
        # Convert NetworkX graph to DataFrames
        nodes_data = [{"id": str(n), **attrs} for n, attrs in graph.nodes(data=True)]
        edges_data = [
            {"source": str(s), "target": str(t), **attrs} for s, t, attrs in graph.edges(data=True)
        ]

        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)

        return self._prepare_and_visualize(nodes_df, edges_df, output_path, open_browser)

    def _prepare_and_visualize(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        output_path: Optional[Path],
        open_browser: bool,
    ) -> Path:
        """Common logic to prepare data and create Cytoscape HTML."""
        # Prepare data
        cytoscape_elements = self.prepare_data_for_cytoscape(nodes_df, edges_df)

        if output_path is None:
            output_path = Path("outputs/temp_graph.html")
        elif not str(output_path).endswith(".html"):
            output_path = Path(str(output_path) + ".html")

        return self._export_and_open(cytoscape_elements, output_path, open_browser)

    def _export_and_open(self, elements: dict, output_path: Path, open_browser: bool) -> Path:
        """
        Write the Cytoscape HTML and optionally open it in the default browser.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_cytoscape_html(elements, output_path)

        if open_browser:
            webbrowser.open("file://" + os.path.abspath(str(output_path)))

        return output_path

    def _write_cytoscape_html(
        self,
        elements: dict,
        path: Path,
    ) -> None:
        """
        Export Cytoscape visualization to a standalone HTML file.
        """
        # Read the template
        template_path = Path(__file__).parent / "assets/interactive_template.html"

        if template_path.exists():
            with open(template_path, encoding="utf-8") as f:
                html_template = f.read()
        else:
            # Fallback: use inline template (minimal version)
            print(
                "[yellow][InteractiveVisualizer][/yellow] HTML template missing - Falling back to minimal version instead"
            )
            html_template = self._get_default_template()

        # Inject the graph data with proper JSON serialization
        elements_json = json.dumps(elements, indent=2, ensure_ascii=False)

        # Replace the placeholder with the actual JavaScript variable declaration
        html_content = html_template.replace(
            "/* ELEMENTS_DATA_PLACEHOLDER */", f"const graphElements = {elements_json};"
        )

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _get_default_template(self) -> str:
        """Get the default HTML template if external file is not found."""
        return """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Knowledge Graph - Cytoscape</title>
                <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
                <style>
                    body { margin: 0; padding: 0; }
                    #cy { width: 100%; height: 100vh; background: #1a1a2e; }
                </style>
            </head>
            <body>
                <div id="cy"></div>
                <script>
                    /* ELEMENTS_DATA_PLACEHOLDER */
                    cytoscape({
                        container: document.getElementById('cy'),
                        elements: [...graphElements.nodes, ...graphElements.edges],
                        style: [
                            { selector: 'node', style: { 'background-color': '#667eea', 'label': 'data(label)' }},
                            { selector: 'edge', style: { 'width': 2, 'line-color': '#718096', 'target-arrow-shape': 'triangle' }}
                        ],
                        layout: { name: 'cose', animate: true }
                    });
                </script>
            </body>
            </html>"""
