"""CSV exporter for Neo4j-compatible format."""

import csv
from pathlib import Path
from typing import Optional, cast

import networkx as nx
import pandas as pd

from ..converters.config import ExportConfig


class CSVExporter:
    """Export graph to CSV format compatible with Neo4j import."""

    def __init__(self, config: ExportConfig | None = None) -> None:
        """Initialize CSV exporter.

        Args:
            config: Export configuration. Uses defaults if None.
        """
        self.config = config or ExportConfig()

    def export(self, graph: nx.DiGraph, output_path: Path) -> None:
        """Export graph to CSV files (nodes and edges).

        Args:
            graph: NetworkX directed graph to export.
            output_path: Directory path where to save CSV files.

        Raises:
            ValueError: If graph is empty.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot export empty graph")

        output_path.mkdir(parents=True, exist_ok=True)

        # Export nodes
        nodes_path = output_path / self.config.CSV_NODE_FILENAME
        self._export_nodes(graph, nodes_path)

        # Export edges
        edges_path = output_path / self.config.CSV_EDGE_FILENAME
        self._export_edges(graph, edges_path)

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty.

        Args:
            graph: NetworkX directed graph.

        Returns:
            True if graph has nodes.
        """
        num_nodes = cast(int, graph.number_of_nodes())
        return num_nodes > 0

    def _export_nodes(self, graph: nx.DiGraph, path: Path) -> None:
        """Export nodes to CSV.

        Args:
            graph: NetworkX directed graph.
            path: Path to save nodes CSV.
        """
        nodes_data = []

        for node_id, data in graph.nodes(data=True):
            node_dict = {"id": node_id, **data}
            nodes_data.append(node_dict)

        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(
            path,
            index=False,
            encoding=self.config.CSV_ENCODING,
            quoting=csv.QUOTE_NONNUMERIC,
            doublequote=True,
        )

    def _export_edges(self, graph: nx.DiGraph, path: Path) -> None:
        """Export edges to CSV.

        Args:
            graph: NetworkX directed graph.
            path: Path to save edges CSV.
        """
        edges_data = []

        for source, target, data in graph.edges(data=True):
            edge_dict = {"source": source, "target": target, **data}
            edges_data.append(edge_dict)

        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(
            path,
            index=False,
            encoding=self.config.CSV_ENCODING,
            quoting=csv.QUOTE_NONNUMERIC,
            doublequote=True,
        )
