"""JSON exporter for graph serialization."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import networkx as nx

from ..converters.config import ExportConfig
from ..utils.string_formatter import json_serializable


def graph_to_dict(graph: nx.DiGraph) -> Dict[str, Any]:
    """Serialize a graph to the canonical docling-graph JSON shape.

    Public, file-free entry point for the shape ``graph.json`` uses:
    ``{"nodes": [...], "edges": [...], "metadata": {...}, "graph": {...}}``. The
    round-trip inverse is
    :func:`docling_graph.core.importers.graph_json.load_graph_from_dict`.

    Note: node/edge attributes may contain non-JSON-native values (dates,
    Decimals, …); pass :func:`docling_graph.core.utils.string_formatter.json_serializable`
    as ``json.dumps(..., default=...)`` when encoding the result.
    """
    return JSONExporter._graph_to_dict(graph)


class JSONExporter:
    """Export graph to JSON format."""

    def __init__(self, config: ExportConfig | None = None) -> None:
        """Initialize JSON exporter.

        Args:
            config: Export configuration. Uses defaults if None.
        """
        self.config = config or ExportConfig()

    def export(self, graph: nx.DiGraph, output_path: Path) -> None:
        """Export graph to JSON.

        Args:
            graph: NetworkX directed graph to export.
            output_path: File path where to save JSON.

        Raises:
            ValueError: If graph is empty.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot export empty graph")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph_dict = self._graph_to_dict(graph)

        with open(output_path, "w", encoding=self.config.JSON_ENCODING) as f:
            json.dump(
                graph_dict,
                f,
                indent=self.config.JSON_INDENT,
                ensure_ascii=self.config.ENSURE_ASCII,
                default=json_serializable,
            )

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty.

        Args:
            graph: NetworkX directed graph.

        Returns:
            True if graph has nodes.
        """
        num_nodes = cast(int, graph.number_of_nodes())
        return num_nodes > 0

    @staticmethod
    def _graph_to_dict(graph: nx.DiGraph) -> Dict[str, Any]:
        """Convert graph to dictionary format.

        Graph-level metadata (``graph.graph``) is serialized under a top-level
        ``"graph"`` key — format-v2 exports carry their identity contract
        (id_fields_map, template name/schema hash) there. ``nodes``/``edges``/
        ``metadata`` keep their exact historical shape for old consumers.

        Args:
            graph: NetworkX directed graph.

        Returns:
            Dictionary representation of graph.
        """
        nodes: List[Dict[str, Any]] = []
        for node_id, data in graph.nodes(data=True):
            node_dict = {"id": node_id, **data}
            nodes.append(node_dict)

        edges: List[Dict[str, Any]] = []
        for source, target, data in graph.edges(data=True):
            edge_dict = {"source": source, "target": target, **data}
            edges.append(edge_dict)

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {"node_count": len(nodes), "edge_count": len(edges)},
            "graph": dict(graph.graph),
        }
