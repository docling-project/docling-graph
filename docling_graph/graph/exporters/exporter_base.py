"""Base protocol for graph exporters."""

from typing import Protocol, runtime_checkable
from pathlib import Path
import networkx as nx


@runtime_checkable
class GraphExporterProtocol(Protocol):
    """Protocol for graph export implementations."""

    def export(self, graph: nx.DiGraph, output_path: Path) -> None:
        """Export graph to specified format.

        Args:
            graph: NetworkX directed graph to export.
            output_path: Path where to save the exported graph.
        """
        ...

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph can be exported.

        Args:
            graph: NetworkX directed graph to validate.

        Returns:
            True if graph is valid for export.
        """
        ...
