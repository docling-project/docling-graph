"""Base protocol for graph visualizers."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx


@runtime_checkable
class GraphVisualizerProtocol(Protocol):
    """Protocol for graph visualization implementations."""

    def visualize(self, graph: nx.DiGraph, output_path: Path, **kwargs: Any) -> None:
        """Create visualization of the graph.

        Args:
            graph: NetworkX directed graph to visualize.
            output_path: Path where to save visualization.
            **kwargs: Additional visualization options.
        """
        ...

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph can be visualized.

        Args:
            graph: NetworkX directed graph to validate.

        Returns:
            True if graph is valid for visualization.
        """
        ...
