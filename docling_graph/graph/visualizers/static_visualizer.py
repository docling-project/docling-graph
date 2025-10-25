"""Static graph visualizer using Matplotlib."""

from typing import Optional, Literal
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

from ..utils.formatting import format_property_key, format_property_value
from ..core.config import VisualizationConfig


class StaticVisualizer:
    """Create static graph visualizations with Matplotlib."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize static visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()

    def visualize(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        format: Literal["png", "svg", "pdf"] = "png",
        show_properties: bool = True,
        max_properties: int = 5
    ) -> None:
        """Create static visualization of graph.

        Args:
            graph: NetworkX directed graph to visualize.
            output_path: Base path for output file (extension added automatically).
            format: Output format ('png', 'svg', or 'pdf').
            show_properties: Whether to show node properties in labels.
            max_properties: Maximum number of properties to show per node.

        Raises:
            ValueError: If graph is empty or format is invalid.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot visualize empty graph")

        if format not in self.config.STATIC_FORMATS:
            raise ValueError(
                f"Invalid format '{format}'. "
                f"Must be one of: {', '.join(self.config.STATIC_FORMATS)}"
            )

        # Create figure
        plt.figure(figsize=self.config.STATIC_FIGURE_SIZE)

        # Generate layout
        pos = nx.spring_layout(graph, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=self.config.STATIC_NODE_COLOR,
            node_size=self.config.STATIC_NODE_SIZE,
            alpha=0.7
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color=self.config.STATIC_EDGE_COLOR,
            arrows=True,
            arrowsize=20,
            alpha=0.5
        )

        # Create node labels
        if show_properties:
            labels = self._create_node_labels(graph, max_properties)
        else:
            labels = {node: str(node) for node in graph.nodes()}

        nx.draw_networkx_labels(
            graph,
            pos,
            labels,
            font_size=self.config.STATIC_FONT_SIZE,
            font_weight='bold'
        )

        # Create edge labels
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels,
            font_size=self.config.STATIC_FONT_SIZE - 2
        )

        plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save with appropriate extension
        output_file = Path(str(output_path) + f".{format}")
        plt.savefig(
            output_file,
            format=format,
            dpi=self.config.STATIC_DPI,
            bbox_inches='tight'
        )
        plt.close()

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty.

        Args:
            graph: NetworkX directed graph.

        Returns:
            True if graph has nodes.
        """
        return graph.number_of_nodes() > 0

    @staticmethod
    def _create_node_labels(
        graph: nx.DiGraph,
        max_properties: int
    ) -> dict[str, str]:
        """Create detailed labels for nodes with properties.

        Args:
            graph: NetworkX directed graph.
            max_properties: Maximum properties to show.

        Returns:
            Dictionary mapping node IDs to labels.
        """
        labels = {}

        for node_id, data in graph.nodes(data=True):
            # Start with node label
            label_parts = [data.get('label', 'Node')]

            # Add up to max_properties
            prop_count = 0
            for key, value in data.items():
                if key == 'label' or prop_count >= max_properties:
                    continue

                formatted_key = format_property_key(key)
                formatted_value = format_property_value(value, max_length=20)
                label_parts.append(f"{formatted_key}: {formatted_value}")
                prop_count += 1

            labels[node_id] = '\n'.join(label_parts)

        return labels
