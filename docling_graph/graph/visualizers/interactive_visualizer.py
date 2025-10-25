"""Interactive graph visualizer using Pyvis."""

from pyvis.network import Network
from typing import Optional
from pathlib import Path
import networkx as nx

from ..utils.formatting import format_property_key, format_property_value
from ..core.config import VisualizationConfig


class InteractiveVisualizer:
    """Create interactive HTML graph visualizations with Pyvis."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize interactive visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()

    def visualize(
        self,
        graph: nx.DiGraph,
        output_path: Path,
        notebook: bool = False
    ) -> None:
        """Create interactive HTML visualization of graph.

        Args:
            graph: NetworkX directed graph to visualize.
            output_path: Path for output HTML file (extension added if missing).
            notebook: Whether to use notebook mode.

        Raises:
            ValueError: If graph is empty.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot visualize empty graph")

        # Ensure .html extension
        if not str(output_path).endswith('.html'):
            output_path = Path(str(output_path) + '.html')

        # Create Pyvis network
        net = Network(
            height=self.config.INTERACTIVE_HEIGHT,
            width=self.config.INTERACTIVE_WIDTH,
            bgcolor=self.config.INTERACTIVE_BGCOLOR,
            font_color=self.config.INTERACTIVE_FONT_COLOR,
            directed=True,
            notebook=notebook
        )

        # Add nodes with tooltips
        for node_id, data in graph.nodes(data=True):
            title = self._create_node_tooltip(node_id, data)
            label = data.get('label', str(node_id))

            net.add_node(
                str(node_id),
                label=label,
                title=title,
                color='lightblue'
            )

        # Add edges with labels
        for source, target, data in graph.edges(data=True):
            edge_label = data.get('label', '')
            title = self._create_edge_tooltip(source, target, data)

            net.add_edge(
                str(source),
                str(target),
                label=edge_label,
                title=title
            )

        # Configure physics - using Python multiline string properly
        physics_config = '''
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 100
            },
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04
            }
          }
        }
        '''
        net.set_options(physics_config)

        # Save
        net.save_graph(str(output_path))

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty.

        Args:
            graph: NetworkX directed graph.

        Returns:
            True if graph has nodes.
        """
        return graph.number_of_nodes() > 0

    def _create_node_tooltip(self, node_id: str, data: dict) -> str:
        """Create HTML tooltip for node with properties.

        Args:
            node_id: Node identifier.
            data: Node data dictionary.

        Returns:
            HTML string for tooltip.
        """
        tooltip_parts = [f"<b>{data.get('label', 'Node')}</b>"]
        tooltip_parts.append(f"<br>ID: {node_id}")

        # Add properties
        for key, value in data.items():
            if key in ('label', 'type'):
                continue

            formatted_key = format_property_key(key)
            formatted_value = format_property_value(
                value,
                max_length=self.config.MAX_TOOLTIP_LENGTH
            )
            tooltip_parts.append(f"<br>{formatted_key}: {formatted_value}")

        return ''.join(tooltip_parts)

    def _create_edge_tooltip(
        self,
        source: str,
        target: str,
        data: dict
    ) -> str:
        """Create HTML tooltip for edge.

        Args:
            source: Source node ID.
            target: Target node ID.
            data: Edge data dictionary.

        Returns:
            HTML string for tooltip.
        """
        label = data.get('label', 'related_to')
        tooltip_parts = [
            f"<b>{label}</b>",
            f"<br>From: {source}",
            f"<br>To: {target}"
        ]

        # Add edge properties if any
        for key, value in data.items():
            if key == 'label':
                continue
            formatted_key = format_property_key(key)
            formatted_value = format_property_value(value, max_length=50)
            tooltip_parts.append(f"<br>{formatted_key}: {formatted_value}")

        return ''.join(tooltip_parts)
