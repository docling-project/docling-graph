"""Interactive graph visualizer using Pyvis."""

from pyvis.network import Network
from typing import Optional
from pathlib import Path
import networkx as nx

from ..utils.formatting import format_property_key, format_property_value
from ..base.config import VisualizationConfig


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
        notebook: bool = False,
        tooltip_max_length: int = 60
    ) -> None:
        """Create interactive HTML visualization matching original graph_visualizer.py.

        Physics is disabled after initial stabilization to allow manual node positioning.

        Args:
            graph: NetworkX directed graph to visualize.
            output_path: Path for output HTML file (without extension).
            notebook: Whether to use notebook mode.
            tooltip_max_length: Maximum character length for tooltip values.

        Raises:
            ValueError: If graph is empty.
        """
        if not self.validate_graph(graph):
            raise ValueError("Cannot visualize empty graph")

        # Ensure .html extension (matches original: Path(filename).with_suffix(".html"))
        output_file = Path(str(output_path)).with_suffix(".html")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create network with enhanced options
        net = Network(
            height=self.config.INTERACTIVE_HEIGHT,
            width=self.config.INTERACTIVE_WIDTH,
            directed=self.config.INTERACTIVE_DIRECTED,
            bgcolor=self.config.INTERACTIVE_BGCOLOR,
            font_color=self.config.INTERACTIVE_FONT_COLOR,
            notebook=notebook,
            cdn_resources=self.config.INTERACTIVE_CDN_RESOURCES
        )

        # Configure physics for better layout
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08,
                    "avoidOverlap": 1.0
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 150
                }
            },
            "nodes": {
                "shape": "dot",
                "size": 25,
                "font": {"size": 16, "face": "arial"},
                "borderWidth": 2
            },
            "edges": {
                "width": 2,
                "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
                "smooth": {"type": "continuous"}
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true,
                "dragNodes": true
            }
        }
        """)

        # Add nodes with enhanced styling and proper tooltips
        for node_id, data in graph.nodes(data=True):
            label = data.get('label', str(node_id))

            # Create plain text tooltip
            tooltip_parts = [f"{label}"]
            tooltip_parts.append("-" * 30)

            # Filter properties
            important_props = {k: v for k, v in data.items()
                             if k not in ['label', 'pos', 'id'] and v is not None and str(v).strip()}

            if important_props:
                for k, v in important_props.items():
                    formatted_key = format_property_key(k)
                    formatted_value = format_property_value(v, tooltip_max_length)
                    tooltip_parts.append(f"{formatted_key}: {formatted_value}")
            else:
                tooltip_parts.append("No additional properties")

            # Join with newlines for plain text tooltip
            tooltip_text = "\n".join(tooltip_parts)

            net.add_node(
                node_id,
                label=label,
                title=tooltip_text,  # Plain text, not HTML
                color='#4A90E2',
                borderWidth=2,
                borderWidthSelected=4
            )

        # Add edges with labels
        for source, target, data in graph.edges(data=True):
            edge_label = data.get('label', '')

            net.add_edge(
                source,
                target,
                label=edge_label,
                color="#B35045",
                title=edge_label
            )

        # Save the graph first
        net.save_graph(str(output_file))

        # Read and inject custom JavaScript
        # This disables physics after stabilization to allow manual positioning
        with open(output_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Inject JavaScript before the closing </script> tag
        injection_point = html_content.rfind('</script>')
        if injection_point != -1:
            custom_js = """

            // Disable physics after stabilization to allow manual positioning
            network.on("stabilizationIterationsDone", function () {
                network.setOptions({ physics: false });
                console.log("Physics disabled after stabilization");
            });
            """
            html_content = html_content[:injection_point] + custom_js + html_content[injection_point:]

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def validate_graph(self, graph: nx.DiGraph) -> bool:
        """Validate that graph is not empty."""
        return graph.number_of_nodes() > 0
