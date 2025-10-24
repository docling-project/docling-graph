"""
Enhanced graph visualization with multiple export formats.

Provides functions to create static and interactive visualizations
of knowledge graphs using Matplotlib and Pyvis, along with
markdown report generation.
"""

from pyvis.network import Network
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

def _format_property_value(value, max_length: int = 80) -> str:
    """Format property value with smart truncation."""
    str_val = str(value)
    if len(str_val) <= max_length:
        return str_val
    return str_val[:max_length-3] + "..."


def _format_property_key(key: str) -> str:
    """Convert snake_case or camelCase to Title Case."""
    # Handle snake_case
    if '_' in key:
        return ' '.join(word.capitalize() for word in key.split('_'))
    # Handle camelCase
    import re
    return re.sub(r'([A-Z])', r' \1', key).strip().title()


def create_static_graph(
    graph: nx.DiGraph,
    filename: str = "knowledge_graph_static",
    format: str = "png",  # Options: png, svg, pdf
    show_properties: bool = True,
    max_properties: int = 5
):
    """
    Enhanced matplotlib visualization with SVG/PDF support for publication quality.

    Args:
        graph: NetworkX DiGraph to visualize
        filename: Output filename (without extension)
        format: Output format (png, svg, pdf)
        show_properties: Whether to show property boxes next to nodes
        max_properties: Maximum number of properties to display per node
    """
    if not graph.nodes:
        print("Graph is empty. Nothing to draw.")
        return

    output_path = Path(filename).with_suffix(f".{format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context():
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(24, 18), facecolor='white')

        # Choose layout based on graph size
        if graph.number_of_nodes() < 20:
            try:
                pos = nx.kamada_kawai_layout(graph)
            except:
                pos = nx.spring_layout(graph, seed=42, k=2, iterations=100)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=150)

        # Enhanced node styling
        nx.draw_networkx_nodes(
            graph, pos,
            node_size=4000,
            node_color='#4A90E2',
            edgecolors='#2C3E50',
            linewidths=2.5,
            alpha=0.9,
            ax=ax
        )

        # Enhanced edge styling with better visibility
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='#95A5A6',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=30,
            width=2.5,
            connectionstyle='arc3,rad=0.15',
            alpha=0.7,
            ax=ax
        )

        # Edge labels with background
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_color='#B35045',
            font_size=12,
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8),
            ax=ax
        )

        # Node labels with properties
        for node, (x, y) in pos.items():
            data = graph.nodes[node]
            label = data.get('label', str(node))

            # Main label
            ax.text(
                x, y, label,
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white',
                zorder=10
            )

            # Properties box with improved formatting
            if show_properties:
                # Exclude internal keys and filter meaningful properties
                important_props = {k: v for k, v in data.items()
                                if k not in ['label', 'pos', 'id'] and v is not None and str(v).strip()}

                if important_props:
                    # Limit to most important properties
                    props_items = list(important_props.items())[:max_properties]

                    # Format each property with clear labels
                    props_lines = []
                    for k, v in props_items:
                        formatted_key = _format_property_key(k)
                        formatted_value = _format_property_value(v, max_length=40)
                        props_lines.append(f"{formatted_key}:")
                        props_lines.append(f"  {formatted_value}")

                    props_text = '\n'.join(props_lines)

                    ax.text(
                        x + 0.12, y, props_text,
                        ha='left', va='center',
                        fontsize=8,
                        family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', fc='#FFF8DC',
                                ec='#2C3E50', alpha=0.95, linewidth=1.5),
                        zorder=5
                    )

        ax.set_title(
            "Document Knowledge Graph",
            fontsize=24, fontweight='bold',
            pad=30, color='#2C3E50'
        )
        ax.axis('off')
        plt.tight_layout()

        # Save with appropriate DPI based on format
        dpi = 300 if format == 'png' else None
        plt.savefig(
            output_path,
            format=format.upper(),
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)


def create_interactive_graph(
    graph: nx.DiGraph,
    filename: str = "knowledge_graph_interactive",
    tooltip_max_length: int = 60
):
    """
    Enhanced Pyvis interactive visualization with better styling and tooltips.
    Physics is disabled after initial stabilization to allow manual node positioning.
    
    Args:
        graph: NetworkX DiGraph to visualize
        filename: Output filename (without extension)
        tooltip_max_length: Maximum character length for tooltip values
    """
    output_path = Path(filename).with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create network with enhanced options
    net = Network(
        height="calc(100vh - 20px)",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#2C3E50",
        notebook=False,
        cdn_resources='remote'
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
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
        
        # Create plain text tooltip (pyvis escapes HTML, so use plain text with line breaks)
        tooltip_parts = [f"{label}"]
        tooltip_parts.append("-" * 30)
        
        # Filter properties
        important_props = {k: v for k, v in data.items()
                          if k not in ['label', 'pos', 'id'] and v is not None and str(v).strip()}
        
        if important_props:
            for k, v in important_props.items():
                formatted_key = _format_property_key(k)
                formatted_value = _format_property_value(v, tooltip_max_length)
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
    net.save_graph(str(output_path))
    
    # Read the generated HTML and inject custom JavaScript to disable physics after stabilization
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Inject JavaScript before the closing </body> tag
    custom_script = """
    <script type="text/javascript">
        // Disable physics after the network stabilizes
        network.once("stabilizationIterationsDone", function() {
            network.setOptions({ physics: false });
            console.log("Physics disabled - nodes will stay where you drag them");
        });
    </script>
    """
    
    # Insert the script before </body>
    html_content = html_content.replace('</body>', custom_script + '\n</body>')
    
    # Write the modified HTML back
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def create_markdown_report(
    graph: nx.Graph,
    filename: str = "knowledge_graph_report"
):
    """Enhanced markdown report with statistics."""
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty. Skipping markdown report.")
        return

    output_path = Path(filename).with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = []
    md_content.append("# Knowledge Graph Report\n")
    md_content.append("## Graph Statistics\n")
    md_content.append(f"- **Total Nodes**: {graph.number_of_nodes()}")
    md_content.append(f"- **Total Edges**: {graph.number_of_edges()}")

    if graph.number_of_nodes() > 0:
        avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        md_content.append(f"- **Average Degree**: {avg_degree:.2f}")

    md_content.append("\n---\n")
    md_content.append("## Nodes (with properties)\n")

    for node_id, data in graph.nodes(data=True):
        label = data.get("label", node_id)
        md_content.append(f"### Node: {label} (`{node_id}`)")

        if data:
            for key, value in data.items():
                formatted_key = _format_property_key(key)
                md_content.append(f"- **{formatted_key}**: {value}")
        else:
            md_content.append("- *(no properties)*")
        md_content.append("")

    md_content.append("## Edges (with labels)\n")
    for u, v, data in graph.edges(data=True):
        u_label = graph.nodes[u].get('label', u)
        v_label = graph.nodes[v].get('label', v)
        edge_label = data.get('label', 'HAS_RELATION')
        md_content.append(f"- **{u_label} → {v_label}** `{edge_label}`")

    md_content.append("\n---\n")
    md_content.append("_End of Graph Summary_\n")

    output_path.write_text("\n".join(md_content), encoding="utf-8")