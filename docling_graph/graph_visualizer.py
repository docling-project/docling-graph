"""
Handles all graph visualization tasks for the Docling-Graph package.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def _ensure_output_dir(output_dir="outputs"):
    """Helper function to create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_static_graph(graph: nx.DiGraph, filename: str = "knowledge_graph_static.png"):
    """
    Draws the graph as a static image with node properties displayed next to the nodes.

    Args:
        graph (nx.DiGraph): The NetworkX graph object.
        filename (str): The name of the image file to save.
    """
    if not graph.nodes:
        print("Graph is empty. Nothing to draw.")
        return

    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(24, 18))
    
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        print("Using 'graphviz' for a hierarchical layout.")
    except ImportError:
        print("PyGraphviz not found, using spring layout.")
        pos = nx.spring_layout(graph, seed=42, k=1.5, iterations=50)

    # 1. Draw graph elements (nodes, edges)
    nx.draw_networkx_nodes(graph, pos, node_size=2500, node_color='skyblue', node_shape='o')
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20, width=1.5, connectionstyle='arc3,rad=0.05')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=10)

    # 2. Draw main node labels inside the nodes
    main_labels = {node: data.get('label', '') for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=main_labels, font_size=10, font_weight='bold', font_color='black')

    # 3. Draw properties text to the right of each node
    x_coords, _ = zip(*pos.values())
    x_range = max(x_coords) - min(x_coords)
    offset = x_range * 0.03  # 3% of graph width

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        props_list = [f"{k}: {v}" for k, v in data.items() if k not in ['label']]
        if not props_list:
            continue
        
        props_str = "\n".join(props_list)
        plt.text(x + offset, y, props_str,
                 horizontalalignment='left', verticalalignment='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', fc='wheat', ec='black', alpha=0.6))

    plt.title("Document Knowledge Graph (Static)", fontsize=24)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, format="PNG", dpi=300)
    plt.show()
    print(f"Static graph with properties saved to {output_path}")

def create_interactive_graph(graph: nx.DiGraph, filename: str = "knowledge_graph_interactive.html") -> str:
    """
    Creates a high-quality, interactive Plotly visualization.
    
    Args:
        graph (nx.DiGraph): The NetworkX graph object.
        filename (str): The name of the HTML file to generate.

    Returns:
        str: The absolute path to the generated HTML file.
    """
    output_dir = _ensure_output_dir()
    output_path = os.path.join(output_dir, filename)
    
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    except ImportError:
        pos = nx.spring_layout(graph, k=0.9, iterations=50)

    # ... (rest of the Plotly code is unchanged) ...
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_hover_text = [], [], [], []
    for node_id, data in graph.nodes(data=True):
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get('label', ''))
        
        hover_html = f"<b>{data.get('label', 'Node')}</b><br>" + "-"*20 + "<br>"
        hover_html += "<br>".join([f"<b>{k}:</b> {v}" for k, v in data.items() if k != 'label'])
        node_hover_text.append(hover_html)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text,
        textposition="top center", hovertext=node_hover_text,
        marker=dict(showscale=False, color='skyblue', size=25, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(text='<br>Document Knowledge Graph (Interactive)', font=dict(size=16)),
                    showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.write_html(output_path)
    print(f"Interactive Plotly graph saved to: {output_path}")
    return output_path
