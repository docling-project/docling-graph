"""Graph statistics and analysis utilities."""

from ..core.models import GraphMetadata
from typing import Dict
import networkx as nx


def calculate_graph_stats(graph: nx.DiGraph, source_model_count: int) -> GraphMetadata:
    """Calculate statistics for a graph.

    Args:
        graph: NetworkX directed graph.
        source_model_count: Number of source Pydantic models.

    Returns:
        GraphMetadata object with statistics.
    """
    node_types = get_node_type_distribution(graph)
    edge_types = get_edge_type_distribution(graph)

    return GraphMetadata(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        node_types=node_types,
        edge_types=edge_types,
        source_models=source_model_count
    )


def get_node_type_distribution(graph: nx.DiGraph) -> Dict[str, int]:
    """Get distribution of node types in graph.

    Args:
        graph: NetworkX directed graph.

    Returns:
        Dictionary mapping node type/label to count.
    """
    type_counts: Dict[str, int] = {}

    for _, data in graph.nodes(data=True):
        node_type = data.get('label', 'Unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    return type_counts


def get_edge_type_distribution(graph: nx.DiGraph) -> Dict[str, int]:
    """Get distribution of edge types in graph.

    Args:
        graph: NetworkX directed graph.

    Returns:
        Dictionary mapping edge type/label to count.
    """
    type_counts: Dict[str, int] = {}

    for _, _, data in graph.edges(data=True):
        edge_type = data.get('label', 'Unknown')
        type_counts[edge_type] = type_counts.get(edge_type, 0) + 1

    return type_counts
