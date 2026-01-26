"""
Domain-agnostic graph cleanup utilities.

Fixes common issues from batch-based extraction:
- Empty/phantom nodes
- Duplicate entities with different IDs
- Orphaned references
- Inconsistent edges
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
from rich import print as rich_print

# Initialize logger for error reporting
logger = logging.getLogger(__name__)


def is_meaningful_value(value: Any) -> bool:
    """
    Check if a value is meaningful (not empty/null/default).

    Used to detect phantom nodes and empty fields.

    Args:
        value: Value to check

    Returns:
        True if value is meaningful, False otherwise

    Examples:
        >>> is_meaningful_value(None)
        False
        >>> is_meaningful_value("")
        False
        >>> is_meaningful_value([])
        False
        >>> is_meaningful_value({})
        False
        >>> is_meaningful_value("Hello")
        True
        >>> is_meaningful_value(0)
        True
        >>> is_meaningful_value(False)
        True
    """
    # None is not meaningful
    if value is None:
        return False

    # Empty strings are not meaningful
    if isinstance(value, str) and not value.strip():
        return False

    # Empty collections are not meaningful
    if isinstance(value, list | dict | set | tuple) and len(value) == 0:
        return False

    # All other values are meaningful (including 0, False, etc.)
    return True


class GraphCleaner:
    """
    Post-processing cleanup for graphs built from merged batch extractions.

    Domain-agnostic: works with any Pydantic schema.
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize cleaner."""
        self.verbose = verbose

    def clean_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Apply all cleanup operations to a graph.

        Operations applied:
        1. Remove empty/phantom nodes
        2. Deduplicate semantically identical nodes
        3. Remove orphaned edges
        4. Consolidate duplicate edges

        Args:
            graph: Input graph (modified in-place and returned)

        Returns:
            Cleaned graph (same object as input)
        """
        initial_nodes = graph.number_of_nodes()
        initial_edges = graph.number_of_edges()

        if self.verbose:
            rich_print(
                f"[blue][GraphCleaner][/blue] Starting cleanup: "
                f"{initial_nodes} nodes, {initial_edges} edges"
            )

        # Step 1: Remove phantom nodes (nodes with only ID)
        removed_phantoms = self._remove_phantom_nodes(graph)

        # Step 2: Deduplicate semantically identical nodes
        merged_nodes = self._deduplicate_nodes(graph)

        # Step 3: Remove orphaned edges (edges to non-existent nodes)
        removed_edges = self._remove_orphaned_edges(graph)

        # Step 4: Remove duplicate edges
        removed_duplicates = self._remove_duplicate_edges(graph)

        final_nodes = graph.number_of_nodes()
        final_edges = graph.number_of_edges()

        if self.verbose:
            rich_print(
                f"[green][GraphCleaner][/green] Cleanup complete:\n"
                f"  • Removed {removed_phantoms} phantom nodes\n"
                f"  • Merged {merged_nodes} duplicate nodes\n"
                f"  • Removed {removed_edges} orphaned edges\n"
                f"  • Removed {removed_duplicates} duplicate edges\n"
                f"  • Result: {final_nodes} nodes (-{initial_nodes - final_nodes}), "
                f"{final_edges} edges (-{initial_edges - final_edges})"
            )

        return graph

    def _remove_phantom_nodes(self, graph: nx.DiGraph) -> int:
        """
        Remove nodes that have no meaningful data (only 'id' field).

        These are typically created by:
        - Merge conflicts during batch processing
        - Incomplete LLM extraction
        - Component data being incorrectly set to None (the bug we're fixing)

        A node is considered a phantom if it only has metadata fields
        (id, label, type) and no actual data fields with meaningful values.
        """
        phantom_nodes = []

        for node_id, node_data in graph.nodes(data=True):
            # Check if node has any meaningful data beyond metadata
            has_meaningful_data = False

            for key, value in node_data.items():
                # Skip metadata fields
                if key in {"id", "label", "type"}:
                    continue

                # Check if this field has meaningful value
                if is_meaningful_value(value):
                    has_meaningful_data = True
                    break

            if not has_meaningful_data:
                phantom_nodes.append(node_id)

        # Remove phantoms and redirect edges
        for phantom_id in phantom_nodes:
            # Find edges pointing to this phantom
            incoming = list(graph.in_edges(phantom_id, data=True))
            outgoing = list(graph.out_edges(phantom_id, data=True))

            # Remove the phantom
            graph.remove_node(phantom_id)

            if self.verbose and len(phantom_nodes) <= 5:
                rich_print(
                    f"[yellow][GraphCleaner][/yellow] Removed phantom: {phantom_id} "
                    f"(had {len(incoming)} incoming, {len(outgoing)} outgoing edges)"
                )

        return len(phantom_nodes)

    def _deduplicate_nodes(self, graph: nx.DiGraph) -> int:
        """
        Merge nodes that represent the same entity but have different IDs.

        Uses content-based hashing to identify semantic duplicates.
        """
        # Group nodes by class and content hash
        node_groups: Dict[str, List[str]] = {}

        for node_id, node_data in graph.nodes(data=True):
            content_hash = self._compute_content_hash(node_data)

            if content_hash not in node_groups:
                node_groups[content_hash] = []
            node_groups[content_hash].append(node_id)

        # Merge duplicate groups
        merged_count = 0

        for _content_hash, node_ids in node_groups.items():
            if len(node_ids) > 1:
                # Keep first node, merge others into it
                canonical_id = node_ids[0]
                duplicates = node_ids[1:]

                for dup_id in duplicates:
                    # Redirect all edges from duplicate to canonical
                    self._redirect_edges(graph, dup_id, canonical_id)

                    # Remove duplicate
                    graph.remove_node(dup_id)
                    merged_count += 1

                if self.verbose and merged_count <= 5:
                    rich_print(
                        f"[blue][GraphCleaner][/blue] Merged {len(duplicates)} duplicates "
                        f"into {canonical_id}"
                    )

        return merged_count

    def _remove_orphaned_edges(self, graph: nx.DiGraph) -> int:
        """Remove edges that point to non-existent nodes."""
        valid_nodes = set(graph.nodes())
        orphaned_edges = []

        for source, target in graph.edges():
            if source not in valid_nodes or target not in valid_nodes:
                orphaned_edges.append((source, target))

        for source, target in orphaned_edges:
            try:
                graph.remove_edge(source, target)
            except nx.NetworkXError as e:
                # Log specific NetworkX errors (e.g., edge doesn't exist)
                logger.warning(f"Could not remove orphaned edge {source} -> {target}: {e}")
            except Exception as e:
                # Log unexpected errors for debugging
                logger.error(f"Unexpected error removing edge {source} -> {target}: {e}")

        return len(orphaned_edges)

    def _remove_duplicate_edges(self, graph: nx.DiGraph) -> int:
        """Remove duplicate edges (same source, target, and label)."""
        seen_edges: Set[Tuple[str, str, str]] = set()
        duplicate_edges = []

        for source, target, edge_data in graph.edges(data=True):
            label = edge_data.get("label", "")
            edge_sig = (source, target, label)

            if edge_sig in seen_edges:
                duplicate_edges.append((source, target))
            else:
                seen_edges.add(edge_sig)

        for source, target in duplicate_edges:
            # Keep only first occurrence
            try:
                edges = list(graph[source][target].keys())
                if len(edges) > 1:
                    for edge_key in edges[1:]:
                        graph.remove_edge(source, target, edge_key)
            except Exception:
                pass

        return len(duplicate_edges)

    def _compute_content_hash(self, node_data: dict) -> str:
        """
        Compute a content-based hash for a node.

        Nodes with identical content (ignoring ID) get the same hash.
        """
        import hashlib
        import json

        # Extract content fields (exclude id, generated metadata)
        content_fields = {
            k: v for k, v in node_data.items() if k not in {"id", "label", "type"} and v is not None
        }

        # Normalize and sort
        content_str = json.dumps(content_fields, sort_keys=True, default=str)

        # Hash
        return hashlib.blake2b(content_str.encode()).hexdigest()[:16]

    def _redirect_edges(
        self,
        graph: nx.DiGraph,
        old_node: str,
        new_node: str,
    ) -> None:
        """
        Redirect all edges from old_node to new_node.

        Preserves edge labels and attributes.
        """
        # Redirect incoming edges
        for source, _, edge_data in list(graph.in_edges(old_node, data=True)):
            if source != new_node:  # Avoid self-loops
                graph.add_edge(source, new_node, **edge_data)

        # Redirect outgoing edges
        for _, target, edge_data in list(graph.out_edges(old_node, data=True)):
            if target != new_node:  # Avoid self-loops
                graph.add_edge(new_node, target, **edge_data)


def validate_graph_structure(graph: nx.DiGraph, raise_on_error: bool = True) -> bool:
    """
    Validate that a graph has consistent structure.

    Checks:
    - All edge endpoints exist as nodes
    - No empty nodes
    - No duplicate edges

    Args:
        graph: Graph to validate
        raise_on_error: If True, raises ValueError on validation failure

    Returns:
        True if valid, False otherwise
    """
    issues = []

    # Check 1: All edge endpoints exist
    valid_nodes = set(graph.nodes())
    for source, target in graph.edges():
        if source not in valid_nodes:
            issues.append(f"Edge source not in graph: {source}")
        if target not in valid_nodes:
            issues.append(f"Edge target not in graph: {target}")

    # Check 2: No empty nodes
    for node_id, node_data in graph.nodes(data=True):
        has_meaningful_data = any(
            is_meaningful_value(v) for k, v in node_data.items() if k not in {"id", "label", "type"}
        )
        if not has_meaningful_data:
            issues.append(f"Empty node: {node_id}")

    # Check 3: Count nodes/edges
    if graph.number_of_nodes() == 0:
        issues.append("Graph has no nodes")

    if graph.number_of_edges() == 0:
        issues.append("Graph has no edges")

    # Report
    if issues:
        error_msg = f"Graph validation failed with {len(issues)} issue(s):\n"
        for issue in issues[:10]:
            error_msg += f"  - {issue}\n"
        if len(issues) > 10:
            error_msg += f"  - ... and {len(issues) - 10} more\n"

        if raise_on_error:
            raise ValueError(error_msg)
        else:
            rich_print(f"[red][GraphValidator][/red] {error_msg}")
            return False

    return True
