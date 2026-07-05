"""
Domain-agnostic graph cleanup utilities.

Fixes common issues from batch-based extraction:
- Empty/phantom nodes
- Duplicate entities with different IDs
- Orphaned references
- Inconsistent edges
"""

from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from ...logging_utils import get_component_logger
from ..provenance.identity import PROVENANCE_NODE_ATTR, merge_compact_views

logger = get_component_logger("GraphCleaner", __name__)

# Framework-owned node attributes that carry no extracted content. Excluded
# from phantom detection and content hashing so grounding metadata can never
# keep an empty node alive or stop identical entities from deduplicating.
_METADATA_NODE_ATTRS = {"id", "label", "type", PROVENANCE_NODE_ATTR}


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


def drop_self_edges(graph: nx.DiGraph) -> int:
    """
    Remove edges where source equals target (self-loops).

    Modifies the graph in-place. Returns the number of edges removed.
    """
    to_remove = [(u, v) for u, v in graph.edges() if u == v]
    for u, v in to_remove:
        graph.remove_edge(u, v)
    return len(to_remove)


def cap_edge_keywords(
    graph: nx.DiGraph,
    edge_attr: str = "keywords",
    max_keywords: int = 5,
) -> int:
    """
    Truncate list-like edge attribute (e.g. keywords) to at most max_keywords.

    Modifies edge data in-place. Returns the number of edges whose attribute was truncated.
    """
    capped = 0
    for _u, _v, data in graph.edges(data=True):
        val = data.get(edge_attr)
        if isinstance(val, list | tuple) and len(val) > max_keywords:
            data[edge_attr] = list(val)[:max_keywords]
            capped += 1
    return capped


class GraphCleaner:
    """
    Post-processing cleanup for graphs built from merged batch extractions.

    Domain-agnostic: works with any Pydantic schema.
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize cleaner."""
        self.verbose = verbose
        # Relationships lost when a dangling (phantom) target was removed, e.g.
        # {"source": ..., "label": ..., "target": ...}. Also stashed on the graph
        # (graph.graph["dropped_relationships"]) so the report can surface them.
        self.last_dropped_relationships: list[dict[str, str]] = []

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
            logger.info("Starting cleanup: %s nodes, %s edges", initial_nodes, initial_edges)

        # Step 1: Remove phantom nodes (nodes with only ID)
        removed_phantoms = self._remove_phantom_nodes(graph)

        # Step 2: Deduplicate semantically identical nodes
        merged_nodes = self._deduplicate_nodes(graph)

        # Step 2.5: Remove self-edges (source == target)
        removed_self = drop_self_edges(graph)

        # Step 3: Remove orphaned edges (edges to non-existent nodes)
        removed_edges = self._remove_orphaned_edges(graph)

        # Step 4: Remove duplicate edges
        removed_duplicates = self._remove_duplicate_edges(graph)

        # Step 5: Cap edge keywords per edge
        capped_keywords = cap_edge_keywords(graph, edge_attr="keywords", max_keywords=5)

        final_nodes = graph.number_of_nodes()
        final_edges = graph.number_of_edges()

        if self.verbose:
            logger.info(
                "Cleanup complete: removed %s phantom nodes, merged %s duplicate nodes, "
                "removed %s self-edges, %s orphaned edges, %s duplicate edges, "
                "capped keywords on %s edges | result: %s nodes (-%s), %s edges (-%s)",
                removed_phantoms,
                merged_nodes,
                removed_self,
                removed_edges,
                removed_duplicates,
                capped_keywords,
                final_nodes,
                initial_nodes - final_nodes,
                final_edges,
                initial_edges - final_edges,
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
                if key in _METADATA_NODE_ATTRS:
                    continue

                # Check if this field has meaningful value
                if is_meaningful_value(value):
                    has_meaningful_data = True
                    break

            if not has_meaningful_data:
                phantom_nodes.append(node_id)

        # Remove phantoms and redirect edges
        dropped_relationships: list[dict[str, str]] = []
        for phantom_id in phantom_nodes:
            # Find edges pointing to this phantom
            incoming = list(graph.in_edges(phantom_id, data=True))
            outgoing = list(graph.out_edges(phantom_id, data=True))

            # A dangling edge means a *lost relationship*, not just a count.
            # Record each (source, label, target) so the report can show WHAT
            # was dropped, not merely how many.
            for src, _dst, data in incoming:
                dropped_relationships.append(
                    {
                        "source": str(src),
                        "label": str(data.get("label") or ""),
                        "target": phantom_id,
                    }
                )
            for _src, dst, data in outgoing:
                dropped_relationships.append(
                    {
                        "source": phantom_id,
                        "label": str(data.get("label") or ""),
                        "target": str(dst),
                    }
                )

            # Remove the phantom
            graph.remove_node(phantom_id)

            if self.verbose and len(phantom_nodes) <= 5:
                logger.info(
                    "Removed phantom: %s (had %s incoming, %s outgoing edges)",
                    phantom_id,
                    len(incoming),
                    len(outgoing),
                )

        self.last_dropped_relationships = dropped_relationships
        # Stash on the graph so ExportStage/ReportGenerator can surface it without
        # a new plumbing path (the graph is what they already carry).
        if dropped_relationships:
            graph.graph["dropped_relationships"] = dropped_relationships
            for rel in dropped_relationships[:10]:
                logger.info(
                    "Dropped relationship (phantom target removed): %s -[%s]-> %s",
                    rel["source"],
                    rel["label"],
                    rel["target"],
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
            content_hash = self._compute_content_hash(node_data, node_id=node_id)

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

                    # Union provenance so the duplicate's grounding survives
                    merged_provenance = merge_compact_views(
                        graph.nodes[canonical_id].get(PROVENANCE_NODE_ATTR),
                        graph.nodes[dup_id].get(PROVENANCE_NODE_ATTR),
                    )
                    if merged_provenance is not None:
                        graph.nodes[canonical_id][PROVENANCE_NODE_ATTR] = merged_provenance

                    # Remove duplicate
                    graph.remove_node(dup_id)
                    merged_count += 1

                if self.verbose and merged_count <= 5:
                    logger.info("Merged %s duplicates into %s", len(duplicates), canonical_id)

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

    def _compute_content_hash(self, node_data: dict, node_id: str = "") -> str:
        """
        Compute a content-based hash for a node.

        Nodes with identical content (ignoring ID) get the same hash.
        When a node has a placeholder identity (e.g. nom="Unknown"), include node_id
        in the hash so list siblings with the same placeholder are not merged.
        """
        import hashlib
        import json

        # Extract content fields (exclude id, generated metadata, provenance —
        # identical entities with different anchors must still deduplicate)
        content_fields = {
            k: v for k, v in node_data.items() if k not in _METADATA_NODE_ATTRS and v is not None
        }

        # Normalize and sort
        content_str = json.dumps(content_fields, sort_keys=True, default=str)

        # If any field is the exact placeholder "Unknown", include node_id so list siblings stay distinct
        if any(v == "Unknown" for v in content_fields.values() if isinstance(v, str)):
            content_str += f"|{node_id}"

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
            is_meaningful_value(v) for k, v in node_data.items() if k not in _METADATA_NODE_ATTRS
        )
        if not has_meaningful_data:
            issues.append(f"Empty node: {node_id}")

    # Check 3: Count nodes/edges
    if graph.number_of_nodes() == 0:
        issues.append("Graph has no nodes")

    # Allow zero edges (e.g. degenerate extraction after best-effort salvage leaves only root)
    if graph.number_of_edges() == 0 and graph.number_of_nodes() > 0:
        logger.warning(
            "Graph has no edges (%s node(s) only). This can happen when extraction was salvaged and list entities were dropped.",
            graph.number_of_nodes(),
        )

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
            logger.error(error_msg, extra={"component": "GraphValidator"})
            return False

    return True
