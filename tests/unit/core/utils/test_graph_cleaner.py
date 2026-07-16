import networkx as nx
import pytest

from docling_graph.core.utils.graph_cleaner import (
    GraphCleaner,
    cap_edge_keywords,
    drop_self_edges,
    is_meaningful_value,
    validate_graph_structure,
)


@pytest.fixture
def cleaner():
    # Disable verbose printing during tests
    return GraphCleaner(verbose=False)


@pytest.fixture
def dirty_graph() -> nx.DiGraph:
    """Returns a graph with duplicates, phantoms, and orphans."""
    g = nx.DiGraph()
    # Add nodes
    g.add_node("node-1", name="Alice")
    g.add_node("node-2", name="Acme")
    g.add_node("node-3", name="Bob")
    # Add a semantic duplicate node
    g.add_node("node-4", name="Alice")
    # Add a phantom node (only metadata)
    g.add_node("phantom-1", id="phantom-1", label="Person")

    # Add edges
    g.add_edge("node-1", "node-2", label="WORKS_AT")
    # Add a duplicate edge
    g.add_edge("node-1", "node-2", label="WORKS_AT")
    # Add edge from the semantic duplicate
    g.add_edge("node-4", "node-2", label="WORKS_AT")
    # Add edge to the phantom node
    g.add_edge("node-3", "phantom-1", label="KNOWS")
    # Add an orphaned edge (node-99 doesn't exist)
    g.add_edge("node-1", "node-99", label="ORPHAN")

    return g


def test_clean_graph(cleaner: GraphCleaner, dirty_graph: nx.DiGraph):
    """Test the full clean_graph method."""
    assert dirty_graph.number_of_nodes() == 6
    assert dirty_graph.number_of_edges() == 4

    # Run the cleanup
    cleaned_graph = cleaner.clean_graph(dirty_graph)

    # Check nodes:
    # "node-1" (canonical)
    # "node-2"
    # "node-3"
    # "node-4" (merged into "node-1")
    # "phantom-1" (removed)
    assert cleaned_graph.number_of_nodes() == 3
    assert "node-1" in cleaned_graph
    assert "node-2" in cleaned_graph
    assert "node-3" in cleaned_graph
    assert "node-4" not in cleaned_graph
    assert "phantom-1" not in cleaned_graph

    # Check edges:
    # 1. ("node-1", "node-2") - original
    # 2. ("node-1", "node-2") - duplicate, removed
    # 3. ("node-4", "node-2") - redirected to ("node-1", "node-2"), removed as duplicate
    # 4. ("node-3", "phantom-1") - removed (phantom node deleted)
    # 5. ("node-1", "node-99") - removed (orphaned)
    assert cleaned_graph.number_of_edges() == 1
    assert cleaned_graph.has_edge("node-1", "node-2")


def test_phantom_removal_records_dropped_relationships(
    cleaner: GraphCleaner, dirty_graph: nx.DiGraph
):
    """Q3: removing a phantom records the (source, label, target) of the lost edge."""
    cleaned = cleaner.clean_graph(dirty_graph)
    dropped = cleaner.last_dropped_relationships
    assert {"source": "node-3", "label": "KNOWS", "target": "phantom-1"} in dropped
    # Stashed on the graph so the report can surface it without extra plumbing.
    assert cleaned.graph.get("dropped_relationships") == dropped


def test_phantom_removal_records_outgoing_edges_too():
    """Q3: a phantom's OUTGOING edges are also lost relationships, not just incoming."""
    g = nx.DiGraph()
    g.add_node("phantom-1", id="phantom-1", label="Person")
    g.add_node("node-a", name="Alice")
    g.add_edge("node-a", "phantom-1", label="KNOWS")
    g.add_edge("phantom-1", "node-a", label="MANAGES")

    cleaner = GraphCleaner(verbose=True)
    cleaner.clean_graph(g)

    dropped = cleaner.last_dropped_relationships
    assert {"source": "node-a", "label": "KNOWS", "target": "phantom-1"} in dropped
    assert {"source": "phantom-1", "label": "MANAGES", "target": "node-a"} in dropped


def test_validate_graph_structure_valid():
    """Test validation on a clean graph."""
    g = nx.DiGraph()
    g.add_node("A", name="Node A")
    g.add_node("B", name="Node B")
    g.add_edge("A", "B", label="CONNECTS")

    assert validate_graph_structure(g, raise_on_error=True) is True


def test_validate_graph_structure_orphan_edge():
    """Test validation failure for an auto-created empty node."""
    g = nx.DiGraph()
    g.add_node("A", name="Node A")
    g.add_edge("A", "B", label="CONNECTS")  # networkx auto-creates node "B" with no data

    with pytest.raises(ValueError, match="Empty node: B"):
        validate_graph_structure(g, raise_on_error=True)


def test_validate_graph_structure_empty_node():
    """Test validation failure for an empty node."""
    g = nx.DiGraph()
    g.add_node("A", name="Node A")
    g.add_node("B", id="B", label="Test")  # Empty node

    with pytest.raises(ValueError, match="Empty node: B"):
        validate_graph_structure(g, raise_on_error=True)


def test_validate_graph_structure_allows_single_node_no_edges():
    """Single-node graph with no edges is allowed (e.g. degenerate extraction after salvage)."""
    g = nx.DiGraph()
    g.add_node("root", id="root", label="Document", __class__="AssuranceMRH")
    assert validate_graph_structure(g, raise_on_error=True) is True
    assert g.number_of_nodes() == 1
    assert g.number_of_edges() == 0


def test_validate_graph_structure_empty_graph_reports_no_nodes():
    with pytest.raises(ValueError, match="Graph has no nodes"):
        validate_graph_structure(nx.DiGraph(), raise_on_error=True)


def test_validate_graph_structure_truncates_long_issue_list():
    g = nx.DiGraph()
    for i in range(15):
        g.add_node(f"empty-{i}", id=f"empty-{i}", label="Thing")  # no meaningful data
    with pytest.raises(ValueError, match=r"and 5 more") as exc_info:
        validate_graph_structure(g, raise_on_error=True)
    assert "15 issue(s)" in str(exc_info.value)


def test_validate_graph_structure_returns_false_without_raising():
    g = nx.DiGraph()
    g.add_node("empty", id="empty", label="Thing")  # no meaningful data
    assert validate_graph_structure(g, raise_on_error=False) is False


def test_is_meaningful_value_none_and_empty_collections():
    assert is_meaningful_value(None) is False
    assert is_meaningful_value("") is False
    assert is_meaningful_value("   ") is False
    assert is_meaningful_value([]) is False
    assert is_meaningful_value({}) is False
    assert is_meaningful_value("Hello") is True
    assert is_meaningful_value(0) is True
    assert is_meaningful_value(False) is True


def test_content_hash_disambiguates_placeholder_siblings_by_node_id(cleaner: GraphCleaner):
    """A literal 'Unknown' placeholder value folds in node_id so list siblings
    that all share the same placeholder don't spuriously dedupe."""
    base = {"label": "Person", "type": "entity", "__class__": "Person", "nom": "Unknown"}
    h1 = cleaner._compute_content_hash(base, node_id="person-1")
    h2 = cleaner._compute_content_hash(base, node_id="person-2")
    assert h1 != h2


def test_merge_audit_attrs_do_not_block_dedup(cleaner: GraphCleaner):
    """__conflicts__/merged_from (graph-merge audit records) are metadata:
    content-identical nodes still deduplicate when one side carries them."""
    g = nx.DiGraph()
    g.add_node("node-1", name="Alice")
    g.add_node(
        "node-2",
        name="Alice",
        __conflicts__=[{"field": "age", "dropped": 30}],
        merged_from=[{"document_id": "doc-a"}],
    )
    cleaned = cleaner.clean_graph(g)
    assert cleaned.number_of_nodes() == 1


def test_node_with_only_merge_audit_attrs_is_phantom(cleaner: GraphCleaner):
    """Merge audit records alone can't keep an otherwise-empty node alive."""
    g = nx.DiGraph()
    g.add_node("keeper", name="Alice")
    g.add_node(
        "audit-only",
        id="audit-only",
        label="Person",
        merged_from=[{"document_id": "doc-a"}],
    )
    cleaned = cleaner.clean_graph(g)
    assert "keeper" in cleaned
    assert "audit-only" not in cleaned


def test_redirect_edges_skips_self_loop_creation(cleaner: GraphCleaner):
    """Redirecting a duplicate's edges never creates a self-loop on the canonical node."""
    g = nx.DiGraph()
    g.add_node("canonical", name="A")
    g.add_node("dup", name="A")
    g.add_edge("canonical", "dup", label="REL")  # would become a self-loop if redirected
    g.add_edge("dup", "canonical", label="REL")  # likewise

    cleaner._redirect_edges(g, old_node="dup", new_node="canonical")

    assert not g.has_edge("canonical", "canonical")


def test_redirect_edges_moves_genuine_incoming_edge(cleaner: GraphCleaner):
    """A duplicate's incoming edge from an unrelated node is redirected to the canonical."""
    g = nx.DiGraph()
    g.add_node("canonical", name="A")
    g.add_node("dup", name="A")
    g.add_node("other", name="B")
    g.add_edge("other", "dup", label="KNOWS")

    cleaner._redirect_edges(g, old_node="dup", new_node="canonical")

    assert g.has_edge("other", "canonical")
    assert g["other"]["canonical"]["label"] == "KNOWS"


def test_drop_self_edges():
    """Self-edges (source == target) are removed."""
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", label="X")
    g.add_edge("A", "A", label="self")
    g.add_edge("B", "B", label="self2")
    removed = drop_self_edges(g)
    assert removed == 2
    assert not g.has_edge("A", "A")
    assert not g.has_edge("B", "B")
    assert g.has_edge("A", "B")
    assert g.number_of_edges() == 1


def test_cap_edge_keywords():
    """Edge keywords list is truncated to max_keywords."""
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", keywords=["a", "b", "c", "d", "e", "f"], label="X")
    capped = cap_edge_keywords(g, edge_attr="keywords", max_keywords=5)
    assert capped == 1
    assert g.edges[("A", "B")]["keywords"] == ["a", "b", "c", "d", "e"]
