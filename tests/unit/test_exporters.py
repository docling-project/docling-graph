"""
Unit tests for exporters (CSV, Cypher, JSON).
"""

from pathlib import Path

import pytest

from docling_graph.core.exporters import CSVExporter, CypherExporter, JSONExporter


class TestCSVExporter:
    """Tests for CSV exporter."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to CSV."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)  # CSVExporter exports to directory
        nodes_file = temp_dir / "nodes.csv"
        edges_file = temp_dir / "edges.csv"
        assert nodes_file.exists()
        assert edges_file.exists()

    def test_nodes_csv_has_header(self, simple_graph, temp_dir):
        """Test that nodes CSV has correct header."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)
        nodes_file = temp_dir / "nodes.csv"
        content = nodes_file.read_text()
        assert "id" in content.lower() or ":id" in content.lower()
        assert "label" in content.lower() or ":label" in content.lower()

    def test_edges_csv_has_header(self, simple_graph, temp_dir):
        """Test that edges CSV has correct header."""
        exporter = CSVExporter()
        exporter.export(simple_graph, temp_dir)
        edges_file = temp_dir / "edges.csv"
        content = edges_file.read_text()
        assert "source" in content.lower() or ":start_id" in content.lower()
        assert "target" in content.lower() or ":end_id" in content.lower()
        assert "label" in content.lower() or ":type" in content.lower()

    def test_export_empty_graph(self, temp_dir):
        """Test exporting an empty graph."""
        import networkx as nx

        empty_graph = nx.DiGraph()
        exporter = CSVExporter()

        # Expect ValueError for empty graph
        with pytest.raises(ValueError, match="Cannot export empty graph"):
            exporter.export(empty_graph, temp_dir)


class TestCypherExporter:
    """Tests for Cypher exporter."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to Cypher."""
        exporter = CypherExporter()
        cypher_file = temp_dir / "graph.cypher"
        exporter.export(simple_graph, cypher_file)
        assert cypher_file.exists()

    def test_cypher_has_create_statements(self, simple_graph, temp_dir):
        """Test that Cypher file has CREATE statements."""
        exporter = CypherExporter()
        cypher_file = temp_dir / "graph.cypher"
        exporter.export(simple_graph, cypher_file)
        content = cypher_file.read_text()
        assert "CREATE" in content

    def test_cypher_has_match_statements(self, simple_graph, temp_dir):
        """Test that Cypher file has MATCH statements for edges."""
        exporter = CypherExporter()
        cypher_file = temp_dir / "graph.cypher"
        exporter.export(simple_graph, cypher_file)
        content = cypher_file.read_text()
        assert "MATCH" in content


class TestJSONExporter:
    """Tests for JSON exporter."""

    def test_export_simple_graph(self, simple_graph, temp_dir):
        """Test exporting a simple graph to JSON."""
        exporter = JSONExporter()
        json_file = temp_dir / "graph.json"
        exporter.export(simple_graph, json_file)
        assert json_file.exists()

    def test_json_structure(self, simple_graph, temp_dir):
        """Test that JSON has correct structure."""
        import json

        exporter = JSONExporter()
        json_file = temp_dir / "graph.json"
        exporter.export(simple_graph, json_file)
        with open(json_file) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data
