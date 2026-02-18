"""
Tests for CSV exporter.
"""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from docling_graph.core.converters.config import ExportConfig
from docling_graph.core.exporters.csv_exporter import CSVExporter


@pytest.fixture
def sample_graph():
    """Create a sample graph."""
    graph = nx.DiGraph()
    graph.add_node("n1", label="Person", name="John")
    graph.add_node("n2", label="Company", name="ACME")
    graph.add_edge("n1", "n2", label="works_for", strength=0.9)
    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return nx.DiGraph()


class TestCSVExporterInitialization:
    """Test CSVExporter initialization."""

    def test_initialization_default(self):
        """Should initialize with default config."""
        exporter = CSVExporter()
        assert exporter.config is not None

    def test_initialization_custom_config(self):
        """Should accept custom config."""
        config = ExportConfig()
        exporter = CSVExporter(config=config)
        assert exporter.config is config


class TestCSVExporterValidation:
    """Test graph validation."""

    def test_validate_graph_with_nodes(self, sample_graph):
        """Should return True for non-empty graph."""
        exporter = CSVExporter()
        assert exporter.validate_graph(sample_graph) is True

    def test_validate_graph_empty(self, empty_graph):
        """Should return False for empty graph."""
        exporter = CSVExporter()
        assert exporter.validate_graph(empty_graph) is False


class TestCSVExporterExport:
    """Test CSV export functionality."""

    def test_export_creates_files(self, sample_graph, tmp_path):
        """Should create CSV files."""
        exporter = CSVExporter()
        output_dir = tmp_path / "csv_output"

        exporter.export(sample_graph, output_dir)

        assert output_dir.exists()
        assert (output_dir / "nodes.csv").exists()
        assert (output_dir / "edges.csv").exists()

    def test_export_empty_graph_raises_error(self, empty_graph, tmp_path):
        """Should raise error for empty graph."""
        exporter = CSVExporter()

        with pytest.raises(ValueError):
            exporter.export(empty_graph, tmp_path)

    def test_export_creates_parent_directories(self, sample_graph, tmp_path):
        """Should create parent directories if needed."""
        exporter = CSVExporter()
        output_dir = tmp_path / "nested" / "deep" / "output"

        exporter.export(sample_graph, output_dir)

        assert output_dir.exists()

    def test_nodes_csv_contains_data(self, sample_graph, tmp_path):
        """Nodes CSV should contain graph nodes."""
        exporter = CSVExporter()
        output_dir = tmp_path

        exporter.export(sample_graph, output_dir)

        nodes_df = pd.read_csv(output_dir / "nodes.csv")
        assert len(nodes_df) == 2
        assert "n1" in nodes_df["id"].to_numpy()
        assert "n2" in nodes_df["id"].to_numpy()

    def test_edges_csv_contains_data(self, sample_graph, tmp_path):
        """Edges CSV should contain graph edges."""
        exporter = CSVExporter()
        output_dir = tmp_path

        exporter.export(sample_graph, output_dir)

        edges_df = pd.read_csv(output_dir / "edges.csv")
        assert len(edges_df) == 1
        assert edges_df.iloc[0]["source"] == "n1"
        assert edges_df.iloc[0]["target"] == "n2"

    def test_nodes_csv_includes_attributes(self, sample_graph, tmp_path):
        """Nodes CSV should include node attributes."""
        exporter = CSVExporter()
        output_dir = tmp_path

        exporter.export(sample_graph, output_dir)

        nodes_df = pd.read_csv(output_dir / "nodes.csv")
        assert "label" in nodes_df.columns
        assert "name" in nodes_df.columns

    def test_edges_csv_includes_attributes(self, sample_graph, tmp_path):
        """Edges CSV should include edge attributes."""
        exporter = CSVExporter()
        output_dir = tmp_path

        exporter.export(sample_graph, output_dir)

        edges_df = pd.read_csv(output_dir / "edges.csv")
        assert "label" in edges_df.columns
        assert "strength" in edges_df.columns

    def test_export_uses_configured_filenames(self, sample_graph, tmp_path):
        """Should use configured filenames."""
        config = ExportConfig()
        exporter = CSVExporter(config=config)
        output_dir = tmp_path

        exporter.export(sample_graph, output_dir)

        assert (output_dir / config.CSV_NODE_FILENAME).exists()
        assert (output_dir / config.CSV_EDGE_FILENAME).exists()

    def test_export_handles_special_characters_in_attributes(self, tmp_path):
        """Nodes/edges with commas, quotes, or newlines in attributes export without error."""
        graph = nx.DiGraph()
        graph.add_node(
            "n1",
            label="Item",
            name='Text with "quotes" and, commas',
            note="line1\nline2",
        )
        graph.add_edge("n1", "n2", label="ref", description='Say "hello"')
        graph.add_node("n2", label="Other", name="simple")
        exporter = CSVExporter()
        output_dir = tmp_path
        exporter.export(graph, output_dir)
        nodes_df = pd.read_csv(output_dir / "nodes.csv")
        assert len(nodes_df) == 2
        assert 'Text with "quotes" and, commas' in nodes_df["name"].to_numpy()
        edges_df = pd.read_csv(output_dir / "edges.csv")
        assert 'Say "hello"' in edges_df["description"].to_numpy()
