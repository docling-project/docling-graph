"""
Tests for ReportGenerator class.
"""

from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.converters.models import GraphMetadata
from docling_graph.core.visualizers.report_generator import ReportGenerator


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = nx.DiGraph()
    graph.add_node("node_1", label="Person", name="John")
    graph.add_node("node_2", label="Company", name="ACME")
    graph.add_edge("node_1", "node_2", label="works_for")
    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return nx.DiGraph()


class TestReportGeneratorValidation:
    """Test graph validation."""

    def test_validate_graph_with_nodes(self, sample_graph):
        """Should return True for non-empty graph."""
        generator = ReportGenerator()
        assert generator.validate_graph(sample_graph) is True

    def test_validate_graph_empty(self, empty_graph):
        """Should return False for empty graph."""
        generator = ReportGenerator()
        assert generator.validate_graph(empty_graph) is False


class TestReportGeneratorReportSections:
    """Test report section generation."""

    def test_create_header(self):
        """Should create valid header section."""
        header = ReportGenerator._create_header()

        assert isinstance(header, str)
        assert "Knowledge Graph Report" in header
        assert "#" in header  # Markdown header

    def test_create_overview(self):
        """Should create overview section."""
        metadata = GraphMetadata(node_count=10, edge_count=15, source_models=2)

        overview = ReportGenerator._create_overview(metadata)

        assert "10" in overview  # Node count
        assert "15" in overview  # Edge count
        assert "2" in overview  # Source models

    def test_create_node_type_distribution(self):
        """Should create node type distribution section."""
        metadata = GraphMetadata(
            node_count=10, edge_count=15, source_models=1, node_types={"Person": 6, "Company": 4}
        )

        dist = ReportGenerator._create_node_type_distribution(metadata)

        assert "Node Type" in dist
        assert "Person" in dist
        assert "6" in dist

    def test_create_edge_type_distribution(self):
        """Should create edge type distribution section."""
        metadata = GraphMetadata(
            node_count=10,
            edge_count=15,
            source_models=1,
            edge_types={"works_for": 8, "located_in": 7},
        )

        dist = ReportGenerator._create_edge_type_distribution(metadata)

        assert "Edge Type" in dist
        assert "works_for" in dist
        assert "8" in dist

    def test_create_sample_nodes(self, sample_graph):
        """Should create sample nodes section."""
        samples = ReportGenerator._create_sample_nodes(sample_graph, max_samples=5)

        assert "Sample Nodes" in samples
        assert "node_" in samples

    def test_create_sample_edges(self, sample_graph):
        """Should create sample edges section."""
        samples = ReportGenerator._create_sample_edges(sample_graph, max_samples=5)

        assert "Sample Edges" in samples
        assert "works_for" in samples

    def test_create_extraction_diagnostics(self):
        """Should create extraction diagnostics section when provided."""
        section = ReportGenerator._create_extraction_diagnostics(
            extraction_contract="staged",
            staged_passes_count=5,
        )
        assert "## Extraction Diagnostics" in section
        assert "staged" in section
        assert "5" in section

    def test_create_extraction_diagnostics_empty(self):
        """Should create placeholder when no diagnostics."""
        section = ReportGenerator._create_extraction_diagnostics()
        assert "## Extraction Diagnostics" in section
        assert "No extraction diagnostics" in section

    def test_create_extraction_diagnostics_with_structured_fallback_metadata(self):
        section = ReportGenerator._create_extraction_diagnostics(
            extraction_contract="direct",
            llm_diagnostics={
                "structured_attempted": True,
                "structured_failed": True,
                "fallback_used": True,
                "fallback_error_class": "ClientError",
            },
        )
        assert "Structured attempted" in section
        assert "Structured failed" in section
        assert "Legacy fallback used" in section
        assert "ClientError" in section


class TestReportGeneratorOutput:
    """Test report file generation."""

    def test_visualize_creates_file(self, sample_graph, tmp_path):
        """Should create markdown file."""
        generator = ReportGenerator()
        output_path = tmp_path / "report.md"

        generator.visualize(sample_graph, output_path, source_model_count=1)

        assert output_path.exists()
        assert output_path.suffix == ".md"

    def test_visualize_adds_md_extension(self, sample_graph, tmp_path):
        """Should add .md extension if missing."""
        generator = ReportGenerator()
        output_path = tmp_path / "report"  # No extension

        generator.visualize(sample_graph, output_path, source_model_count=1)

        assert (tmp_path / "report.md").exists()

    def test_visualize_creates_valid_markdown(self, sample_graph, tmp_path):
        """Generated file should be valid markdown."""
        generator = ReportGenerator()
        output_path = tmp_path / "report.md"

        generator.visualize(sample_graph, output_path, source_model_count=1)

        content = output_path.read_text()
        assert "# Knowledge Graph Report" in content
        assert "## Overview" in content
        assert "## Node Type Distribution" in content

    def test_visualize_empty_graph_raises_error(self, empty_graph, tmp_path):
        """Should raise error for empty graph."""
        generator = ReportGenerator()
        output_path = tmp_path / "report.md"

        with pytest.raises(ValueError):
            generator.visualize(empty_graph, output_path)

    def test_visualize_without_samples(self, sample_graph, tmp_path):
        """Should omit samples when disabled."""
        generator = ReportGenerator()
        output_path = tmp_path / "report.md"

        generator.visualize(sample_graph, output_path, source_model_count=1, include_samples=False)

        content = output_path.read_text()
        assert "Sample Nodes" not in content
        assert "Sample Edges" not in content

    def test_visualize_includes_extraction_diagnostics_when_provided(self, sample_graph, tmp_path):
        """Should include extraction diagnostics section when params passed."""
        generator = ReportGenerator()
        output_path = tmp_path / "report.md"

        generator.visualize(
            sample_graph,
            output_path,
            source_model_count=1,
            extraction_contract="staged",
            staged_passes_count=4,
        )

        content = output_path.read_text()
        assert "## Extraction Diagnostics" in content
        assert "staged" in content
        assert "4" in content
