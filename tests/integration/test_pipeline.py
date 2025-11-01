"""
Integration tests for the complete pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.pipeline import run_pipeline


@pytest.mark.integration
class TestPipelineBasic:
    """Basic pipeline integration tests."""

    @patch("docling_graph.pipeline.GraphConverter")
    def test_pipeline_creates_graph(self, mock_converter, temp_dir):
        """Test that pipeline creates a graph."""
        import networkx as nx

        from docling_graph.core.base.models import GraphMetadata

        # Mock converter
        mock_graph = nx.DiGraph()
        mock_graph.add_node("test_node")
        mock_metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        mock_converter.return_value.pydantic_list_to_graph.return_value = (
            mock_graph,
            mock_metadata,
        )

        # Would need to mock other components too
        # This is a skeleton showing the structure

    def test_pipeline_with_invalid_config(self):
        """Test pipeline with invalid configuration."""
        config = {
            "source": "/nonexistent/file.pdf",
            "template": "invalid.Template",
        }

        result = run_pipeline(config)
        assert result is None


@pytest.mark.integration
class TestPipelineOutputs:
    """Test pipeline output generation."""

    @patch("docling_graph.pipeline.CSVExporter")
    @patch("docling_graph.pipeline.GraphConverter")
    def test_pipeline_exports_csv(self, mock_converter, mock_exporter, temp_dir):
        """Test that pipeline exports CSV files."""
        import networkx as nx

        # Setup mocks
        mock_graph = nx.DiGraph()
        mock_graph.add_node("node_1")

        # This test would verify CSV export is called
        # Full implementation requires mocking entire pipeline

    def test_pipeline_creates_output_directory(self, temp_dir):
        """Test that pipeline creates output directory."""
        output_dir = temp_dir / "outputs" / "nested"

        # Mock and run pipeline with this output_dir
        # Verify directory is created
        # Skeleton for now


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_flow(self, temp_dir):
        """Test complete pipeline from document to graph export."""
        # Skeleton for comprehensive test

    @pytest.mark.requires_ollama
    def test_pipeline_with_real_llm(self, temp_dir):
        """Test pipeline with real LLM (requires Ollama running)."""
        pytest.skip("Requires Ollama to be running locally")

    @pytest.mark.requires_api
    def test_pipeline_with_api_backend(self, temp_dir):
        """Test pipeline with API backend."""
        pytest.skip("Requires API credentials")


@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test error handling in pipeline."""

    def test_pipeline_handles_missing_file(self):
        """Test pipeline handles missing source file."""
        config = {"source": "/path/to/nonexistent.pdf", "template": "some.Template"}

        result = run_pipeline(config)
        assert result is None

    def test_pipeline_handles_invalid_template(self, temp_dir):
        """Test pipeline handles invalid template class."""
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"test")

        config = {"source": str(test_file), "template": "nonexistent.module.Class"}

        result = run_pipeline(config)
        assert result is None

    def test_pipeline_cleanup_on_error(self, temp_dir):
        """Test that pipeline cleans up resources on error."""
        # Mock document processor with cleanup method
        # Verify cleanup is called even when error occurs


@pytest.mark.integration
class TestPipelineConfiguration:
    """Test different pipeline configurations."""

    def test_pipeline_one_to_one_mode(self):
        """Test pipeline with one-to-one processing mode."""
        # Mock and test one-to-one strategy

    def test_pipeline_many_to_one_mode(self):
        """Test pipeline with many-to-one processing mode."""
        # Mock and test many-to-one strategy

    def test_pipeline_with_vlm_backend(self):
        """Test pipeline with VLM backend."""
        # Mock VLM backend

    def test_pipeline_with_llm_backend(self):
        """Test pipeline with LLM backend."""
        # Mock LLM backend
