"""
Tests for convert CLI command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.cli.commands.convert import convert_command


class TestConvertCommand:
    """CLI convert command tests."""

    @patch("docling_graph.cli.commands.convert.load_config")
    @patch(
        "docling_graph.cli.commands.convert.run_pipeline", side_effect=Exception("Pipeline error")
    )
    def test_convert_command_pipeline_error_exits(
        self, mock_run_pipeline, mock_load_config, tmp_path
    ):
        """Should exit with error on pipeline failure."""
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
        }

        with (
            patch(
                "docling_graph.cli.commands.convert.validate_processing_mode",
                return_value="many-to-one",
            ),
            patch("docling_graph.cli.commands.convert.validate_backend_type", return_value="llm"),
            patch("docling_graph.cli.commands.convert.validate_inference", return_value="local"),
            patch("docling_graph.cli.commands.convert.validate_docling_config", return_value="ocr"),
            patch("docling_graph.cli.commands.convert.validate_export_format", return_value="csv"),
            patch("docling_graph.cli.commands.convert.validate_vlm_constraints"),
        ):
            with pytest.raises(ValueError):
                convert_command(source=doc_path, template="templates.invoice.Invoice")

        # Ensure run_pipeline was called once
        assert mock_run_pipeline.call_count == 1

    @patch("docling_graph.cli.commands.convert.load_config")
    @patch("docling_graph.cli.commands.convert.run_pipeline")
    def test_convert_command_happy_path(self, mock_run_pipeline, mock_load_config, tmp_path):
        """Should pass config to pipeline and not raise."""
        doc_path = tmp_path / "test.pdf"
        doc_path.write_text("test")

        mock_load_config.return_value = {
            "defaults": {
                "processing_mode": "one-to-one",
                "backend": "vlm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr", "export": {}},
        }

        with (
            patch(
                "docling_graph.cli.commands.convert.validate_processing_mode",
                return_value="one-to-one",
            ),
            patch("docling_graph.cli.commands.convert.validate_backend_type", return_value="vlm"),
            patch("docling_graph.cli.commands.convert.validate_inference", return_value="local"),
            patch("docling_graph.cli.commands.convert.validate_docling_config", return_value="ocr"),
            patch("docling_graph.cli.commands.convert.validate_export_format", return_value="csv"),
            patch("docling_graph.cli.commands.convert.validate_vlm_constraints"),
        ):
            # Should not raise
            convert_command(source=doc_path, template="templates.invoice.Invoice")

        assert mock_run_pipeline.call_count == 1
