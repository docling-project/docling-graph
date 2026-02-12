"""Tests for convert command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.commands.convert import convert_command


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_llm_base_url_passed_to_config(mock_load_config, mock_run_pipeline):
    """--llm-base-url is merged into llm_overrides.connection.base_url."""
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
        },
        "docling": {"pipeline": "ocr"},
        "models": {"llm": {"remote": {"provider": "openai", "model": "gpt-4o"}}},
        "llm_overrides": {},
    }
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                llm_base_url="https://onprem.example.com/v1",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.llm_overrides.connection.base_url == "https://onprem.example.com/v1"
