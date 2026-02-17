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


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_structured_output_defaults_to_true(mock_load_config, mock_run_pipeline):
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
            "structured_output": True,
        },
        "docling": {"pipeline": "ocr"},
        "models": {"llm": {"remote": {"provider": "openai", "model": "gpt-4o"}}},
        "llm_overrides": {},
    }
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(source="doc.pdf", template="templates.Foo", output_dir=Path("out"))
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.structured_output is True


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_structured_output_can_be_disabled(mock_load_config, mock_run_pipeline):
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
            "structured_output": True,
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
                schema_enforced_llm=False,
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.structured_output is False


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_structured_sparse_check_defaults_to_true(mock_load_config, mock_run_pipeline):
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
            "structured_output": True,
            "structured_sparse_check": True,
        },
        "docling": {"pipeline": "ocr"},
        "models": {"llm": {"remote": {"provider": "openai", "model": "gpt-4o"}}},
        "llm_overrides": {},
    }
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(source="doc.pdf", template="templates.Foo", output_dir=Path("out"))
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.structured_sparse_check is True


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_structured_sparse_check_can_be_disabled(mock_load_config, mock_run_pipeline):
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
            "structured_output": True,
            "structured_sparse_check": True,
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
                structured_sparse_check=False,
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.structured_sparse_check is False


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_gleaning_enabled_and_max_passes_passed_to_config(mock_load_config, mock_run_pipeline):
    """--gleaning-enabled and --gleaning-max-passes are passed to PipelineConfig."""
    mock_load_config.return_value = {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "delta",
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
                gleaning_enabled=True,
                gleaning_max_passes=2,
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.gleaning_enabled is True
    assert cfg.gleaning_max_passes == 2
