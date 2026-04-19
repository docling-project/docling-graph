"""Tests for convert command."""

from pathlib import Path
from typing import Any
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


def _base_config() -> dict[str, Any]:
    return {
        "defaults": {
            "backend": "llm",
            "inference": "remote",
            "processing_mode": "many-to-one",
            "extraction_contract": "direct",
            "export_format": "csv",
        },
        "docling": {
            "pipeline": "ocr",
            "export": {"docling_json": True, "markdown": True, "per_page_markdown": False},
        },
        "models": {"llm": {"remote": {"provider": "openai", "model": "gpt-4o"}}},
        "llm_overrides": {},
    }


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_cli_overrides_passed_to_config(mock_load_config, mock_run_pipeline):
    """Multiple CLI overrides are passed to PipelineConfig (312-427 branches)."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
                chunk_max_tokens=256,
                staged_tuning_preset="advanced",
                staged_pass_retries=3,
                delta_normalizer_validate_paths=False,
                delta_resolvers_mode="fuzzy",
                delta_quality_min_instances=5,
                staged_nodes_fill_cap=100,
                gleaning_max_passes=2,
                export_docling_json=False,
                export_markdown=False,
                export_per_page=True,
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.chunk_max_tokens == 256
    assert cfg.staged_tuning_preset == "advanced"
    assert cfg.staged_pass_retries == 3
    assert cfg.delta_normalizer_validate_paths is False
    assert cfg.delta_resolvers_mode == "fuzzy"
    assert cfg.delta_quality_min_instances == 5
    assert cfg.staged_nodes_fill_cap == 100
    assert cfg.gleaning_max_passes == 2
    assert cfg.export_docling_json is False
    assert cfg.export_markdown is False
    assert cfg.export_per_page_markdown is True


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_invalid_staged_tuning_preset_fallback(mock_load_config, mock_run_pipeline):
    """Invalid staged_tuning_preset from config falls back to 'standard' (324-325)."""
    mock_load_config.return_value = _base_config()
    mock_load_config.return_value["defaults"]["staged_tuning_preset"] = "custom"
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.staged_tuning_preset == "standard"


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_invalid_delta_resolvers_mode_fallback(mock_load_config, mock_run_pipeline):
    """Invalid delta_resolvers_mode falls back to 'off' (362-363)."""
    mock_load_config.return_value = _base_config()
    mock_load_config.return_value["defaults"]["delta_resolvers_mode"] = "invalid"
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.delta_resolvers_mode == "off"


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_input_type_detector_exception_shows_unknown(mock_load_config, mock_run_pipeline):
    """When InputTypeDetector.detect raises, input_type_display is 'Unknown' (449-450)."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.side_effect = ValueError("detect failed")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()


@patch("docling_graph.cli.commands.convert.rich_print")
@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_input_type_detector_exception_continues_execution(
    mock_load_config, mock_run_pipeline, mock_rich_print
):
    """Verify exception in InputTypeDetector doesn't break pipeline execution.

    When InputTypeDetector.detect() raises an exception, the code should:
    1. Catch the exception
    2. Set input_type_display to "Unknown"
    3. Continue execution without raising the exception
    4. Successfully complete pipeline execution
    """
    mock_load_config.return_value = _base_config()

    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        # Make InputTypeDetector.detect() raise an exception
        mock_detector.detect.side_effect = RuntimeError("Detection failed")

        # Call convert_command - should NOT raise an exception
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        except typer.Exit:
            # Expected exit after successful pipeline execution
            pass

    # Verify run_pipeline was called (execution continued)
    mock_run_pipeline.assert_called_once()

    # Verify "Unknown" was displayed in the configuration output
    # Find the call that displays "Input Type: Unknown"
    input_type_calls = [
        call for call in mock_rich_print.call_args_list if "Input Type:" in str(call)
    ]
    assert len(input_type_calls) > 0, "Input Type should be displayed"
    assert "Unknown" in str(input_type_calls[0]), (
        "Input Type should be 'Unknown' when detection fails"
    )


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
@patch("docling_graph.llm_clients.config.resolve_effective_model_config")
def test_show_llm_config_exits_zero(mock_resolve, mock_load_config, mock_run_pipeline):
    """show_llm_config=True with backend=llm calls resolve_effective_model_config and exits 0 (580-593)."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        with pytest.raises(typer.Exit) as exc_info:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
                show_llm_config=True,
            )
        assert exc_info.value.exit_code == 0
    mock_resolve.assert_called_once()
    mock_run_pipeline.assert_not_called()


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: __import__(
            "docling_graph.exceptions", fromlist=["ConfigurationError"]
        ).ConfigurationError("Config failed", details={"key": "value"}),
        lambda: __import__(
            "docling_graph.exceptions", fromlist=["ExtractionError"]
        ).ExtractionError("Extract failed", details={"key": "value"}),
        lambda: __import__("docling_graph.exceptions", fromlist=["PipelineError"]).PipelineError(
            "Pipeline failed", details={"key": "value"}
        ),
        lambda: __import__(
            "docling_graph.exceptions", fromlist=["DoclingGraphError"]
        ).DoclingGraphError("Graph failed", details={"key": "value"}),
    ],
)
@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_exception_handlers_with_details(mock_load_config, mock_run_pipeline, error_factory):
    """Exception handlers (602-634): run_pipeline raises with e.details, Exit(1)."""
    mock_load_config.return_value = _base_config()
    mock_run_pipeline.side_effect = error_factory()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        with pytest.raises(typer.Exit) as exc_info:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        assert exc_info.value.exit_code == 1


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_generic_exception_handler_exit_one(mock_load_config, mock_run_pipeline):
    """Generic Exception handler (634): run_pipeline raises Exception, Exit(1)."""
    mock_load_config.return_value = _base_config()
    err = RuntimeError("Unexpected")
    err.details = {"key": "value"}
    mock_run_pipeline.side_effect = err
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        with pytest.raises(typer.Exit) as exc_info:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
            )
        assert exc_info.value.exit_code == 1
