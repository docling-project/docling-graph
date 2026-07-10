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
def test_gleaning_and_dedupe_passed_to_config(mock_load_config, mock_run_pipeline):
    """--gleaning and --dense-dedupe are passed to PipelineConfig."""
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
                gleaning_enabled=True,
                dense_dedupe="aggressive",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.gleaning_enabled is True
    assert cfg.dense_dedupe == "aggressive"


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
                export_docling_json=False,
                export_markdown=False,
                export_doclang=False,
                export_per_page=True,
                llm_input_format="doclang-geo",
            )
        except typer.Exit:
            pass
    mock_run_pipeline.assert_called_once()
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.chunk_max_tokens == 256
    assert cfg.dense_dedupe == "standard"
    assert cfg.export_docling_json is False
    assert cfg.export_markdown is False
    assert cfg.export_doclang is False
    assert cfg.export_per_page_markdown is True
    assert cfg.llm_input_format == "doclang-geo"


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_no_gleaning_flag_overrides_config_default(mock_load_config, mock_run_pipeline):
    """--no-gleaning must win over a truthy config-file default."""
    config = _base_config()
    config["defaults"]["gleaning_enabled"] = True
    mock_load_config.return_value = config
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                gleaning_enabled=False,
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.gleaning_enabled is False


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_dense_sizing_flags_passed_to_config(mock_load_config, mock_run_pipeline):
    """Dense sizing/context flags map 1:1 onto PipelineConfig fields."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                dense_skeleton_batch_tokens=2048,
                dense_fill_nodes_cap=8,
                dense_fill_context="full",
                output_dir=Path("out"),
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.dense_skeleton_batch_tokens == 2048
    assert cfg.dense_fill_nodes_cap == 8
    assert cfg.dense_fill_context == "full"


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


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_input_type_detector_exception_continues_execution(
    mock_load_config, mock_run_pipeline, caplog
):
    """Verify exception in InputTypeDetector doesn't break pipeline execution.

    When InputTypeDetector.detect() raises an exception, the code should:
    1. Catch the exception
    2. Set input_type_display to "Unknown"
    3. Continue execution without raising the exception
    4. Successfully complete pipeline execution
    """
    import logging

    mock_load_config.return_value = _base_config()

    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        # Make InputTypeDetector.detect() raise an exception
        mock_detector.detect.side_effect = RuntimeError("Detection failed")

        # Call convert_command - should NOT raise an exception
        with caplog.at_level(logging.INFO, logger="docling_graph"):
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

    # Verify "Unknown" was logged in the configuration summary
    input_type_lines = [r.getMessage() for r in caplog.records if "input_type=" in r.getMessage()]
    assert len(input_type_lines) > 0, "Input type should be displayed"
    assert "input_type=Unknown" in input_type_lines[0], (
        "Input type should be 'Unknown' when detection fails"
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


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_invalid_enum_values_fall_back_to_defaults(mock_load_config, mock_run_pipeline):
    """Unrecognized enum-like flags fall back to their safe defaults instead of
    propagating (covers the four validation fallbacks in _resolve_cli_settings,
    incl. llm_input_format -> 'auto', the flipped default this branch introduced)."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
                llm_input_format="nonsense",
                dense_dedupe="nonsense",
                dense_fill_context="nonsense",
                provenance="nonsense",
            )
        except typer.Exit:
            pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.llm_input_format == "auto"
    assert cfg.dense_dedupe == "standard"
    assert cfg.dense_fill_context == "scoped"
    assert cfg.provenance == "standard"


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_dense_contract_logs_dense_tuning(mock_load_config, mock_run_pipeline, caplog):
    """A dense/auto contract emits the DenseTuning summary line (dense-only block)."""
    import logging

    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        with caplog.at_level(logging.INFO, logger="docling_graph"):
            try:
                convert_command(
                    source="doc.pdf",
                    template="templates.Foo",
                    output_dir=Path("out"),
                    extraction_contract="dense",
                )
            except typer.Exit:
                pass
    cfg = mock_run_pipeline.call_args[0][0]
    assert cfg.extraction_contract == "dense"
    assert any("skeleton_batch_tokens=" in r.getMessage() for r in caplog.records)


@patch("docling_graph.cli.commands.convert.run_pipeline")
@patch("docling_graph.cli.commands.convert.load_config")
def test_llm_generation_reliability_and_toplevel_overrides_merged(
    mock_load_config, mock_run_pipeline
):
    """Every --llm-* generation/reliability/top-level override lands on the
    resolved config (covers the per-flag merge block in convert_command)."""
    mock_load_config.return_value = _base_config()
    with patch("docling_graph.core.input.types.InputTypeDetector") as mock_detector:
        mock_detector.detect.return_value = MagicMock(value="file")
        try:
            convert_command(
                source="doc.pdf",
                template="templates.Foo",
                output_dir=Path("out"),
                llm_temperature=0.3,
                llm_max_tokens=1024,
                llm_top_p=0.9,
                llm_timeout=42,
                llm_retries=7,
                llm_context_limit=16000,
                llm_max_output_tokens=2048,
                llm_streaming=True,
            )
        except typer.Exit:
            pass
    overrides = mock_run_pipeline.call_args[0][0].llm_overrides
    assert overrides.generation.temperature == 0.3
    assert overrides.generation.max_tokens == 1024
    assert overrides.generation.top_p == 0.9
    assert overrides.reliability.timeout_s == 42
    assert overrides.reliability.max_retries == 7
    assert overrides.context_limit == 16000
    assert overrides.max_output_tokens == 2048
    assert overrides.streaming is True
