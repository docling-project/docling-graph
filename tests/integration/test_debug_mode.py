"""Integration tests for debug mode functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from docling_graph import PipelineConfig, run_pipeline
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.orchestrator import PipelineOrchestrator


class SimpleTestModel(BaseModel):
    """Simple test model for integration tests."""

    name: str
    value: int


class TestDebugMode:
    """Tests for debug mode behavior."""

    def test_debug_disabled_by_default(self, tmp_path):
        """Test that debug is disabled by default."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path),
        )

        assert config.debug is False

    def test_debug_enabled_explicitly(self, tmp_path):
        """Test that debug can be enabled explicitly."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            output_dir=str(tmp_path),
        )

        assert config.debug is True

    def test_cli_mode_with_debug_enabled(self, tmp_path):
        """Test CLI mode with debug enabled."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")

        # CLI mode should dump to disk by default
        assert orchestrator.dump_to_disk is True
        assert config.debug is True

    def test_api_mode_with_debug_enabled(self, tmp_path):
        """Test API mode with debug enabled."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # API mode should not dump to disk by default
        assert orchestrator.dump_to_disk is False
        assert config.debug is True

    def test_debug_with_dump_to_disk_true(self, tmp_path):
        """Test debug mode with explicit dump_to_disk=True."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            dump_to_disk=True,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # Explicit dump_to_disk should override mode default
        assert orchestrator.dump_to_disk is True
        assert config.debug is True

    def test_debug_with_dump_to_disk_false(self, tmp_path):
        """Test debug mode with explicit dump_to_disk=False."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            dump_to_disk=False,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")

        # Explicit dump_to_disk should override mode default
        assert orchestrator.dump_to_disk is False
        assert config.debug is True

    def test_cli_mode_stage_composition(self, tmp_path):
        """Test that CLI mode includes export stages."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # Core stages should be present
        assert "Input Normalization" in stage_names
        assert "Template Loading" in stage_names
        assert "Extraction" in stage_names
        assert "Graph Conversion" in stage_names

        # Export stages should be present in CLI mode
        assert "Docling Export" in stage_names
        assert "Export" in stage_names
        assert "Visualization" in stage_names

    def test_api_mode_stage_composition(self, tmp_path):
        """Test that API mode excludes export stages by default."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="api")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # Core stages should be present
        assert "Input Normalization" in stage_names
        assert "Template Loading" in stage_names
        assert "Extraction" in stage_names
        assert "Graph Conversion" in stage_names

        # Export stages should NOT be present in API mode by default
        assert "Docling Export" not in stage_names
        assert "Export" not in stage_names
        assert "Visualization" not in stage_names

    def test_api_mode_with_dump_to_disk_has_export_stages(self, tmp_path):
        """Test that API mode with dump_to_disk=True includes export stages."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            dump_to_disk=True,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="api")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # With dump_to_disk enabled, should have export stages
        assert "Docling Export" in stage_names
        assert "Export" in stage_names
        assert "Visualization" in stage_names

    def test_dump_to_disk_false_removes_all_export_stages(self, tmp_path):
        """Test that dump_to_disk=False removes all export stages."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            dump_to_disk=False,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="api")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # No export stages should be present
        assert "Docling Export" not in stage_names
        assert "Export" not in stage_names
        assert "Visualization" not in stage_names

    def test_cli_mode_stage_order(self, tmp_path):
        """Test that stages are in correct order in CLI mode."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path),
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # Core stages should come first
        assert stage_names[0] == "Input Normalization"
        assert stage_names[1] == "Template Loading"
        assert stage_names[2] == "Extraction"
        assert stage_names[3] == "Graph Conversion"

        # Export stages should come after core stages
        docling_idx = stage_names.index("Docling Export")
        export_idx = stage_names.index("Export")
        viz_idx = stage_names.index("Visualization")

        assert docling_idx > 3
        assert export_idx > 3
        assert viz_idx > 3

    def test_mode_parameter_validation(self, tmp_path):
        """Test that mode parameter is validated."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path),
        )

        # Valid modes
        orch_cli = PipelineOrchestrator(config, mode="cli")
        assert orch_cli.mode == "cli"

        orch_api = PipelineOrchestrator(config, mode="api")
        assert orch_api.mode == "api"


class TestDebugConfiguration:
    """Tests for debug configuration combinations."""

    def test_all_combinations(self, tmp_path):
        """Test all valid combinations of debug and dump settings."""
        combinations = [
            # (debug, dump_to_disk, mode, expected_dump)
            (False, None, "cli", True),  # CLI defaults to dump
            (False, None, "api", False),  # API defaults to no dump
            (True, None, "cli", True),  # Debug + CLI defaults to dump
            (True, None, "api", False),  # Debug + API defaults to no dump
            (True, True, "api", True),  # Explicit enable all
            (True, False, "api", False),  # Debug but no dump
            (False, True, "cli", True),  # Dump without debug
            (False, False, "api", False),  # All disabled
        ]

        for debug, dump_to_disk, mode, exp_dump in combinations:
            config = PipelineConfig(
                source="test.txt",
                template=SimpleTestModel,
                debug=debug,
                dump_to_disk=dump_to_disk,
                output_dir=str(tmp_path),
            )

            orchestrator = PipelineOrchestrator(config, mode=mode)

            assert config.debug == debug, (
                f"Failed for debug={debug}, dump={dump_to_disk}, mode={mode}"
            )
            assert orchestrator.dump_to_disk == exp_dump, (
                f"Failed for debug={debug}, dump={dump_to_disk}, mode={mode}"
            )

    def test_debug_flag_in_config_dict(self, tmp_path):
        """Test that debug flag is included in config dict."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=True,
            output_dir=str(tmp_path),
        )

        config_dict = config.to_dict()
        assert "debug" in config_dict
        assert config_dict["debug"] is True

    def test_debug_false_in_config_dict(self, tmp_path):
        """Test that debug=False is included in config dict."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            debug=False,
            output_dir=str(tmp_path),
        )

        config_dict = config.to_dict()
        assert "debug" in config_dict
        assert config_dict["debug"] is False


class TestTraceDataWithDebug:
    """Tests for trace_data population and export with debug flag."""

    @patch("docling_graph.core.converters.graph_converter.validate_graph_structure")
    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_trace_data_none_when_debug_false(self, mock_extract, mock_validate, tmp_path):
        """When debug=False, context.trace_data is None after pipeline run."""
        mock_validate.return_value = True
        source_file = tmp_path / "doc.md"
        source_file.write_text("name: Test\nvalue: 42")
        mock_extract.return_value = SimpleTestModel(name="Test", value=42)

        config = PipelineConfig(
            source=str(source_file),
            template=SimpleTestModel,
            backend="llm",
            processing_mode="many-to-one",
            debug=False,
            dump_to_disk=False,
            output_dir=str(tmp_path),
        )

        context = run_pipeline(config, mode="api")
        assert context.trace_data is None

    @patch("docling_graph.core.converters.graph_converter.validate_graph_structure")
    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_trace_data_populated_when_debug_true_api(self, mock_extract, mock_validate, tmp_path):
        """When debug=True and API mode, context.trace_data is populated."""
        mock_validate.return_value = True
        source_file = tmp_path / "doc.md"
        source_file.write_text("name: Test\nvalue: 42")
        mock_extract.return_value = SimpleTestModel(name="Test", value=42)

        config = PipelineConfig(
            source=str(source_file),
            template=SimpleTestModel,
            backend="llm",
            processing_mode="many-to-one",
            debug=True,
            dump_to_disk=False,
            output_dir=str(tmp_path),
        )

        context = run_pipeline(config, mode="api")
        assert context.trace_data is not None
        extraction_events = context.trace_data.find_events("extraction_completed")
        assert len(extraction_events) >= 1
        assert extraction_events[0].payload.get("parsed_model") is not None

    @patch("docling_graph.core.converters.graph_converter.validate_graph_structure")
    @patch("docling_graph.core.extractors.backends.llm_backend.LlmBackend.extract_from_markdown")
    def test_trace_data_json_exported_when_debug_and_dump(
        self, mock_extract, mock_validate, tmp_path
    ):
        """When debug=True and dump_to_disk=True, debug/trace_data.json is written."""
        mock_validate.return_value = True
        source_file = tmp_path / "doc.md"
        source_file.write_text("name: Test\nvalue: 42")
        mock_extract.return_value = SimpleTestModel(name="Test", value=42)

        config = PipelineConfig(
            source=str(source_file),
            template=SimpleTestModel,
            backend="llm",
            processing_mode="many-to-one",
            debug=True,
            dump_to_disk=True,
            output_dir=str(tmp_path),
        )

        run_pipeline(config, mode="api")
        trace_files = list(tmp_path.rglob("trace_data.json"))
        assert len(trace_files) >= 1
        assert trace_files[0].parent.name == "debug"
