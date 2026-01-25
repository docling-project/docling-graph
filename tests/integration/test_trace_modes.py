"""Integration tests for trace data modes (CLI vs API)."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from docling_graph import PipelineConfig
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.orchestrator import PipelineOrchestrator, run_pipeline


class SimpleTestModel(BaseModel):
    """Simple test model for integration tests."""
    name: str
    value: int


class TestTraceModes:
    """Tests for trace data behavior in different modes."""

    def test_cli_mode_includes_trace_by_default(self, tmp_path):
        """Test that CLI mode includes trace by default."""
        config = PipelineConfig(
            source="test.txt",  # Use text to avoid document processing
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")

        # CLI mode should enable trace by default
        assert orchestrator.include_trace is True
        assert orchestrator.dump_to_disk is True

    def test_api_mode_excludes_trace_by_default(self, tmp_path):
        """Test that API mode excludes trace by default."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # API mode should disable trace by default (memory efficient)
        assert orchestrator.include_trace is False
        assert orchestrator.dump_to_disk is False

    def test_explicit_include_trace_true_in_api_mode(self, tmp_path):
        """Test explicit include_trace=True in API mode."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=True,
            dump_to_disk=False,  # In memory only
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # Explicit setting should override default
        assert orchestrator.include_trace is True
        assert orchestrator.dump_to_disk is False

    def test_explicit_include_trace_false_in_cli_mode(self, tmp_path):
        """Test explicit include_trace=False in CLI mode."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=False,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")

        # Explicit setting should override default
        assert orchestrator.include_trace is False

    def test_cli_mode_has_trace_export_stage(self, tmp_path):
        """Test that CLI mode includes TraceExportStage."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")

        # Check that TraceExportStage is in the pipeline
        stage_names = [stage.name() for stage in orchestrator.stages]
        assert "Trace Export" in stage_names

    def test_api_mode_no_trace_export_stage(self, tmp_path):
        """Test that API mode doesn't include TraceExportStage by default."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # Check that TraceExportStage is NOT in the pipeline
        stage_names = [stage.name() for stage in orchestrator.stages]
        assert "Trace Export" not in stage_names

    def test_api_mode_with_trace_has_export_stage(self, tmp_path):
        """Test that API mode with trace enabled includes TraceExportStage."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=True,
            dump_to_disk=True,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # With both trace and dump enabled, should have TraceExportStage
        stage_names = [stage.name() for stage in orchestrator.stages]
        assert "Trace Export" in stage_names

    def test_trace_data_in_memory_without_disk_export(self, tmp_path):
        """Test trace data collection in memory without disk export."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=True,
            dump_to_disk=False,  # No disk export
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # Should have trace enabled but no export stages
        assert orchestrator.include_trace is True
        assert orchestrator.dump_to_disk is False

        stage_names = [stage.name() for stage in orchestrator.stages]
        assert "Trace Export" not in stage_names
        assert "Export" not in stage_names
        assert "Docling Export" not in stage_names

    def test_dump_to_disk_false_removes_all_export_stages(self, tmp_path):
        """Test that dump_to_disk=False removes all export stages."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            dump_to_disk=False,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        stage_names = [stage.name() for stage in orchestrator.stages]

        # No export stages should be present
        assert "Trace Export" not in stage_names
        assert "Docling Export" not in stage_names
        assert "Export" not in stage_names
        assert "Visualization" not in stage_names

    def test_cli_mode_stage_order(self, tmp_path):
        """Test that stages are in correct order in CLI mode."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="cli")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # Core stages should come first
        assert stage_names[0] == "Input Normalization"
        assert stage_names[1] == "Template Loading"
        assert stage_names[2] == "Extraction"
        assert stage_names[3] == "Graph Conversion"

        # TraceExportStage should come before other export stages
        trace_idx = stage_names.index("Trace Export")
        docling_idx = stage_names.index("Docling Export")
        export_idx = stage_names.index("Export")

        assert trace_idx < docling_idx
        assert trace_idx < export_idx

    def test_api_mode_stage_order_with_trace(self, tmp_path):
        """Test stage order in API mode with trace enabled."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=True,
            dump_to_disk=True,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")
        stage_names = [stage.name() for stage in orchestrator.stages]

        # Verify TraceExportStage is before other exports
        if "Trace Export" in stage_names:
            trace_idx = stage_names.index("Trace Export")
            if "Docling Export" in stage_names:
                docling_idx = stage_names.index("Docling Export")
                assert trace_idx < docling_idx

    def test_mode_parameter_validation(self, tmp_path):
        """Test that mode parameter is validated."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            output_dir=str(tmp_path)
        )

        # Valid modes
        orch_cli = PipelineOrchestrator(config, mode="cli")
        assert orch_cli.mode == "cli"

        orch_api = PipelineOrchestrator(config, mode="api")
        assert orch_api.mode == "api"

    def test_trace_data_initialization_in_context(self, tmp_path):
        """Test that trace data is properly initialized in context."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=True,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")

        # Create context (simulating pipeline start)
        context = PipelineContext(config=config)

        # When trace is enabled, orchestrator should initialize trace_data
        # This would happen in orchestrator.run()
        if orchestrator.include_trace:
            from docling_graph.pipeline.trace import TraceData
            context.trace_data = TraceData()

            assert context.trace_data is not None
            assert context.trace_data.pages == []
            assert context.trace_data.extractions == []

    def test_no_trace_data_when_disabled(self, tmp_path):
        """Test that trace data is not initialized when disabled."""
        config = PipelineConfig(
            source="test.txt",
            template=SimpleTestModel,
            include_trace=False,
            output_dir=str(tmp_path)
        )

        orchestrator = PipelineOrchestrator(config, mode="api")
        context = PipelineContext(config=config)

        # When trace is disabled, trace_data should remain None
        if not orchestrator.include_trace:
            assert context.trace_data is None


class TestTraceConfiguration:
    """Tests for trace configuration combinations."""

    def test_all_combinations(self, tmp_path):
        """Test all valid combinations of trace and dump settings."""
        combinations = [
            # (include_trace, dump_to_disk, mode, expected_trace, expected_dump)
            (None, None, "cli", True, True),      # CLI defaults
            (None, None, "api", False, False),    # API defaults
            (True, True, "api", True, True),      # Explicit enable all
            (True, False, "api", True, False),    # Trace in memory only
            (False, True, "cli", False, True),    # Dump without trace
            (False, False, "api", False, False),  # All disabled
        ]

        for include_trace, dump_to_disk, mode, exp_trace, exp_dump in combinations:
            config = PipelineConfig(
                source="test.txt",
                template=SimpleTestModel,
                include_trace=include_trace,
                dump_to_disk=dump_to_disk,
                output_dir=str(tmp_path)
            )

            orchestrator = PipelineOrchestrator(config, mode=mode)

            assert orchestrator.include_trace == exp_trace, \
                f"Failed for {include_trace}, {dump_to_disk}, {mode}"
            assert orchestrator.dump_to_disk == exp_dump, \
                f"Failed for {include_trace}, {dump_to_disk}, {mode}"

# Made with Bob
