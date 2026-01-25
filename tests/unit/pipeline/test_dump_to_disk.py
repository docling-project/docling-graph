"""Test dump_to_disk auto-detection and behavior."""

import pytest

from docling_graph import PipelineConfig
from docling_graph.pipeline.orchestrator import PipelineOrchestrator


class TestDumpToDiskAutoDetection:
    """Test auto-detection of dump_to_disk based on mode."""

    def test_api_mode_default_no_dump(self):
        """Test API mode with dump_to_disk=None defaults to False."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        assert config.dump_to_disk is None

        orchestrator = PipelineOrchestrator(config, mode="api")
        assert orchestrator.dump_to_disk is False

    def test_cli_mode_default_dump(self):
        """Test CLI mode with dump_to_disk=None defaults to True."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        assert config.dump_to_disk is None

        orchestrator = PipelineOrchestrator(config, mode="cli")
        assert orchestrator.dump_to_disk is True

    def test_api_mode_explicit_true(self):
        """Test API mode with explicit dump_to_disk=True."""
        config = PipelineConfig(source="test.pdf", template="test.Template", dump_to_disk=True)

        orchestrator = PipelineOrchestrator(config, mode="api")
        assert orchestrator.dump_to_disk is True

    def test_api_mode_explicit_false(self):
        """Test API mode with explicit dump_to_disk=False."""
        config = PipelineConfig(source="test.pdf", template="test.Template", dump_to_disk=False)

        orchestrator = PipelineOrchestrator(config, mode="api")
        assert orchestrator.dump_to_disk is False

    def test_cli_mode_explicit_false(self):
        """Test CLI mode with explicit dump_to_disk=False."""
        config = PipelineConfig(source="test.pdf", template="test.Template", dump_to_disk=False)

        orchestrator = PipelineOrchestrator(config, mode="cli")
        assert orchestrator.dump_to_disk is False


class TestStageExecution:
    """Test that stages are included/excluded based on dump_to_disk."""

    def test_api_mode_excludes_export_stages(self):
        """Test API mode excludes export stages by default."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        orchestrator = PipelineOrchestrator(config, mode="api")

        stage_names = [stage.name() for stage in orchestrator.stages]

        # Core stages should be present
        assert "Input Normalization" in stage_names
        assert "Template Loading" in stage_names
        assert "Extraction" in stage_names
        assert "Graph Conversion" in stage_names

        # Export stages should NOT be present
        assert "Docling Export" not in stage_names
        assert "Export" not in stage_names
        assert "Visualization" not in stage_names

    def test_cli_mode_includes_export_stages(self):
        """Test CLI mode includes export stages by default."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        orchestrator = PipelineOrchestrator(config, mode="cli")

        stage_names = [stage.name() for stage in orchestrator.stages]

        # All stages should be present
        assert "Input Normalization" in stage_names
        assert "Template Loading" in stage_names
        assert "Extraction" in stage_names
        assert "Graph Conversion" in stage_names
        assert "Docling Export" in stage_names
        assert "Export" in stage_names
        assert "Visualization" in stage_names

    def test_api_mode_with_dump_includes_export_stages(self):
        """Test API mode with dump_to_disk=True includes export stages."""
        config = PipelineConfig(source="test.pdf", template="test.Template", dump_to_disk=True)
        orchestrator = PipelineOrchestrator(config, mode="api")

        stage_names = [stage.name() for stage in orchestrator.stages]

        # Export stages SHOULD be present
        assert "Docling Export" in stage_names
        assert "Export" in stage_names
        assert "Visualization" in stage_names

    def test_api_mode_stage_count(self):
        """Test API mode has correct number of stages."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        orchestrator = PipelineOrchestrator(config, mode="api")

        assert len(orchestrator.stages) == 4, f"API mode should have 4 stages, got {len(orchestrator.stages)}"

    def test_cli_mode_stage_count(self):
        """Test CLI mode has correct number of stages."""
        config = PipelineConfig(source="test.pdf", template="test.Template")
        orchestrator = PipelineOrchestrator(config, mode="cli")

        assert len(orchestrator.stages) == 7, f"CLI mode should have 7 stages, got {len(orchestrator.stages)}"
