"""
Integration tests for pipeline cleanup on failure.
"""

from pathlib import Path

import pytest

from docling_graph import PipelineConfig
from docling_graph.exceptions import PipelineError
from docling_graph.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def invalid_template_config(temp_output_dir):
    """Create a config with an invalid template to trigger failure."""
    return PipelineConfig(
        source="tests/fixtures/sample_documents/sample.jpg",
        template="nonexistent.module.Template",  # This will fail
        backend="vlm",
        inference="local",
        output_dir=str(temp_output_dir),
        dump_to_disk=True,
    )


def test_pipeline_removes_empty_directory_on_failure(invalid_template_config, temp_output_dir):
    """Test that pipeline removes empty output directory when it fails."""
    orchestrator = PipelineOrchestrator(invalid_template_config, mode="cli")

    # Pipeline should fail during template loading
    with pytest.raises(PipelineError):
        orchestrator.run()

    # Check that no empty directories were left behind
    # The output directory itself should exist, but no document subdirectories
    if temp_output_dir.exists():
        subdirs = list(temp_output_dir.iterdir())
        # Either no subdirectories, or only non-empty ones
        for subdir in subdirs:
            if subdir.is_dir():
                # If a directory exists, it should contain files
                files = list(subdir.rglob("*"))
                files = [f for f in files if f.is_file()]
                assert len(files) > 0, f"Empty directory found: {subdir}"


def test_pipeline_keeps_directory_with_partial_results(temp_output_dir, monkeypatch):
    """Test that pipeline keeps directory if any files were written before failure."""
    # This test would require a more complex setup where we:
    # 1. Start the pipeline successfully
    # 2. Write some files
    # 3. Then trigger a failure
    # For now, we'll skip this as it requires mocking internal stages
    pytest.skip("Requires complex mocking of pipeline stages")


def test_pipeline_no_cleanup_when_dump_to_disk_false(temp_output_dir):
    """Test that no cleanup is attempted when dump_to_disk is False."""
    config = PipelineConfig(
        source="tests/fixtures/sample_documents/sample.jpg",
        template="nonexistent.module.Template",
        backend="vlm",
        inference="local",
        output_dir=str(temp_output_dir),
        dump_to_disk=False,  # No disk output
    )

    orchestrator = PipelineOrchestrator(config, mode="api")

    # Pipeline should fail
    with pytest.raises(PipelineError):
        orchestrator.run()

    # No output directory should have been created at all
    # (or if it exists from fixture, it should be empty)
    if temp_output_dir.exists():
        subdirs = list(temp_output_dir.iterdir())
        assert len(subdirs) == 0, "No directories should be created when dump_to_disk=False"
