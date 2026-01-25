"""
Tests for OutputDirectoryManager cleanup functionality.
"""

import shutil
from pathlib import Path

import pytest

from docling_graph.core.utils.output_manager import OutputDirectoryManager


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    yield output_dir
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_is_directory_empty_with_empty_dir(temp_output_dir):
    """Test is_directory_empty returns True for empty directory."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")

    # Directory should be empty initially (only subdirs created, no files)
    assert manager.is_directory_empty() is True


def test_is_directory_empty_with_files(temp_output_dir):
    """Test is_directory_empty returns False when files exist."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")

    # Create a file in the document directory
    test_file = manager.get_document_dir() / "test.txt"
    test_file.write_text("test content")

    assert manager.is_directory_empty() is False


def test_is_directory_empty_with_nested_files(temp_output_dir):
    """Test is_directory_empty detects files in subdirectories."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")

    # Create a file in a subdirectory
    docling_dir = manager.get_docling_dir()
    test_file = docling_dir / "nested_test.json"
    test_file.write_text('{"test": "data"}')

    assert manager.is_directory_empty() is False


def test_is_directory_empty_with_only_empty_subdirs(temp_output_dir):
    """Test is_directory_empty returns True with only empty subdirectories."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")

    # Create empty subdirectories
    manager.get_docling_dir()
    manager.get_docling_graph_dir()
    manager.get_debug_dir()

    # Should still be considered empty (no files)
    assert manager.is_directory_empty() is True


def test_cleanup_if_empty_removes_empty_directory(temp_output_dir):
    """Test cleanup_if_empty removes empty directory."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")
    doc_dir = manager.get_document_dir()

    # Create some empty subdirectories
    manager.get_docling_dir()
    manager.get_debug_dir()

    # Directory should exist
    assert doc_dir.exists()

    # Cleanup should remove it
    result = manager.cleanup_if_empty()
    assert result is True
    assert not doc_dir.exists()


def test_cleanup_if_empty_keeps_directory_with_files(temp_output_dir):
    """Test cleanup_if_empty keeps directory when files exist."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")
    doc_dir = manager.get_document_dir()

    # Create a file
    test_file = doc_dir / "important.json"
    test_file.write_text('{"data": "important"}')

    # Directory should exist
    assert doc_dir.exists()

    # Cleanup should NOT remove it
    result = manager.cleanup_if_empty()
    assert result is False
    assert doc_dir.exists()
    assert test_file.exists()


def test_cleanup_if_empty_keeps_directory_with_nested_files(temp_output_dir):
    """Test cleanup_if_empty keeps directory when nested files exist."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")
    doc_dir = manager.get_document_dir()

    # Create a nested file
    debug_dir = manager.get_debug_dir()
    test_file = debug_dir / "trace.json"
    test_file.write_text('{"trace": "data"}')

    # Directory should exist
    assert doc_dir.exists()

    # Cleanup should NOT remove it
    result = manager.cleanup_if_empty()
    assert result is False
    assert doc_dir.exists()
    assert test_file.exists()


def test_cleanup_if_empty_with_metadata_file(temp_output_dir):
    """Test cleanup_if_empty keeps directory when metadata exists."""
    manager = OutputDirectoryManager(temp_output_dir, "test_doc.pdf")
    doc_dir = manager.get_document_dir()

    # Save metadata (simulating partial pipeline execution)
    manager.save_metadata({"test": "metadata"})

    # Directory should exist with metadata.json
    assert doc_dir.exists()
    assert (doc_dir / "metadata.json").exists()

    # Cleanup should NOT remove it
    result = manager.cleanup_if_empty()
    assert result is False
    assert doc_dir.exists()
