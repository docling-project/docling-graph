"""Unit tests for OutputDirectoryManager."""

from datetime import datetime
from pathlib import Path

import pytest

from docling_graph.core.utils.output_manager import (
    OutputDirectoryManager,
    sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_sanitize_simple_filename(self):
        """Test sanitizing a simple filename."""
        result = sanitize_filename("document.pdf")
        assert result.startswith("document_pdf_")
        assert len(result) <= 197  # 180 + 17 for timestamp

    def test_sanitize_filename_with_spaces(self):
        """Test sanitizing filename with spaces."""
        result = sanitize_filename("My Document.pdf")
        assert result.startswith("My_Document_pdf_")
        assert " " not in result

    def test_sanitize_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        result = sanitize_filename("My Document (2024).pdf")
        assert result.startswith("My_Document__2024__pdf_")
        assert "(" not in result
        assert ")" not in result

    def test_sanitize_long_filename(self):
        """Test sanitizing very long filename."""
        long_name = "a" * 200 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 197  # Max length with timestamp
        assert result.endswith(".pdf") is False  # Extension removed in sanitization

    def test_sanitize_filename_with_dots(self):
        """Test sanitizing filename with multiple dots."""
        result = sanitize_filename("file.name.with.dots.pdf")
        assert result.startswith("file_name_with_dots_pdf_")

    def test_sanitize_filename_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        result = sanitize_filename("File123.pdf")
        assert "File123" in result
        assert result.startswith("File123_pdf_")


class TestOutputDirectoryManager:
    """Tests for OutputDirectoryManager class."""

    def test_output_manager_creation(self, tmp_path):
        """Test basic OutputDirectoryManager creation."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")

        assert manager.base_output_dir == tmp_path
        assert manager.source_filename == "test.pdf"
        assert manager.get_document_dir().exists()

    def test_output_manager_creates_document_dir(self, tmp_path):
        """Test that document directory is created."""
        manager = OutputDirectoryManager(tmp_path, "document.pdf")
        doc_dir = manager.get_document_dir()

        assert doc_dir.exists()
        assert doc_dir.is_dir()
        assert doc_dir.parent == tmp_path

    def test_get_docling_dir(self, tmp_path):
        """Test get_docling_dir method."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")
        docling_dir = manager.get_docling_dir()

        assert docling_dir.exists()
        assert docling_dir.name == "docling"
        assert docling_dir.parent == manager.get_document_dir()

    def test_get_consolidated_graph_dir(self, tmp_path):
        """Test get_consolidated_graph_dir method."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")
        consolidated_dir = manager.get_consolidated_graph_dir()

        assert consolidated_dir.exists()
        assert consolidated_dir.name == "consolidated_graph"
        assert consolidated_dir.parent == manager.get_document_dir()

    def test_get_trace_dir(self, tmp_path):
        """Test get_trace_dir method."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")
        trace_dir = manager.get_trace_dir()

        assert trace_dir.exists()
        assert trace_dir.name == "trace"
        assert trace_dir.parent == manager.get_document_dir()

    def test_get_per_page_dir(self, tmp_path):
        """Test get_per_page_dir method."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")
        per_page_dir = manager.get_per_page_dir()

        assert per_page_dir.exists()
        assert per_page_dir.name == "per_page"
        assert per_page_dir.parent == manager.get_trace_dir()

    def test_get_per_chunk_dir(self, tmp_path):
        """Test get_per_chunk_dir method."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")
        per_chunk_dir = manager.get_per_chunk_dir()

        assert per_chunk_dir.exists()
        assert per_chunk_dir.name == "per_chunk"
        assert per_chunk_dir.parent == manager.get_trace_dir()

    def test_directory_structure(self, tmp_path):
        """Test complete directory structure."""
        manager = OutputDirectoryManager(tmp_path, "document.pdf")

        # Get all directories
        doc_dir = manager.get_document_dir()
        docling_dir = manager.get_docling_dir()
        consolidated_dir = manager.get_consolidated_graph_dir()
        trace_dir = manager.get_trace_dir()
        per_page_dir = manager.get_per_page_dir()
        per_chunk_dir = manager.get_per_chunk_dir()

        # Verify structure
        assert doc_dir.exists()
        assert docling_dir.exists()
        assert consolidated_dir.exists()
        assert trace_dir.exists()
        assert per_page_dir.exists()
        assert per_chunk_dir.exists()

        # Verify hierarchy
        assert docling_dir.parent == doc_dir
        assert consolidated_dir.parent == doc_dir
        assert trace_dir.parent == doc_dir
        assert per_page_dir.parent == trace_dir
        assert per_chunk_dir.parent == trace_dir

    def test_multiple_managers_same_base(self, tmp_path):
        """Test multiple managers with same base directory."""
        manager1 = OutputDirectoryManager(tmp_path, "doc1.pdf")
        manager2 = OutputDirectoryManager(tmp_path, "doc2.pdf")

        dir1 = manager1.get_document_dir()
        dir2 = manager2.get_document_dir()

        assert dir1 != dir2
        assert dir1.parent == tmp_path
        assert dir2.parent == tmp_path

    def test_sanitized_directory_name(self, tmp_path):
        """Test that directory names are properly sanitized."""
        manager = OutputDirectoryManager(tmp_path, "My Document (2024).pdf")
        doc_dir = manager.get_document_dir()

        # Directory name should be sanitized
        assert "(" not in doc_dir.name
        assert ")" not in doc_dir.name
        assert " " not in doc_dir.name

    def test_idempotent_directory_creation(self, tmp_path):
        """Test that calling get methods multiple times is safe."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")

        # Call multiple times
        dir1 = manager.get_docling_dir()
        dir2 = manager.get_docling_dir()
        dir3 = manager.get_docling_dir()

        # Should return same directory
        assert dir1 == dir2 == dir3
        assert dir1.exists()

    def test_nested_directory_creation(self, tmp_path):
        """Test that nested directories are created properly."""
        manager = OutputDirectoryManager(tmp_path, "test.pdf")

        # Create nested structure
        per_page_dir = manager.get_per_page_dir()

        # Verify all parent directories exist
        assert per_page_dir.exists()
        assert per_page_dir.parent.exists()  # trace
        assert per_page_dir.parent.parent.exists()  # document_dir
        assert per_page_dir.parent.parent.parent == tmp_path

    def test_output_manager_with_relative_path(self, tmp_path):
        """Test OutputDirectoryManager with relative path."""
        # Change to tmp_path and use relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            manager = OutputDirectoryManager(Path("outputs"), "test.pdf")
            doc_dir = manager.get_document_dir()

            assert doc_dir.exists()
            assert "outputs" in str(doc_dir)
        finally:
            os.chdir(original_cwd)

    def test_output_manager_preserves_base_dir(self, tmp_path):
        """Test that base directory is preserved."""
        custom_base = tmp_path / "custom_outputs"
        manager = OutputDirectoryManager(custom_base, "test.pdf")

        doc_dir = manager.get_document_dir()
        assert doc_dir.parent == custom_base
        assert custom_base.exists()

# Made with Bob
