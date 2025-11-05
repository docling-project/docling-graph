"""
Tests for document processor.
"""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.core.extractors.document_processor import DocumentProcessor


@pytest.fixture
def mock_document():
    """Create mock Docling document."""
    doc = MagicMock()
    doc.pages = {1: MagicMock(), 2: MagicMock()}
    doc.num_pages.return_value = 2
    doc.export_to_markdown.return_value = "# Document\n\nContent"
    doc.export_to_markdown.side_effect = (
        lambda page_no=None: f"# Page {page_no}\n\nContent"
        if page_no
        else "# Full Document\n\nContent"
    )
    return doc


@pytest.fixture
def mock_converter():
    """Create mock DocumentConverter."""
    converter = MagicMock()
    return converter


class TestDocumentProcessorInitialization:
    """Test DocumentProcessor initialization."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_initialization_default(self, mock_converter_class):
        """Should initialize with default OCR config."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()

        assert processor.docling_config == "ocr"
        mock_converter_class.assert_called_once()

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_initialization_with_ocr_config(self, mock_converter_class):
        """Should initialize with OCR config."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor(docling_config="ocr")

        assert processor.docling_config == "ocr"

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_initialization_with_vision_config(self, mock_converter_class):
        """Should initialize with vision config."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor(docling_config="vision")

        assert processor.docling_config == "vision"

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_initialization_creates_converter(self, mock_converter_class):
        """Should create document converter."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()

        assert hasattr(processor, "converter")


class TestDocumentProcessorConvertToMarkdown:
    """Test convert_to_docling_doc method."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_convert_to_docling_doc_success(self, mock_converter_class, mock_document):
        """Should convert document to markdown."""
        mock_converter_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter_instance

        processor = DocumentProcessor()
        result = processor.convert_to_docling_doc("test.pdf")

        assert result is mock_document
        mock_converter_instance.convert.assert_called_once_with("test.pdf")

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_convert_calls_converter(self, mock_converter_class, mock_document):
        """Should call converter.convert with source path."""
        mock_converter_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter_instance

        processor = DocumentProcessor()
        processor.convert_to_docling_doc("document.pdf")

        mock_converter_instance.convert.assert_called_with("document.pdf")


class TestDocumentProcessorExtractPageMarkdowns:
    """Test extract_page_markdowns method."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_page_markdowns(self, mock_converter_class, mock_document):
        """Should extract markdown for each page."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        result = processor.extract_page_markdowns(mock_document)

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_page_markdowns_calls_export(self, mock_converter_class, mock_document):
        """Should call export_to_markdown for each page."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        processor.extract_page_markdowns(mock_document)

        assert mock_document.export_to_markdown.call_count >= 2

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_page_markdowns_empty_document(self, mock_converter_class):
        """Should handle document with no pages."""
        mock_converter_class.return_value = MagicMock()
        empty_doc = MagicMock()
        empty_doc.pages = {}

        processor = DocumentProcessor()
        result = processor.extract_page_markdowns(empty_doc)

        assert result == []

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_returns_strings(self, mock_converter_class, mock_document):
        """Should return list of strings."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        result = processor.extract_page_markdowns(mock_document)

        assert all(isinstance(md, str) for md in result)


class TestDocumentProcessorExtractFullMarkdown:
    """Test extract_full_markdown method."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_full_markdown(self, mock_converter_class, mock_document):
        """Should extract full document markdown."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        result = processor.extract_full_markdown(mock_document)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_extract_full_calls_export(self, mock_converter_class, mock_document):
        """Should call export_to_markdown without page_no."""
        mock_converter_class.return_value = MagicMock()
        mock_document.export_to_markdown.return_value = "Full content"

        processor = DocumentProcessor()
        processor.extract_full_markdown(mock_document)

        mock_document.export_to_markdown.assert_called()


class TestDocumentProcessorProcessDocument:
    """Test process_document high-level method."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_process_document(self, mock_converter_class, mock_document):
        """Should process document to per-page markdowns."""
        mock_converter_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter_instance

        processor = DocumentProcessor()
        result = processor.process_document("test.pdf")

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_process_document_calls_convert(self, mock_converter_class, mock_document):
        """Should call convert_to_docling_doc."""
        mock_converter_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter_instance

        processor = DocumentProcessor()
        processor.process_document("test.pdf")

        mock_converter_instance.convert.assert_called_once_with("test.pdf")

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    def test_process_document_returns_list_of_strings(self, mock_converter_class, mock_document):
        """Should return list of markdown strings."""
        mock_converter_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.document = mock_document
        mock_converter_instance.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter_instance

        processor = DocumentProcessor()
        result = processor.process_document("test.pdf")

        assert isinstance(result, list)
        assert all(isinstance(md, str) for md in result)


class TestDocumentProcessorCleanup:
    """Test cleanup method."""

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    @patch("docling_graph.core.extractors.document_processor.gc")
    def test_cleanup_removes_converter(self, mock_gc, mock_converter_class):
        """Should remove converter reference."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        processor.cleanup()

        assert not hasattr(processor, "converter") or processor.converter is None

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    @patch("docling_graph.core.extractors.document_processor.gc")
    def test_cleanup_calls_gc_collect(self, mock_gc, mock_converter_class):
        """Should call garbage collection."""
        mock_converter_class.return_value = MagicMock()

        processor = DocumentProcessor()
        processor.cleanup()

        mock_gc.collect.assert_called_once()

    @patch("docling_graph.core.extractors.document_processor.DocumentConverter")
    @patch("docling_graph.core.extractors.document_processor.gc")
    def test_cleanup_handles_errors(self, mock_gc, mock_converter_class):
        """Should handle cleanup errors gracefully."""
        mock_converter_class.return_value = MagicMock()
        mock_gc.collect.side_effect = Exception("GC error")

        processor = DocumentProcessor()

        # Should not raise
        processor.cleanup()
