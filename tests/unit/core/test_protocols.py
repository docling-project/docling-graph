"""
Tests for protocol definitions and type checking utilities.
"""

from typing import List, Type
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from docling_graph.protocols import (
    DocumentProcessorProtocol,
    ExtractionBackendProtocol,
    ExtractorProtocol,
    LLMClientProtocol,
    TextExtractionBackendProtocol,
    get_backend_type,
    is_llm_backend,
    is_vlm_backend,
)


# Test Models
class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestExtractionBackendProtocol:
    """Test VLM backend protocol compliance."""

    def test_is_vlm_backend_with_valid_backend(self):
        """Should return True for valid VLM backend."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock(return_value=[SampleModel(name="test", value=1)])

        result = is_vlm_backend(backend)
        assert result is True

    def test_is_vlm_backend_missing_method(self):
        """Should return False if method is missing."""
        backend = MagicMock(spec=[])  # Empty spec, no methods
        result = is_vlm_backend(backend)
        assert result is False

    def test_is_vlm_backend_with_none(self):
        """Should handle None gracefully."""
        result = is_vlm_backend(None)
        assert result is False


class TestTextExtractionBackendProtocol:
    """Test LLM backend protocol compliance."""

    def test_is_llm_backend_with_valid_backend(self):
        """Should return True for valid LLM backend."""
        backend = MagicMock()
        backend.extract_from_markdown = MagicMock(return_value=SampleModel(name="test", value=1))

        result = is_llm_backend(backend)
        assert result is True

    def test_is_llm_backend_missing_method(self):
        """Should return False if method is missing."""
        backend = MagicMock(spec=[])
        result = is_llm_backend(backend)
        assert result is False

    def test_is_llm_backend_with_none(self):
        """Should handle None gracefully."""
        result = is_llm_backend(None)
        assert result is False


class TestLLMClientProtocol:
    """Test LLM client protocol."""

    def test_llm_client_has_context_limit(self):
        """Client should have context_limit property."""
        client = MagicMock()
        client.context_limit = 4096
        assert client.context_limit == 4096

    def test_llm_client_has_get_json_response(self):
        """Client should have get_json_response method."""
        client = MagicMock()
        response = {"key": "value"}
        client.get_json_response = MagicMock(return_value=response)

        result = client.get_json_response("prompt", '{"schema": "json"}')
        assert result == response


class TestExtractorProtocol:
    """Test extractor strategy protocol."""

    def test_extractor_has_backend(self):
        """Extractor should have backend attribute."""
        extractor = MagicMock()
        extractor.backend = MagicMock()
        assert extractor.backend is not None

    def test_extractor_has_extract_method(self):
        """Extractor should have extract method."""
        extractor = MagicMock()
        extractor.extract = MagicMock(return_value=[SampleModel(name="test", value=1)])

        result = extractor.extract("source.pdf", SampleModel)
        assert isinstance(result, list)


class TestDocumentProcessorProtocol:
    """Test document processor protocol."""

    def test_document_processor_has_convert_to_docling_doc(self):
        """Processor should have convert_to_docling_doc method."""
        processor = MagicMock()
        processor.convert_to_docling_doc = MagicMock(return_value=MagicMock())

        result = processor.convert_to_docling_doc("doc.pdf")
        assert result is not None

    def test_document_processor_has_extract_full_markdown(self):
        """Processor should have extract_full_markdown method."""
        processor = MagicMock()
        doc = MagicMock()
        processor.extract_full_markdown = MagicMock(return_value="# Markdown\ncontent")

        result = processor.extract_full_markdown(doc)
        assert "Markdown" in result

    def test_document_processor_has_extract_page_markdowns(self):
        """Processor should have extract_page_markdowns method."""
        processor = MagicMock()
        doc = MagicMock()
        processor.extract_page_markdowns = MagicMock(return_value=["Page 1", "Page 2"])

        result = processor.extract_page_markdowns(doc)
        assert isinstance(result, list)
        assert len(result) == 2


class TestGetBackendType:
    """Test backend type detection."""

    def test_get_backend_type_vlm(self):
        """Should return 'vlm' for VLM backend."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock()

        result = get_backend_type(backend)
        assert result == "vlm"

    def test_get_backend_type_llm(self):
        """Should return 'llm' for LLM backend."""
        backend = MagicMock()
        backend.extract_from_markdown = MagicMock()
        del backend.extract_from_document  # Remove VLM method

        result = get_backend_type(backend)
        assert result == "llm"

    def test_get_backend_type_unknown(self):
        """Should return 'unknown' for unknown backend."""
        backend = MagicMock(spec=[])
        result = get_backend_type(backend)
        assert result == "unknown"

    def test_get_backend_type_prefers_vlm(self):
        """Should prefer VLM if both methods present."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock()
        backend.extract_from_markdown = MagicMock()

        result = get_backend_type(backend)
        assert result == "vlm"  # VLM check comes first
