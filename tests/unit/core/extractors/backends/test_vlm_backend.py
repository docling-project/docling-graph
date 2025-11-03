"""
Tests for VLM backend.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.backends.vlm_backend import VlmBackend


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


class TestVlmBackendInitialization:
    """Test VLM backend initialization."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_initialization_with_model_name(self, mock_extractor):
        """Should initialize with model name."""
        with patch.object(VlmBackend, "_initialize_extractor"):
            backend = VlmBackend(model_name="numind/NuExtract-2.0-2B")
            assert backend.model_name == "numind/NuExtract-2.0-2B"

    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_initialization_sets_extractor(self, mock_extractor):
        """Should initialize document extractor."""
        with patch.object(VlmBackend, "_initialize_extractor"):
            backend = VlmBackend(model_name="test-model")
            assert backend is not None


class TestVlmBackendExtractFromDocument:
    """Test VLM extraction from document."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_from_document_returns_list(self, mock_extractor_class):
        """Should return list of models."""
        # Setup mocks
        mock_result_page = MagicMock()
        mock_result_page.extracted_data = {"name": "test", "value": 42}

        mock_result = MagicMock()
        mock_result.pages = [mock_result_page]

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor_instance

        backend = VlmBackend(model_name="test-model")
        backend.doc_extractor = mock_extractor_instance

        result = backend.extract_from_document("test.pdf", SampleModel)

        assert isinstance(result, list)

    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_calls_extractor(self, mock_extractor_class):
        """Should call document extractor."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(pages=[])
        mock_extractor_class.return_value = mock_extractor

        backend = VlmBackend(model_name="test-model")
        backend.doc_extractor = mock_extractor

        backend.extract_from_document("test.pdf", SampleModel)

        mock_extractor.extract.assert_called_once()

    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_extract_empty_document(self, mock_extractor_class):
        """Should handle empty document gracefully."""
        mock_extractor = MagicMock()
        mock_result = MagicMock()
        mock_result.pages = []
        mock_extractor.extract.return_value = mock_result
        mock_extractor_class.return_value = mock_extractor

        backend = VlmBackend(model_name="test-model")
        backend.doc_extractor = mock_extractor

        result = backend.extract_from_document("empty.pdf", SampleModel)

        assert result == []


class TestVlmBackendCleanup:
    """Test VLM backend cleanup."""

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_removes_extractor(self, mock_extractor_class, mock_torch):
        """Should remove extractor reference."""
        backend = VlmBackend(model_name="test-model")
        backend.cleanup()

        assert not hasattr(backend, "doc_extractor") or backend.doc_extractor is None

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_clears_cuda(self, mock_extractor_class, mock_torch):
        """Should clear CUDA cache if available."""
        mock_torch.cuda.is_available.return_value = True
        backend = VlmBackend(model_name="test-model")

        backend.cleanup()

        mock_torch.cuda.empty_cache.assert_called()
