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

        assert backend.doc_extractor is None

    @patch("docling_graph.core.extractors.backends.vlm_backend.torch")
    @patch("docling_graph.core.extractors.backends.vlm_backend.DocumentExtractor")
    def test_cleanup_clears_cuda(self, mock_extractor_class, mock_torch):
        """Should clear CUDA cache if available."""
        mock_torch.cuda.is_available.return_value = True
        backend = VlmBackend(model_name="test-model")

        backend.cleanup()

        mock_torch.cuda.empty_cache.assert_called()


# ============================================================================
# Fix 7: Enhanced GPU Cleanup Tests
# ============================================================================


class TestEnhancedGPUCleanup:
    """Tests for Fix 7: Enhanced GPU cleanup with memory tracking."""

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_cleanup_with_gpu(self, mock_sync, mock_empty, mock_mem_alloc, mock_cuda_available):
        """Test cleanup when GPU is available."""
        mock_cuda_available.return_value = True
        mock_mem_alloc.side_effect = [1024 * 1024 * 100, 1024 * 1024 * 10]  # 100MB -> 10MB

        backend = VlmBackend(model_name="test-model")
        backend.cleanup()

        # Verify CUDA operations were called
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
        assert mock_mem_alloc.call_count == 2  # Before and after

    @patch("torch.cuda.is_available")
    def test_cleanup_without_gpu(self, mock_cuda_available):
        """Test cleanup when GPU is not available."""
        mock_cuda_available.return_value = False

        backend = VlmBackend(model_name="test-model")
        backend.cleanup()

        # Should complete without errors
        assert True

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.synchronize")
    def test_cleanup_all_gpus(
        self, mock_sync, mock_empty, mock_mem_alloc, mock_device_count, mock_cuda_available
    ):
        """Test multi-GPU cleanup."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        mock_mem_alloc.side_effect = [
            1024 * 1024 * 50,  # GPU 0 before
            1024 * 1024 * 5,  # GPU 0 after
            1024 * 1024 * 30,  # GPU 1 before
            1024 * 1024 * 3,  # GPU 1 after
        ]

        backend = VlmBackend(model_name="test-model")
        backend.cleanup_all_gpus()

        # Should clear cache for each GPU
        assert mock_empty.call_count == 2
        assert mock_sync.call_count == 2

    @patch("torch.cuda.is_available")
    def test_cleanup_error_handling(self, mock_cuda_available):
        """Test cleanup handles errors gracefully."""
        mock_cuda_available.side_effect = Exception("CUDA error")

        backend = VlmBackend(model_name="test-model")
        # Should not raise exception
        backend.cleanup()
