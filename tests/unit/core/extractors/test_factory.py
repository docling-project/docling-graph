"""
Tests for extractor factory.
"""

from unittest.mock import MagicMock, patch

import pytest

from docling_graph.core.extractors.factory import ExtractorFactory


class TestExtractorFactoryCreateExtractor:
    """Test extractor factory creation."""

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.OneToOneStrategy")
    def test_create_one_to_one_vlm_extractor(self, mock_strategy, mock_vlm):
        """Should create one-to-one VLM extractor."""
        mock_vlm_instance = MagicMock()
        mock_vlm.return_value = mock_vlm_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="one-to-one", backend_name="vlm", model_name="test-model"
        )

        assert extractor is not None
        mock_vlm.assert_called_once_with(model_name="test-model")
        mock_strategy.assert_called_once()

    @patch("docling_graph.core.extractors.factory.LlmBackend")
    @patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
    def test_create_many_to_one_llm_extractor(self, mock_strategy, mock_llm):
        """Should create many-to-one LLM extractor."""
        mock_client = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        extractor = ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_name="llm",
            llm_client=mock_client,
        )

        assert extractor is not None
        mock_llm.assert_called_once_with(llm_client=mock_client)
        mock_strategy.assert_called_once()

    def test_create_vlm_without_model_name_raises_error(self):
        """Should raise error if VLM created without model name."""
        with pytest.raises(ValueError):
            ExtractorFactory.create_extractor(processing_mode="one-to-one", backend_name="vlm")

    def test_create_llm_without_client_raises_error(self):
        """Should raise error if LLM created without client."""
        with pytest.raises(ValueError):
            ExtractorFactory.create_extractor(processing_mode="one-to-one", backend_name="llm")

    def test_create_unknown_backend_raises_error(self):
        """Should raise error for unknown backend type."""
        with pytest.raises(ValueError):
            ExtractorFactory.create_extractor(
                processing_mode="one-to-one",
                backend_name="unknown",  # type: ignore[arg-type]
            )

    def test_create_unknown_mode_raises_error(self):
        """Should raise error for unknown processing mode."""
        with pytest.raises(ValueError):
            ExtractorFactory.create_extractor(
                processing_mode="unknown",  # type: ignore[arg-type]
                backend_name="vlm",
                model_name="test",
            )

    @patch("docling_graph.core.extractors.factory.VlmBackend")
    @patch("docling_graph.core.extractors.factory.OneToOneStrategy")
    def test_create_with_custom_docling_config(self, mock_strategy, mock_vlm):
        """Should pass docling_config to strategy."""
        mock_vlm_instance = MagicMock()
        mock_vlm.return_value = mock_vlm_instance
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance

        ExtractorFactory.create_extractor(
            processing_mode="one-to-one",
            backend_name="vlm",
            model_name="test-model",
            docling_config="vision",
        )

        mock_strategy.assert_called_once()
        _, kwargs = mock_strategy.call_args
        assert kwargs.get("docling_config") == "vision"
