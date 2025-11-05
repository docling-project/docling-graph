from unittest.mock import MagicMock, patch

import pytest

from docling_graph.core.extractors.factory import ExtractorFactory
from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
from docling_graph.core.extractors.strategies.one_to_one import OneToOneStrategy


@patch("docling_graph.core.extractors.factory.LlmBackend")
@patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
def test_create_llm_many_to_one(mock_strategy, mock_backend):
    """Test creating LLM backend with many-to-one strategy."""
    mock_llm_client = MagicMock()
    mock_llm_client.__class__.__name__ = "MockLLMClient"

    ExtractorFactory.create_extractor(
        processing_mode="many-to-one",
        backend_name="llm",
        llm_client=mock_llm_client,
        docling_config="ocr",
        use_chunking=True,
        llm_consolidation=True,
    )

    mock_backend.assert_called_once_with(llm_client=mock_llm_client)
    mock_strategy.assert_called_once()


@patch("docling_graph.core.extractors.factory.VlmBackend")
@patch("docling_graph.core.extractors.factory.ManyToOneStrategy")
def test_create_vlm_many_to_one(mock_strategy, mock_backend):
    """Test creating VLM backend with many-to-one strategy."""
    ExtractorFactory.create_extractor(
        processing_mode="many-to-one",
        backend_name="vlm",
        model_name="docling-vlm",
        docling_config="vision",
    )

    mock_backend.assert_called_once_with(model_name="docling-vlm")
    mock_strategy.assert_called_once()


@patch("docling_graph.core.extractors.factory.LlmBackend")
@patch("docling_graph.core.extractors.factory.OneToOneStrategy")
def test_create_one_to_one(mock_strategy, mock_backend):
    """Test creating one-to-one strategy."""
    mock_llm_client = MagicMock()

    ExtractorFactory.create_extractor(
        processing_mode="one-to-one",
        backend_name="llm",
        llm_client=mock_llm_client,
        docling_config="ocr",
    )

    mock_backend.assert_called_once_with(llm_client=mock_llm_client)
    mock_strategy.assert_called_once()


def test_create_vlm_without_model_name():
    """Test that VLM without model_name raises error."""
    with pytest.raises(ValueError, match="VLM requires model_name"):
        ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_name="vlm",
        )


def test_create_llm_without_client():
    """Test that LLM without llm_client raises error."""
    with pytest.raises(ValueError, match="LLM requires llm_client"):
        ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_name="llm",
        )


def test_unknown_backend():
    """Test that unknown backend raises error."""
    with pytest.raises(ValueError, match="Unknown backend"):
        ExtractorFactory.create_extractor(
            processing_mode="many-to-one",
            backend_name="unknown",
            llm_client=MagicMock(),
        )


def test_unknown_processing_mode():
    """Test that unknown processing mode raises error."""
    with pytest.raises(ValueError, match="Unknown processing_mode"):
        ExtractorFactory.create_extractor(
            processing_mode="unknown",
            backend_name="llm",
            llm_client=MagicMock(),
        )
