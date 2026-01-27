from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core.extractors.document_chunker import DocumentChunker


@pytest.fixture
def mock_hf_tokenizer():
    """Mock HuggingFace tokenizer."""
    tok = MagicMock()
    tok.count_tokens = MagicMock(return_value=10)
    return tok


@pytest.fixture
def mock_hf_tokenizer_wrapper():
    """Mock HuggingFaceTokenizer wrapper."""
    wrapper = MagicMock()
    wrapper.count_tokens = MagicMock(return_value=10)
    wrapper.chunk_max_tokens = 1024
    return wrapper


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunker_init_with_tokenizer_name(
    mock_hybrid_chunker, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test initialization with a specific tokenizer name."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_hf_tokenizer_class.return_value = mock_wrapper

    config = {
        "tokenizer_name": "test-model",
        "chunk_max_tokens": 1024,
        "merge_peers": False,
    }

    chunker = DocumentChunker(**config)

    mock_auto_tokenizer.from_pretrained.assert_called_with("test-model")
    mock_hybrid_chunker.assert_called_with(
        tokenizer=mock_wrapper,
        merge_peers=False,
    )

    assert chunker.tokenizer_name == "test-model"
    assert chunker.chunk_max_tokens == 1024


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunk_document(mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer):
    """Test the chunk_document method."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_hf_tokenizer_class.return_value = mock_wrapper

    # Mock the chunker instance
    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    # Mock chunk iterator
    mock_chunk1 = MagicMock()
    mock_chunk2 = MagicMock()
    mock_chunker_instance.chunk.return_value = iter([mock_chunk1, mock_chunk2])
    mock_chunker_instance.contextualize.side_effect = ["enriched_chunk1", "enriched_chunk2"]

    chunker = DocumentChunker(chunk_max_tokens=1024)
    mock_doc = MagicMock()
    chunks = chunker.chunk_document(mock_doc)

    assert chunks == ["enriched_chunk1", "enriched_chunk2"]
    mock_chunker_instance.chunk.assert_called_with(dl_doc=mock_doc)
    assert mock_chunker_instance.contextualize.call_count == 2


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_chunk_document_with_stats(
    mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test the stats-enabled chunking method."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_wrapper.count_tokens.side_effect = [150, 250]
    mock_hf_tokenizer_class.return_value = mock_wrapper

    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    mock_chunk1 = MagicMock()
    mock_chunk2 = MagicMock()
    mock_chunker_instance.chunk.return_value = iter([mock_chunk1, mock_chunk2])
    mock_chunker_instance.contextualize.side_effect = ["chunk1_text", "chunk2_text"]

    chunker = DocumentChunker(chunk_max_tokens=1024)
    mock_doc = MagicMock()

    chunks, stats = chunker.chunk_document_with_stats(mock_doc)

    assert chunks == ["chunk1_text", "chunk2_text"]
    assert stats["total_chunks"] == 2
    assert stats["total_tokens"] == 400  # 150 + 250
    assert stats["avg_tokens"] == 200.0
    assert stats["max_tokens_in_chunk"] == 250


@patch("docling_graph.core.extractors.document_chunker.AutoTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer")
@patch("docling_graph.core.extractors.document_chunker.HybridChunker")
def test_get_config_summary(
    mock_hybrid_chunker_class, mock_hf_tokenizer_class, mock_auto_tokenizer
):
    """Test the configuration summary."""
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_wrapper = MagicMock()
    mock_wrapper.__class__.__name__ = "HuggingFaceTokenizer"
    mock_hf_tokenizer_class.return_value = mock_wrapper

    mock_chunker_instance = MagicMock()
    mock_hybrid_chunker_class.return_value = mock_chunker_instance

    chunker = DocumentChunker(tokenizer_name="test-model", chunk_max_tokens=1234, merge_peers=True)

    summary = chunker.get_config_summary()

    assert summary["tokenizer_name"] == "test-model"
    assert summary["chunk_max_tokens"] == 1234
    assert summary["merge_peers"] is True
