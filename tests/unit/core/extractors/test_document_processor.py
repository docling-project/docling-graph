from unittest.mock import MagicMock, patch

import pytest

from docling_graph.core.extractors.document_processor import DocumentProcessor


@pytest.fixture
def mock_docling_doc():
    """Mock DoclingDocument."""
    doc = MagicMock()
    doc.pages = {1: "page1", 2: "page2"}
    doc.num_pages.return_value = 2
    doc.export_to_markdown.return_value = "Full Markdown"
    return doc


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_init_default_ocr(mock_chunker_class, mock_converter_class):
    """Test default 'ocr' initialization."""
    processor = DocumentProcessor(docling_config="ocr")

    assert processor.docling_config == "ocr"
    assert processor.chunker is None
    mock_converter_class.assert_called_once()


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_init_vision(mock_chunker_class, mock_converter_class):
    """Test 'vision' initialization."""
    processor = DocumentProcessor(docling_config="vision")

    assert processor.docling_config == "vision"
    assert processor.chunker is None
    mock_converter_class.assert_called_once()


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_init_with_chunker(mock_chunker_class, mock_converter_class):
    """Test initialization with chunker configuration."""
    chunker_config = {"max_tokens": 4096}
    processor = DocumentProcessor(chunker_config=chunker_config)

    assert processor.chunker is not None
    mock_chunker_class.assert_called_with(**chunker_config)


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
def test_convert_to_docling_doc(mock_converter_class, mock_docling_doc):
    """Test document conversion call."""
    mock_converter_instance = mock_converter_class.return_value

    # Mock the result object
    mock_result = MagicMock()
    mock_result.document = mock_docling_doc
    mock_converter_instance.convert.return_value = mock_result

    processor = DocumentProcessor()
    doc = processor.convert_to_docling_doc("source/path")

    mock_converter_instance.convert.assert_called_with("source/path")
    assert doc == mock_docling_doc


def test_extract_page_markdowns(mock_docling_doc):
    """Test extracting markdown page by page."""
    processor = DocumentProcessor()

    def export_side_effect(page_no=None) -> str:
        if page_no == 1:
            return "Page 1 MD"
        if page_no == 2:
            return "Page 2 MD"
        return "Full MD"

    mock_docling_doc.export_to_markdown.side_effect = export_side_effect

    markdowns = processor.extract_page_markdowns(mock_docling_doc)

    assert markdowns == ["Page 1 MD", "Page 2 MD"]
    mock_docling_doc.export_to_markdown.assert_any_call(page_no=1)
    mock_docling_doc.export_to_markdown.assert_any_call(page_no=2)


def test_extract_full_markdown(mock_docling_doc):
    """Test extracting the full document markdown."""
    processor = DocumentProcessor()
    md = processor.extract_full_markdown(mock_docling_doc)

    assert md == "Full Markdown"
    mock_docling_doc.export_to_markdown.assert_called_with()


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
def test_extract_chunks_no_chunker_raises_error(mock_converter_class):
    """Test that extract_chunks fails if chunker is not configured."""
    processor = DocumentProcessor()  # No chunker_config

    with pytest.raises(ValueError, match="Chunker not initialized"):
        processor.extract_chunks(MagicMock())


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_no_stats(mock_chunker_class, mock_converter_class):
    """Test extract_chunks without stats."""
    mock_chunker_instance = mock_chunker_class.return_value
    mock_chunker_instance.chunk_document.return_value = ["chunk1", "chunk2"]

    processor = DocumentProcessor(chunker_config={"max_tokens": 1024})
    chunks = processor.extract_chunks(MagicMock(), with_stats=False)

    assert chunks == ["chunk1", "chunk2"]
    mock_chunker_instance.chunk_document.assert_called_once()


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_with_stats(mock_chunker_class, mock_converter_class):
    """Test extract_chunks with stats."""
    mock_chunker_instance = mock_chunker_class.return_value
    mock_chunker_instance.chunk_document_with_stats.return_value = (
        ["chunk1", "chunk2"],
        {"total_chunks": 2, "avg_tokens": 10, "max_tokens_in_chunk": 12},
    )

    processor = DocumentProcessor(chunker_config={"max_tokens": 1024})
    chunks, stats = processor.extract_chunks(MagicMock(), with_stats=True)

    assert chunks == ["chunk1", "chunk2"]
    assert stats["total_chunks"] == 2
    mock_chunker_instance.chunk_document_with_stats.assert_called_once()


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_process_document(mock_chunker_class, mock_converter_class, mock_docling_doc):
    """Test the high-level process_document helper."""
    mock_converter_instance = mock_converter_class.return_value
    mock_result = MagicMock()
    mock_result.document = mock_docling_doc
    mock_converter_instance.convert.return_value = mock_result

    def export_side_effect(page_no=None) -> str:
        if page_no == 1:
            return "Page 1 MD"
        if page_no == 2:
            return "Page 2 MD"
        return "Full MD"

    mock_docling_doc.export_to_markdown.side_effect = export_side_effect

    processor = DocumentProcessor()
    markdowns = processor.process_document("source/path")

    mock_converter_instance.convert.assert_called_with("source/path")
    assert markdowns == ["Page 1 MD", "Page 2 MD"]
