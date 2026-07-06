from unittest.mock import MagicMock, patch

import pytest
from docling_core.types.doc import DoclingDocument

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
    chunker_config = {"chunk_max_tokens": 4096}
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


def _real_doc(text: str = "Invoice #12345") -> DoclingDocument:
    from docling_core.types.doc.base import BoundingBox, CoordOrigin
    from docling_core.types.doc.document import ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="s")
    doc.add_page(page_no=1, size={"width": 612, "height": 792})
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=10, t=10, r=100, b=30, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, len(text)),
    )
    doc.add_heading(text=text, prov=prov)
    return doc


def test_serialize_document_markdown_default():
    processor = DocumentProcessor()  # llm_input_format defaults to markdown
    out = processor.serialize_document(_real_doc())
    assert "<heading" not in out
    assert "Invoice #12345" in out


def test_serialize_document_doclang():
    processor = DocumentProcessor(llm_input_format="doclang")
    out = processor.serialize_document(_real_doc())
    assert "<heading" in out
    assert "<location" not in out


def test_serialize_document_doclang_geo_includes_location():
    processor = DocumentProcessor(llm_input_format="doclang-geo")
    out = processor.serialize_document(_real_doc())
    assert "<heading" in out
    assert "<location" in out


def test_extract_full_markdown_honors_doclang_format():
    processor = DocumentProcessor(llm_input_format="doclang")
    out = processor.extract_full_markdown(_real_doc())
    assert "<heading" in out


@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_doclang_mode_passes_serializer_provider_to_chunker(mock_chunker_class):
    """DocLang mode hands the chunker a serializer_provider; markdown does not."""
    from docling_graph.core.utils.doclang_format import DocLangSerializerProvider

    DocumentProcessor(chunker_config={"chunk_max_tokens": 512}, llm_input_format="doclang")
    kwargs = mock_chunker_class.call_args.kwargs
    assert isinstance(kwargs.get("serializer_provider"), DocLangSerializerProvider)

    mock_chunker_class.reset_mock()
    DocumentProcessor(chunker_config={"chunk_max_tokens": 512}, llm_input_format="markdown")
    assert "serializer_provider" not in mock_chunker_class.call_args.kwargs


@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_doclang_mode_defaults_to_bpe_tokenizer(mock_chunker_class):
    """DocLang chunking counts tokens with a BPE tokenizer (tiktoken) by default:
    the DocLang syntax vocabulary maps onto LLM BPE tokens, while wordpiece
    tokenizers fragment the XML and overcount it."""
    DocumentProcessor(chunker_config={"chunk_max_tokens": 512}, llm_input_format="doclang")
    assert mock_chunker_class.call_args.kwargs.get("tokenizer_name") == "tiktoken"

    # An explicit tokenizer choice is always respected.
    mock_chunker_class.reset_mock()
    DocumentProcessor(
        chunker_config={"chunk_max_tokens": 512, "tokenizer_name": "bert-base-uncased"},
        llm_input_format="doclang",
    )
    assert mock_chunker_class.call_args.kwargs.get("tokenizer_name") == "bert-base-uncased"

    # Markdown mode keeps the chunker's own default (no injection).
    mock_chunker_class.reset_mock()
    DocumentProcessor(chunker_config={"chunk_max_tokens": 512}, llm_input_format="markdown")
    assert "tokenizer_name" not in mock_chunker_class.call_args.kwargs


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

    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 1024})
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

    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 1024})
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


def _make_chunk_obj(
    page_no: int, refs: list[str], headings: list[str], with_bbox: bool = True
) -> MagicMock:
    """Build a mock docling chunk with doc_items provenance and headings."""
    chunk = MagicMock()
    items = []
    for i, ref in enumerate(refs):
        item = MagicMock()
        prov = MagicMock()
        prov.page_no = page_no
        if with_bbox:
            bbox = MagicMock()
            bbox.l, bbox.t, bbox.r, bbox.b = 10.0 + i, 20.0, 100.0, 40.0
            bbox.coord_origin.value = "TOPLEFT"
            prov.bbox = bbox
        else:
            prov.bbox = None
        item.prov = [prov]
        item.self_ref = ref
        items.append(item)
    chunk.meta.doc_items = items
    chunk.meta.headings = headings
    return chunk


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_with_metadata_provenance_fields(mock_chunker_class, mock_converter_class):
    """Chunk metadata carries doc_item_refs, headings, text_hash, char_length, resplit_of."""
    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 100})
    chunker = mock_chunker_class.return_value
    chunker.chunk_max_tokens = 100
    chunker.tokenizer.count_tokens.side_effect = lambda t: len(t.split())

    chunk_obj = _make_chunk_obj(3, ["#/texts/7", "#/tables/1"], ["Results"])
    chunker.chunker.chunk.return_value = [chunk_obj]
    chunker.chunker.contextualize.return_value = "Results small chunk text"

    chunks, meta = processor.extract_chunks_with_metadata(MagicMock())

    assert chunks == ["Results small chunk text"]
    assert len(meta) == 1
    m = meta[0]
    assert m["chunk_id"] == 0
    assert m["page_numbers"] == [3]
    assert m["doc_item_refs"] == ["#/texts/7", "#/tables/1"]
    assert m["headings"] == ["Results"]
    assert m["char_length"] == len("Results small chunk text")
    assert isinstance(m["text_hash"], str) and len(m["text_hash"]) == 16
    assert m["resplit_of"] is None
    # Geometry: one entry per (item, prov); bbox rounded to whole pixels. No page
    # size in this mock, so page dims / dclg_location are absent, and coord_origin
    # is no longer emitted (boxes are always top-left).
    assert len(m["item_geometry"]) == 2
    g0 = m["item_geometry"][0]
    assert g0["ref"] == "#/texts/7"
    assert g0["page_no"] == 3
    assert g0["bbox"] == [10, 20, 100, 40]
    assert all(isinstance(x, int) for x in g0["bbox"])
    assert "coord_origin" not in g0
    assert g0["page_width"] is None
    assert g0["dclg_location"] is None


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_geometry_converts_bottomleft_to_topleft(
    mock_chunker_class, mock_converter_class
):
    """BOTTOMLEFT boxes (PDF/OCR output) are normalized to TOPLEFT origin, rounded
    to whole pixels, with page dims and the exact DocLang 512-grid location."""
    from docling_core.types.doc.base import BoundingBox, CoordOrigin

    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 100})
    chunker = mock_chunker_class.return_value
    chunker.chunk_max_tokens = 100
    chunker.tokenizer.count_tokens.side_effect = lambda t: len(t.split())

    # Real invoice numbers: 1021x1423 page, BOTTOMLEFT box for texts/0
    chunk = MagicMock()
    item = MagicMock()
    prov = MagicMock()
    prov.page_no = 1
    prov.bbox = BoundingBox(
        l=117.7, t=1316.85, r=238.23, b=1234.37, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    item.prov = [prov]
    item.self_ref = "#/texts/0"
    chunk.meta.doc_items = [item]
    chunk.meta.headings = []
    chunker.chunker.chunk.return_value = [chunk]
    chunker.chunker.contextualize.return_value = "small text"

    document = MagicMock()
    page = MagicMock()
    page.size.width, page.size.height = 1021.0, 1423.0
    document.pages = {1: page}

    _chunks, meta = processor.extract_chunks_with_metadata(document)
    g = meta[0]["item_geometry"][0]
    assert "coord_origin" not in g
    assert g["page_width"] == 1021
    assert g["page_height"] == 1423
    # bbox is rounded, whole-pixel, top-left (top-left y = page_height - bottomleft y)
    assert all(isinstance(x, int) for x in g["bbox"])
    assert g["bbox"][0] == round(117.7)
    assert g["bbox"][1] == round(1423.0 - 1316.85)
    assert g["bbox"][3] == round(1423.0 - 1234.37)
    # dclg_location reproduces document.dclg values exactly (computed from floats).
    assert g["dclg_location"] == [59, 38, 119, 68]


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_with_metadata_no_geometry_when_bbox_missing(
    mock_chunker_class, mock_converter_class
):
    """Formats without geometry (no bbox) yield an empty item_geometry list."""
    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 100})
    chunker = mock_chunker_class.return_value
    chunker.chunk_max_tokens = 100
    chunker.tokenizer.count_tokens.side_effect = lambda t: len(t.split())

    chunk_obj = _make_chunk_obj(1, ["#/texts/1"], ["Intro"], with_bbox=False)
    chunker.chunker.chunk.return_value = [chunk_obj]
    chunker.chunker.contextualize.return_value = "small text"

    _chunks, meta = processor.extract_chunks_with_metadata(MagicMock())
    assert meta[0]["item_geometry"] == []
    assert meta[0]["doc_item_refs"] == ["#/texts/1"]


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_extract_chunks_with_metadata_resplit_inherits_location(
    mock_chunker_class, mock_converter_class
):
    """Re-split sub-chunks inherit pages/refs/headings and point at their parent."""
    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 2})
    chunker = mock_chunker_class.return_value
    chunker.chunk_max_tokens = 2
    chunker.tokenizer.count_tokens.side_effect = lambda t: len(t.split())

    chunk_obj = _make_chunk_obj(5, ["#/texts/9"], ["Methods"])
    chunker.chunker.chunk.return_value = [chunk_obj]
    chunker.chunker.contextualize.return_value = "one two three four five"
    chunker.chunk_text_fallback.return_value = ["one two", "three four"]

    chunks, meta = processor.extract_chunks_with_metadata(MagicMock())

    assert chunks == ["one two", "three four"]
    assert [m["chunk_id"] for m in meta] == [0, 1]
    for m in meta:
        assert m["page_numbers"] == [5]
        assert m["doc_item_refs"] == ["#/texts/9"]
        assert m["headings"] == ["Methods"]
        assert m["resplit_of"] == 0
        # Re-split sub-chunks inherit the parent item's geometry.
        assert len(m["item_geometry"]) == 1
        assert m["item_geometry"][0]["ref"] == "#/texts/9"
    assert meta[0]["text_hash"] != meta[1]["text_hash"]


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DocumentChunker")
def test_chunk_text_metadata_provenance_fields(mock_chunker_class, mock_converter_class):
    """Raw-text chunking emits the same metadata keys with empty location data."""
    processor = DocumentProcessor(chunker_config={"chunk_max_tokens": 100})
    chunker = mock_chunker_class.return_value
    chunker.chunk_max_tokens = 100
    chunker.tokenizer.count_tokens.side_effect = lambda t: len(t.split())
    chunker.chunk_text_fallback.return_value = ["alpha beta", "gamma"]

    chunks, meta = processor.chunk_text("alpha beta gamma")

    assert chunks == ["alpha beta", "gamma"]
    for m in meta:
        assert m["page_numbers"] == [0]
        assert m["doc_item_refs"] == []
        assert m["item_geometry"] == []
        assert m["headings"] == []
        assert m["resplit_of"] is None
        assert isinstance(m["text_hash"], str) and len(m["text_hash"]) == 16
        assert m["char_length"] > 0


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DoclingServeClient")
def test_init_with_docling_serve_skips_local_converter(mock_serve_class, mock_converter_class):
    """A docling-serve config builds the remote client and no local converter."""
    processor = DocumentProcessor(
        docling_config="vision",
        docling_serve_config={"base_url": "http://serve:5001", "api_key": "k", "timeout": 60},
    )

    mock_converter_class.assert_not_called()
    mock_serve_class.assert_called_once_with(
        base_url="http://serve:5001",
        api_key="k",
        timeout=60.0,
        docling_config="vision",
    )
    assert processor.converter is None
    assert processor.serve_client is mock_serve_class.return_value


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DoclingServeClient")
def test_convert_to_docling_doc_delegates_to_serve_client(
    mock_serve_class, mock_converter_class, mock_docling_doc
):
    """Serve mode routes conversion through the remote client."""
    mock_serve_class.return_value.convert_to_docling_doc.return_value = mock_docling_doc
    processor = DocumentProcessor(docling_serve_config={"base_url": "http://serve:5001"})

    doc = processor.convert_to_docling_doc("source/path.pdf")

    mock_serve_class.return_value.convert_to_docling_doc.assert_called_once_with("source/path.pdf")
    mock_converter_class.return_value.convert.assert_not_called()
    assert doc == mock_docling_doc


@patch("docling_graph.core.extractors.document_processor.DocumentConverter")
@patch("docling_graph.core.extractors.document_processor.DoclingServeClient")
def test_docling_serve_config_without_url_uses_local_converter(
    mock_serve_class, mock_converter_class
):
    """An empty serve config (no base_url) falls back to local conversion."""
    processor = DocumentProcessor(docling_serve_config={})

    mock_serve_class.assert_not_called()
    mock_converter_class.assert_called_once()
    assert processor.serve_client is None
