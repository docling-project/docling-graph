"""
Shared document processing utilities.
"""

import gc
import logging
from typing import Any, List, Literal, Optional, cast, overload

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument

from ...exceptions import ExtractionError
from ...logging_utils import get_component_logger
from ..provenance.models import dclg_location_from_bbox, text_hash as _chunk_text_hash
from ..utils.doclang_format import (
    DocLangSerializerProvider,
    is_doclang_format,
    serialize_doclang,
    wants_location,
)
from .docling_serve_client import DoclingServeClient
from .document_chunker import DocumentChunker

logger = get_component_logger("DocumentProcessor", __name__)


def _chunk_doc_item_refs(chunk_obj: Any) -> list[str]:
    """Docling item self_refs for a chunk; defensive against docling API drift."""
    refs: list[str] = []
    for item in getattr(getattr(chunk_obj, "meta", None), "doc_items", None) or []:
        ref = getattr(item, "self_ref", None)
        if ref:
            refs.append(str(ref))
    return refs


def _page_sizes(document: Any) -> dict[int, tuple[float, float]]:
    """Page number -> (width, height) map; defensive against docling API drift."""
    sizes: dict[int, tuple[float, float]] = {}
    for page_no, page in (getattr(document, "pages", None) or {}).items():
        size = getattr(page, "size", None)
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
        try:
            if width is not None and height is not None:
                sizes[int(page_no)] = (float(width), float(height))
        except (TypeError, ValueError):
            continue
    return sizes


def _chunk_item_geometry(
    chunk_obj: Any, page_sizes: dict[int, tuple[float, float]] | None = None
) -> list[dict[str, Any]]:
    """Page + bbox per backing Docling item; defensive against docling API drift.

    Returns one dict per (item, provenance) as ``{"ref", "page_no",
    "bbox": [l, t, r, b], "page_width", "page_height", "dclg_location"}``.

    Boxes are always normalized to a **top-left** origin: BOTTOMLEFT sources
    (typical PDF/OCR output) are converted using the page height, and a
    BOTTOMLEFT box that can't be normalized (page height unknown) is dropped
    rather than emitted ambiguously. ``bbox`` is rounded to whole pixels for
    readability, while ``dclg_location`` (the exact ``document.dclg`` grid
    values) is quantized from the pre-rounding coordinates. Empty when the source
    format exposes no geometry (e.g. markdown/text input).
    """
    page_sizes = page_sizes or {}
    geometry: list[dict[str, Any]] = []
    for item in getattr(getattr(chunk_obj, "meta", None), "doc_items", None) or []:
        ref = getattr(item, "self_ref", None)
        for prov in getattr(item, "prov", None) or []:
            bbox = getattr(prov, "bbox", None)
            page_no = getattr(prov, "page_no", None)
            if bbox is None or page_no is None:
                continue
            try:
                page_no = int(page_no)
                size = page_sizes.get(page_no)
                coord_origin = getattr(getattr(bbox, "coord_origin", None), "value", "TOPLEFT")
                if coord_origin == "BOTTOMLEFT":
                    # Can only normalize with a known page height; otherwise the
                    # box orientation is ambiguous, so skip it.
                    converter = getattr(bbox, "to_top_left_origin", None)
                    if size is None or not callable(converter):
                        continue
                    bbox = converter(page_height=size[1])
                left, top, right, bottom = (
                    float(bbox.l),
                    float(bbox.t),
                    float(bbox.r),
                    float(bbox.b),
                )
                entry: dict[str, Any] = {
                    "ref": str(ref) if ref else "",
                    "page_no": page_no,
                    "bbox": [round(left), round(top), round(right), round(bottom)],
                    "page_width": None,
                    "page_height": None,
                    "dclg_location": None,
                }
                if size is not None:
                    entry["page_width"] = round(size[0])
                    entry["page_height"] = round(size[1])
                    entry["dclg_location"] = list(
                        dclg_location_from_bbox(left, top, right, bottom, size[0], size[1])
                    )
                geometry.append(entry)
            except (AttributeError, TypeError, ValueError):
                continue
    return geometry


def _chunk_headings(chunk_obj: Any) -> list[str]:
    """Heading trail for a chunk; defensive against docling API drift."""
    headings = getattr(getattr(chunk_obj, "meta", None), "headings", None) or []
    return [str(h) for h in headings if h]


class DocumentProcessor:
    """Handles document conversion to Markdown format and chunking."""

    def __init__(
        self,
        docling_config: str = "ocr",
        chunker_config: dict | None = None,
        llm_input_format: str = "markdown",
        docling_serve_config: dict | None = None,
    ) -> None:
        """
        Initialize document processor with specified pipeline.

        Args:
            docling_config (str): Either "vision" or "ocr" by default.
                vision: Uses VLM pipeline for complex layouts.
                ocr: Uses classic OCR pipeline for standard documents.
            chunker_config (dict): Configuration for DocumentChunker.
                Example: {
                    "tokenizer_name": "mistralai/Mistral-7B-Instruct-v0.2",
                    "chunk_max_tokens": 1024,
                    "merge_peers": True
                }
                Or use provider shortcut:
                {
                    "provider": "mistral",
                    "merge_peers": True
                }
            llm_input_format (str): Serialization sent to the LLM — 'markdown'
                (default), 'doclang', or 'doclang-geo'. DocLang formats also drive
                the chunker's serializer so chunk text matches. Note: DocLang text
                is ~25-45% larger than markdown, so raising chunk_max_tokens is
                recommended when using it (see .claude/specs/doclang).
            docling_serve_config (dict): When set (with a "base_url" key),
                document conversion is delegated to a remote docling-serve
                instance and no local conversion models are loaded. Keys:
                base_url (required), api_key, timeout. The docling_config
                pipeline selection still applies (mapped to the server's
                standard/vlm pipelines).
        """
        self.docling_config = docling_config
        self.llm_input_format = llm_input_format
        # Pristine copy kept so set_llm_input_format can rebuild the chunker
        # with format-appropriate defaults (llm-format "auto" resolves per
        # document, after this processor is constructed).
        self._chunker_base_config = dict(chunker_config) if chunker_config else None

        # Initialize chunker if config provided. In DocLang mode, hand the chunker
        # a serializer provider so chunk text is DocLang instead of markdown, and
        # default token counting to a modern BPE tokenizer (tiktoken): DocLang's
        # syntax vocabulary is designed to map efficiently onto LLM BPE tokens,
        # while wordpiece tokenizers (the MiniLM default) fragment the XML markup
        # and overcount it, skewing chunk budgets.
        self.chunker = self._build_chunker(llm_input_format)

        # Remote conversion via docling-serve: no local converter (and none of
        # its model stack) is created; the server does the conversion.
        self.serve_client: DoclingServeClient | None = None
        self.converter: DocumentConverter | None = None
        if docling_serve_config and docling_serve_config.get("base_url"):
            self.serve_client = DoclingServeClient(
                base_url=str(docling_serve_config["base_url"]),
                api_key=docling_serve_config.get("api_key"),
                timeout=float(docling_serve_config.get("timeout") or 300.0),
                docling_config=docling_config,
            )
            logger.info(
                "Initialized with remote docling-serve conversion (%s)",
                self.serve_client.base_url,
            )
        elif docling_config == "vision":
            # VLM Pipeline - Best for complex layouts and images
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                    ),
                    InputFormat.IMAGE: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                    ),
                }
            )
            logger.info("Initialized with Vision pipeline")
        else:
            # Default Pipeline - Most accurate with OCR for standard documents
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            # Note: do_cell_matching attribute removed in docling v2.60.0+
            # pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.ocr_options.lang = ["en", "fr"]
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.AUTO
            )

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: ImageFormatOption(),
                }
            )
            logger.info("Initialized with Classic OCR pipeline (English, French)")

    def _build_chunker(self, llm_input_format: str) -> "DocumentChunker | None":
        """Build the chunker with format-appropriate serializer/tokenizer defaults."""
        if not self._chunker_base_config:
            return None
        chunker_config = dict(self._chunker_base_config)
        if is_doclang_format(llm_input_format):
            chunker_config.setdefault(
                "serializer_provider",
                DocLangSerializerProvider(add_location=wants_location(llm_input_format)),
            )
            chunker_config.setdefault("tokenizer_name", "tiktoken")
        return DocumentChunker(**chunker_config)

    def set_llm_input_format(self, llm_input_format: str) -> None:
        """Switch the LLM-facing serialization after construction.

        Exists for ``llm_input_format="auto"``, which resolves per document once
        the extraction contract is known. Rebuilds the chunker so chunk text
        matches the new serialization; a no-op when the format is unchanged.
        """
        if llm_input_format == self.llm_input_format:
            return
        logger.info(
            "LLM input format: %s -> %s (chunker rebuilt to match)",
            self.llm_input_format,
            llm_input_format,
        )
        self.llm_input_format = llm_input_format
        self.chunker = self._build_chunker(llm_input_format)

    def convert_to_docling_doc(self, source: str) -> DoclingDocument:
        """
        Converts a document to Docling's Document format.

        Any format supported by Docling (PDF, Office, HTML, images, markdown, etc.)
        is accepted; Docling validates and may raise for unsupported types.

        Args:
            source (str): Path to the source document (or URL).

        Returns:
            Document: Docling document object.

        Raises:
            Exception: Re-raises Docling conversion errors with context.
        """
        # Suppress RapidOCR INFO logs (RapidOCR() resets logger to INFO in __init__; set handler level so it sticks)
        if self.docling_config == "ocr" and self.serve_client is None:
            try:
                from rapidocr.utils import log as _rapidocr_log  # type: ignore[import-untyped]

                _log = _rapidocr_log.logger
                _log.setLevel(logging.WARNING)
                for _h in _log.handlers:
                    _h.setLevel(logging.WARNING)
            except Exception:
                pass

        logger.info("Converting document: %s", source)
        if self.serve_client is not None:
            # Remote conversion; the client raises ExtractionError with context.
            document = self.serve_client.convert_to_docling_doc(source)
        else:
            assert self.converter is not None  # exactly one conversion path is configured
            try:
                result = self.converter.convert(source)
            except Exception as e:
                raise ExtractionError(
                    f"Conversion failed in Docling: {e}",
                    details={"source": source},
                    cause=e,
                ) from e
            document = result.document

        # Some formats (e.g., DOCX/MD/HTML) may not expose page metadata in Docling.
        page_count = 0
        try:
            num_pages_fn = getattr(document, "num_pages", None)
            if callable(num_pages_fn):
                page_count = int(num_pages_fn() or 0)
        except Exception:
            page_count = 0

        if page_count <= 0:
            try:
                pages = getattr(document, "pages", None)
                if pages is not None:
                    page_count = len(pages)
            except Exception:
                page_count = 0

        if page_count > 0:
            logger.info("Converted %s pages", page_count)
        else:
            logger.info("Converted document (page metadata not available for this input format)")
        return document

    @overload
    def extract_chunks(
        self, document: DoclingDocument, with_stats: Literal[True]
    ) -> tuple[List[str], dict]: ...

    @overload
    def extract_chunks(
        self, document: DoclingDocument, with_stats: Literal[False] = False
    ) -> List[str]: ...

    def extract_chunks(self, document: DoclingDocument, with_stats: bool = False) -> Any:
        """
        Extract structure-aware chunks from document using HybridChunker.

        This replaces naive text splitting with semantic chunking that preserves:
        - Tables
        - Lists
        - Section hierarchies
        - Semantic boundaries

        Args:
            document: DoclingDocument from convert_to_docling_doc()
            with_stats: If True, return (chunks, stats). If False, return just chunks.

        Returns:
            List of contextualized text chunks (or tuple with stats if with_stats=True)
        """
        if not self.chunker:
            raise ValueError(
                "Chunker not initialized. Pass chunker_config to __init__() to enable chunking."
            )

        if with_stats:
            chunks, stats = self.chunker.chunk_document_with_stats(document)
            logger.info(
                "Created %s chunks (avg: %.0f tokens, max: %s tokens)",
                stats["total_chunks"],
                stats["avg_tokens"],
                stats["max_tokens_in_chunk"],
            )
            return chunks, stats
        else:
            chunks = self.chunker.chunk_document(document)
            logger.info("Created %s structure-aware chunks", len(chunks))
            return chunks

    def extract_chunks_with_metadata(
        self, document: DoclingDocument
    ) -> tuple[List[str], List[dict]]:
        """
        Extract chunks with metadata for trace capture.

        Returns:
            Tuple of (chunks, metadata_list) where metadata_list contains:
            - chunk_id: int
            - page_numbers: list[int]
            - token_count: int
            - doc_item_refs: list[str] (docling item self_refs)
            - headings: list[str] (heading trail from chunk meta)
            - text_hash: str (hash of the chunk text, for provenance drift checks)
            - char_length: int
            - resplit_of: int | None (ordinal of the oversized parent chunk when re-split)
        """
        if not self.chunker:
            raise ValueError(
                "Chunker not initialized. Pass chunker_config to __init__() to enable chunking."
            )

        raw_chunker = self.chunker.chunker
        if raw_chunker is None:
            raise ValueError("Chunker not initialized.")

        # Build chunks and metadata in one pass so re-split chunks get one metadata each
        chunks = []
        metadata_list = []
        chunk_id = 0
        page_sizes = _page_sizes(document)
        for raw_idx, chunk_obj in enumerate(raw_chunker.chunk(document)):
            enriched_text = raw_chunker.contextualize(chunk=chunk_obj)
            enriched_tokens = self.chunker.tokenizer.count_tokens(enriched_text)
            page_numbers = sorted(
                {
                    item.prov[0].page_no
                    for item in getattr(chunk_obj.meta, "doc_items", [])
                    if hasattr(item, "prov") and item.prov
                }
            )
            doc_item_refs = _chunk_doc_item_refs(chunk_obj)
            item_geometry = _chunk_item_geometry(chunk_obj, page_sizes)
            headings = _chunk_headings(chunk_obj)
            if enriched_tokens <= self.chunker.chunk_max_tokens:
                chunks.append(enriched_text)
                metadata_list.append(
                    {
                        "chunk_id": chunk_id,
                        "page_numbers": page_numbers,
                        "token_count": enriched_tokens,
                        "doc_item_refs": doc_item_refs,
                        "item_geometry": item_geometry,
                        "headings": headings,
                        "text_hash": _chunk_text_hash(enriched_text),
                        "char_length": len(enriched_text),
                        "resplit_of": None,
                    }
                )
                chunk_id += 1
            else:
                sub_chunks = self.chunker.chunk_text_fallback(enriched_text)
                chunks.extend(sub_chunks)
                for sub in sub_chunks:
                    # Sub-chunks inherit the parent's location metadata (pages,
                    # item refs, geometry, headings) — the split is textual, not
                    # structural.
                    metadata_list.append(
                        {
                            "chunk_id": chunk_id,
                            "page_numbers": page_numbers,
                            "token_count": self.chunker.tokenizer.count_tokens(sub),
                            "doc_item_refs": doc_item_refs,
                            "item_geometry": item_geometry,
                            "headings": headings,
                            "text_hash": _chunk_text_hash(sub),
                            "char_length": len(sub),
                            "resplit_of": raw_idx,
                        }
                    )
                    chunk_id += 1

        logger.info("Extracted %s chunks with metadata", len(chunks))
        return chunks, metadata_list

    def serialize_document(self, document: DoclingDocument, page_no: int | None = None) -> str:
        """Serialize a document (or one page) in the configured LLM input format.

        Markdown is the default; DocLang formats render structure (and geometry
        for 'doclang-geo'). If DocLang serialization fails for any reason the
        method falls back to markdown so extraction is never blocked by a
        serialization glitch.
        """
        if is_doclang_format(self.llm_input_format):
            try:
                return serialize_doclang(
                    document,
                    add_location=wants_location(self.llm_input_format),
                    page_no=page_no,
                )
            except Exception as e:
                logger.warning("DocLang serialization failed (%s); falling back to markdown", e)
        if page_no is None:
            return document.export_to_markdown()
        return document.export_to_markdown(page_no=page_no)

    def extract_page_markdowns(self, document: DoclingDocument) -> List[str]:
        """
        Extract per-page document text in the configured LLM input format.

        Named for backward compatibility; the content honors ``llm_input_format``
        (markdown by default, DocLang when configured).

        Args:
            document (Document): Docling document object.

        Returns:
            List[str]: List of serialized strings, one per page.
        """
        page_texts = []
        for page_no in sorted(document.pages.keys()):
            page_texts.append(self.serialize_document(document, page_no=page_no))

        logger.info("Extracted %s for %s pages", self.llm_input_format, len(page_texts))
        return page_texts

    def process_document(self, source: str) -> List[str]:
        """High-level helper to get per-page markdowns from a source file.

        This wraps conversion and page extraction into a single call, which
        simplifies strategy code and matches the interface commonly mocked in tests.

        Args:
            source: Path to the source document.

        Returns:
            List of Markdown strings, one per page.
        """
        logger.info("Processing document into per-page markdowns")
        document = self.convert_to_docling_doc(source)
        return self.extract_page_markdowns(document)

    def process_document_with_chunking(self, source: str) -> List[str]:
        """
        Process document with structure-aware chunking instead of page-by-page.

        This is the recommended approach for LLM extraction as it:
        - Preserves tables and lists
        - Respects semantic boundaries
        - Optimizes for context window usage

        Args:
            source: Path to the source document.

        Returns:
            List of structure-aware text chunks
        """
        logger.info("Processing document with structure-aware chunking")
        document = self.convert_to_docling_doc(source)
        return self.extract_chunks(document)

    def extract_full_markdown(self, document: DoclingDocument) -> str:
        """
        Extract the full document as a single string in the configured LLM format.

        Named for backward compatibility; the content honors ``llm_input_format``
        (markdown by default, DocLang when configured).

        Args:
            document (Document): Docling document object.

        Returns:
            str: Complete document serialized for the LLM.
        """
        text = self.serialize_document(document)
        logger.info("Extracted full document %s (%s chars)", self.llm_input_format, len(text))
        return text

    def chunk_text(self, text: str) -> tuple[List[str], List[dict]]:
        """
        Chunk raw text/markdown content without DoclingDocument.

        This method chunks text-based inputs (TEXT, TEXT_FILE, MARKDOWN)
        using the DocumentChunker's fallback method which respects sentence boundaries.

        Note: This uses sentence-aware chunking, not the full HybridChunker which requires
        a DoclingDocument with structure information. For best results with markdown,
        consider converting to DoclingDocument first.

        Args:
            text: Raw text or markdown content to chunk

        Returns:
            Tuple of (chunks, metadata_list) where metadata_list contains:
            - chunk_id: int
            - page_numbers: list[int] (always [0] for text inputs)
            - token_count: int

        Raises:
            ValueError: If chunker is not initialized
        """
        if not self.chunker:
            raise ValueError(
                "Chunker not initialized. Pass chunker_config to __init__() to enable chunking."
            )

        # Use the chunker's fallback method which respects sentence boundaries
        chunks = self.chunker.chunk_text_fallback(text)

        # Build metadata for each chunk
        metadata_list: list[dict[str, Any]] = []
        for chunk_id, chunk_text in enumerate(chunks):
            token_count = self.chunker.tokenizer.count_tokens(chunk_text)
            metadata_list.append(
                {
                    "chunk_id": chunk_id,
                    "page_numbers": [0],  # Text inputs don't have pages
                    "token_count": token_count,
                    "doc_item_refs": [],
                    "item_geometry": [],  # No geometry for raw text input
                    "headings": [],
                    "text_hash": _chunk_text_hash(chunk_text),
                    "char_length": len(chunk_text),
                    "resplit_of": None,
                }
            )

        total_tokens = sum(cast(int, m["token_count"]) for m in metadata_list)
        logger.info(
            "Chunked text into %s chunks (%s total tokens, max %s per chunk)",
            len(chunks),
            total_tokens,
            self.chunker.chunk_max_tokens,
        )

        return chunks, metadata_list

    def cleanup(self) -> None:
        """Clean up document converter resources."""
        try:
            if hasattr(self, "converter"):
                del self.converter
            gc.collect()
            logger.info("Cleaned up resources")
        except Exception as e:
            logger.warning("Warning during cleanup: %s", e)
