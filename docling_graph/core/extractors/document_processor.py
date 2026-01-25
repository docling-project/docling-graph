"""
Shared document processing utilities.
"""

import gc
from typing import Any, List, Literal, Optional, overload

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument
from rich import print as rich_print

from .document_chunker import DocumentChunker


class DocumentProcessor:
    """Handles document conversion to Markdown format and chunking."""

    def __init__(
        self,
        docling_config: str = "ocr",
        chunker_config: dict | None = None,
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
                    "max_tokens": 4096,
                    "merge_peers": True
                }
                Or use provider shortcut:
                {
                    "provider": "mistral",
                    "merge_peers": True
                }
        """
        self.docling_config = docling_config

        # Initialize chunker if config provided
        self.chunker = None
        if chunker_config:
            self.chunker = DocumentChunker(**chunker_config)
            rich_print(
                f"[blue][DocumentProcessor][/blue] Initialized chunker with "
                f"configuration: {self.chunker.get_config_summary()}"
            )

        if docling_config == "vision":
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
            rich_print(
                "[blue][DocumentProcessor][/blue] Initialized with [magenta]Vision pipeline[/magenta]"
            )
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
            rich_print(
                "[blue][DocumentProcessor][/blue] Initialized with [green]Classic OCR pipeline[/green] (English, French)"
            )

    def convert_to_docling_doc(self, source: str) -> DoclingDocument:
        """
        Converts a document to Docling's Document format.

        Args:
            source (str): Path to the source document.

        Returns:
            Document: Docling document object.
        """
        rich_print(
            f"[blue][DocumentProcessor][/blue] Converting document: [yellow]{source}[/yellow]"
        )
        result = self.converter.convert(source)

        rich_print(
            f"[blue][DocumentProcessor][/blue] Converted [cyan]{result.document.num_pages()}[/cyan] pages"
        )
        return result.document

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
            rich_print(
                f"[blue][DocumentProcessor][/blue] Created [cyan]{stats['total_chunks']}[/cyan] chunks "
                f"(avg: {stats['avg_tokens']:.0f} tokens, max: {stats['max_tokens_in_chunk']} tokens)"
            )
            return chunks, stats
        else:
            chunks = self.chunker.chunk_document(document)
            rich_print(
                f"[blue][DocumentProcessor][/blue] Created [cyan]{len(chunks)}[/cyan] "
                "structure-aware chunks"
            )
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
        """
        if not self.chunker:
            raise ValueError(
                "Chunker not initialized. Pass chunker_config to __init__() to enable chunking."
            )

        chunks = self.chunker.chunk_document(document)

        # Extract metadata from chunks
        metadata_list = []
        for i, chunk_obj in enumerate(self.chunker.chunker.chunk(document)):
            # Get page numbers from chunk
            page_numbers = list(
                {
                    item.prov[0].page_no
                    for item in chunk_obj.meta.doc_items  # type: ignore[attr-defined]
                    if hasattr(item, "prov") and item.prov
                }
            )

            metadata_list.append(
                {
                    "chunk_id": i,
                    "page_numbers": sorted(page_numbers),
                    "token_count": len(self.chunker.tokenizer.encode(chunks[i])),  # type: ignore[union-attr]
                }
            )

        rich_print(
            f"[blue][DocumentProcessor][/blue] Extracted [cyan]{len(chunks)}[/cyan] chunks with metadata"
        )
        return chunks, metadata_list

    def extract_page_markdowns(self, document: DoclingDocument) -> List[str]:
        """
        Extracts Markdown content for each page.

        Args:
            document (Document): Docling document object.

        Returns:
            List[str]: List of Markdown strings, one per page.
        """
        page_markdowns = []
        for page_no in sorted(document.pages.keys()):
            md = document.export_to_markdown(page_no=page_no)
            page_markdowns.append(md)

        rich_print(
            f"[blue][DocumentProcessor][/blue] Extracted Markdown for [cyan]{len(page_markdowns)}[/cyan] pages"
        )
        return page_markdowns

    def process_document(self, source: str) -> List[str]:
        """High-level helper to get per-page markdowns from a source file.

        This wraps conversion and page extraction into a single call, which
        simplifies strategy code and matches the interface commonly mocked in tests.

        Args:
            source: Path to the source document.

        Returns:
            List of Markdown strings, one per page.
        """
        rich_print("[blue][DocumentProcessor][/blue] Processing document into per-page markdowns")
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
        rich_print(
            "[blue][DocumentProcessor][/blue] Processing document with structure-aware chunking"
        )
        document = self.convert_to_docling_doc(source)
        return self.extract_chunks(document)

    def extract_full_markdown(self, document: DoclingDocument) -> str:
        """
        Extracts the full document as a single Markdown string.

        Args:
            document (Document): Docling document object.

        Returns:
            str: Complete document in Markdown format.
        """
        md = document.export_to_markdown()
        rich_print(
            f"[blue][DocumentProcessor][/blue] Extracted full document Markdown ([cyan]{len(md)}[/cyan] chars)"
        )
        return md

    def cleanup(self) -> None:
        """Clean up document converter resources."""
        try:
            if hasattr(self, "converter"):
                del self.converter
            gc.collect()
            rich_print("[blue][DocumentProcessor][/blue] [green]Cleaned up resources[/green]")
        except Exception as e:
            rich_print(
                f"[blue][DocumentProcessor][/blue] [yellow]Warning during cleanup:[/yellow] {e}"
            )
