"""
Many-to-one extraction strategy.
Processes entire document and returns single consolidated model.
"""

from typing import List, Optional, Type

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel
from rich import print as rich_print

from ....protocols import (
    Backend,
    ExtractionBackendProtocol,
    TextExtractionBackendProtocol,
    get_backend_type,
    is_llm_backend,
    is_vlm_backend,
)
from ...utils.dict_merger import merge_pydantic_models
from ..chunk_batcher import ChunkBatcher
from ..document_processor import DocumentProcessor
from ..extractor_base import BaseExtractor


class ManyToOneStrategy(BaseExtractor):
    """Many-to-one extraction strategy.

    Extracts one consolidated model from an entire document
    using Protocol-based backend type checking (VLM or LLM).
    """

    def __init__(
        self,
        backend: Backend,
        docling_config: str = "default",
        use_chunking: bool = True,
        llm_consolidation: bool = False,
        chunker_config: Optional[dict] = None,
    ) -> None:
        """
        Initialize the extraction strategy with a backend and document processor.

        Args:
            backend: Extraction backend (VLM or LLM)
            docling_config: Docling pipeline config ("ocr" or "vision")
            llm_consolidation: If True, run a final LLM pass to merge results.
            use_chunking: Use structure-aware chunking instead of page-by-page (default: True)
            chunker_config: Configuration for HybridChunker. Example:
                {
                    "tokenizer_name": "mistralai/Mistral-7B-v0.1",
                    "max_tokens": 8000,
                    "merge_peers": True
                }
                If None and use_chunking=True, uses default tokenizer with backend's context limit.
        """
        self.backend = backend
        self.llm_consolidation = llm_consolidation
        self.use_chunking = use_chunking

        # Auto-configure chunker based on backend if not provided
        if use_chunking and chunker_config is None and hasattr(backend, "client"):
            context_limit = getattr(backend.client, "context_limit", 8000)
            content_ratio = getattr(backend.client, "content_ratio", 0.8)
            # Reserve room for prompt/template overhead
            max_tokens = int(context_limit * content_ratio)
            chunker_config = {"max_tokens": max_tokens}

        self.doc_processor = DocumentProcessor(
            docling_config=docling_config,
            chunker_config=chunker_config if use_chunking else None,
        )

        backend_type = get_backend_type(self.backend)
        rich_print(
            f"[blue][ManyToOneStrategy][/blue] Initialized with {backend_type.upper()} backend: "
            f"[cyan]{self.backend.__class__.__name__}[/cyan]\n"
            f"  • Chunking: {'enabled' if self.use_chunking else 'disabled'}\n"
            f"  • LLM Consolidation: {'enabled' if self.llm_consolidation and is_llm_backend(self.backend) else 'disabled'}"
        )

    # Public extraction entry point
    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """Extract structured data using a many-to-one strategy.

        - VLM backend: Extracts all pages and merges the results.
        - LLM backend: Uses structure-aware chunking (if enabled) or falls back to page-by-page.

        Returns:
            A list containing a single merged model instance, or an empty list on failure.
        """
        try:
            if is_vlm_backend(self.backend):
                rich_print("[blue][ManyToOneStrategy][/blue] Using VLM backend for extraction")
                return self._extract_with_vlm(self.backend, source, template)
            elif is_llm_backend(self.backend):
                rich_print("[blue][ManyToOneStrategy][/blue] Using LLM backend for extraction")
                return self._extract_with_llm(self.backend, source, template)
            else:
                backend_class = self.backend.__class__.__name__
                raise TypeError(
                    f"Backend '{backend_class}' does not implement a recognized extraction protocol. "
                    "Expected either a VLM or LLM backend."
                )
        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] Extraction error: {e}")
            return []  # Graceful fallback

    # VLM backend extraction
    def _extract_with_vlm(
        self, backend: ExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract using a Vision-Language Model (VLM) backend, merging page-level models."""
        try:
            rich_print("[blue][ManyToOneStrategy][/blue] Running VLM extraction...")
            models = backend.extract_from_document(source, template)

            if not models:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] No models extracted by VLM backend"
                )
                return []

            if len(models) == 1:
                rich_print(
                    "[blue][ManyToOneStrategy][/blue] Single-page document extracted successfully"
                )
                return models

            # Merge multiple page-level models
            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Merging [cyan]{len(models)}[/cyan] extracted page models..."
            )
            merged_model = merge_pydantic_models(models, template)

            if merged_model:
                rich_print(
                    "[green][ManyToOneStrategy][/green] Successfully merged all VLM page models"
                )
                return [merged_model]
            else:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Merge failed — returning first page result"
                )
                return [models[0]]

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] VLM extraction failed: {e}")
            return []

    # LLM backend extraction
    def _extract_with_llm(
        self, backend: TextExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract using an LLM backend with intelligent strategy selection."""
        try:
            document = self.doc_processor.convert_to_docling_doc(source)

            # Use chunking if enabled
            if self.use_chunking:
                return self._extract_with_chunks(backend, document, template)

            # Fallback to legacy page-by-page or full-doc extraction
            if hasattr(backend.client, "context_limit"):
                context_limit = backend.client.context_limit
                full_markdown = self.doc_processor.extract_full_markdown(document)
                estimated_tokens = len(full_markdown) / 3.5  # Rough heuristic

                if estimated_tokens < (context_limit * 0.9):
                    rich_print(
                        f"[blue][ManyToOneStrategy][/blue] Document fits context "
                        f"({int(estimated_tokens)} tokens) — using full-document extraction"
                    )
                    return self._extract_full_document(backend, full_markdown, template)
                else:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] Document too large "
                        f"({int(estimated_tokens)} tokens) — using page-by-page fallback"
                    )
                    return self._extract_pages_and_merge(backend, document, template)
            else:
                # No context info, default to full-document attempt
                full_markdown = self.doc_processor.extract_full_markdown(document)
                return self._extract_full_document(backend, full_markdown, template)

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] LLM extraction failed: {e}")
            return []

    # Chunk-based extraction
    def _extract_with_chunks(
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
    ) -> List[BaseModel]:
        """Extract using structure-aware chunks with adaptive batching."""
        try:
            chunks = self.doc_processor.extract_chunks(document)
            total_chunks = len(chunks)

            # Get context limit from backend
            context_limit = getattr(
                backend.client, "context_limit", 3500
            )  # Fallback for unknown backends

            # Create batcher
            batcher = ChunkBatcher(
                context_limit=context_limit,
                system_prompt_tokens=500,
                response_buffer_tokens=500,
                merge_threshold=0.85,
            )

            # Batch chunks for efficient processing
            batches = batcher.batch_chunks(chunks)

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Starting batch extraction "
                f"({len(batches)} batches from {total_chunks} chunks)..."
            )

            extracted_models: List[BaseModel] = []

            for batch in batches:
                batch_label = f"batch {batch.batch_id + 1} ({batch.chunk_count} chunks)"
                rich_print(f"[blue][ManyToOneStrategy][/blue] Extracting from {batch_label}")

                # Send combined batch to LLM
                model = backend.extract_from_markdown(
                    markdown=batch.combined_text,
                    template=template,
                    context=batch_label,
                    is_partial=True,
                )

                if model:
                    extracted_models.append(model)
                else:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] {batch_label} returned no model"
                    )

            if not extracted_models:
                rich_print("[red][ManyToOneStrategy][/red] No models extracted from any batch")
                return []

            if len(extracted_models) == 1:
                rich_print(
                    "[blue][ManyToOneStrategy][/blue] Single batch extracted — no merge needed"
                )
                return extracted_models

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Programmatically merging "
                f"[cyan]{len(extracted_models)}[/cyan] batch models..."
            )
            programmatic_model = merge_pydantic_models(extracted_models, template)

            if not programmatic_model:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Programmatic merge failed. Returning first extracted model."
                )
                return [extracted_models[0]]

            # Consolidation step
            if self.llm_consolidation and is_llm_backend(self.backend):
                rich_print(
                    "[green][ManyToOneStrategy][/green] Programmatic merge complete. Starting LLM consolidation pass..."
                )
                final_model = self.backend.consolidate_from_pydantic_models(
                    raw_models=extracted_models,
                    programmatic_model=programmatic_model,
                    template=template,
                )
                if final_model:
                    return [final_model]
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] LLM consolidation failed. Falling back to programmatic merge."
                )
                return [programmatic_model]
            else:
                return [programmatic_model]

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] Batch extraction failed: {e}")
            return []

    # Full-document extraction (LLM)
    def _extract_full_document(
        self, backend: TextExtractionBackendProtocol, full_markdown: str, template: Type[BaseModel]
    ) -> List[BaseModel]:
        """Extract a single consolidated model from full document markdown."""
        try:
            model = backend.extract_from_markdown(
                markdown=full_markdown,
                template=template,
                context="full document",
                is_partial=False,
            )

            if model:
                rich_print(
                    "[green][ManyToOneStrategy][/green] Successfully extracted consolidated model from full document"
                )
                return [model]
            else:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] Full-document extraction returned no model"
                )
                return []

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] Full-document extraction failed: {e}")
            return []

    # Page-by-page extraction + merging (LLM)
    def _extract_pages_and_merge(
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
    ) -> List[BaseModel]:
        """Extract individual page models and intelligently merge them into one."""
        try:
            page_markdowns = self.doc_processor.extract_page_markdowns(document)
            total_pages = len(page_markdowns)

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Starting page-by-page extraction ({total_pages} pages)..."
            )

            extracted_models: List[BaseModel] = []

            for page_num, page_md in enumerate(page_markdowns, 1):
                rich_print(
                    f"[blue][ManyToOneStrategy][/blue] Extracting from page {page_num}/{total_pages}"
                )

                model = backend.extract_from_markdown(
                    markdown=page_md,
                    template=template,
                    context=f"page {page_num}",
                    is_partial=True,
                )

                if model:
                    extracted_models.append(model)
                else:
                    rich_print(
                        f"[yellow][ManyToOneStrategy][/yellow] Page {page_num} returned no model"
                    )

            if not extracted_models:
                rich_print("[red][ManyToOneStrategy][/red] No models extracted from any page")
                return []

            if len(extracted_models) == 1:
                rich_print(
                    "[blue][ManyToOneStrategy][/blue] Single page extracted — no merge needed"
                )
                return extracted_models

            rich_print(
                f"[blue][ManyToOneStrategy][/blue] Programmatically merging "
                f"[cyan]{len(extracted_models)}[/cyan] page models..."
            )
            programmatic_model = merge_pydantic_models(extracted_models, template)

            if not programmatic_model:
                rich_print(
                    "[yellow][ManyToOneStrategy][/yellow] "
                    "Programmatic merge failed. Returning first extracted model."
                )
                return [extracted_models[0]]

            # Consolidation step
            if self.llm_consolidation and is_llm_backend(self.backend):
                rich_print(
                    "[blue]Programmatic merge complete. Starting LLM consolidation pass...[/blue]"
                )
                final_model = self.backend.consolidate_from_pydantic_models(
                    raw_models=extracted_models,
                    programmatic_model=programmatic_model,
                    template=template,
                )
                if final_model:
                    return [final_model]
                rich_print(
                    "[yellow]LLM consolidation failed. Falling back to programmatic merge.[/yellow]"
                )
                return [programmatic_model]
            else:
                return [programmatic_model]

        except Exception as e:
            rich_print(f"[red][ManyToOneStrategy][/red] Page-by-page extraction failed: {e}")
            return []
