"""
One-to-one extraction strategy.
Processes each page independently and returns multiple models.
"""

from typing import List, Tuple, Type

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
from ..document_processor import DocumentProcessor
from ..extractor_base import BaseExtractor


class OneToOneStrategy(BaseExtractor):
    """One-to-one extraction strategy.
    Extracts one model per page/item using Protocol-based type checking.
    """

    def __init__(self, backend: Backend, docling_config: str = "default") -> None:
        """Initialize with a backend (VlmBackend or LlmBackend).

        Args:
            backend: Extraction backend instance implementing either
                ExtractionBackendProtocol or TextExtractionBackendProtocol.
            docling_config: Docling pipeline configuration ('ocr' or 'vision').
        """
        self.backend = backend
        self.doc_processor = DocumentProcessor(docling_config=docling_config)

        backend_type = get_backend_type(self.backend)
        rich_print(
            f"[blue][OneToOneStrategy][/blue] Initialized with {backend_type.upper()} backend: "
            f"[cyan]{self.backend.__class__.__name__}[/cyan]"
        )

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """Extract data using one-to-one strategy.

        For VLM: Uses direct VLM extraction (already page-based).
        For LLM: Converts to markdown and processes each page separately.

        Args:
            source: Path to the source document.
            template: Pydantic model template to extract into.

        Returns:
            Tuple containing:
                - List of extracted Pydantic model instances (one per page).
                - The DoclingDocument object used during extraction (or None if extraction failed).
        """
        try:
            if is_vlm_backend(self.backend):
                rich_print("[blue][OneToOneStrategy][/blue] Using VLM backend for extraction")
                return self._extract_with_vlm(self.backend, source, template)
            elif is_llm_backend(self.backend):
                rich_print("[blue][OneToOneStrategy][/blue] Using LLM backend for extraction")
                return self._extract_with_llm(self.backend, source, template)
            else:
                backend_class = self.backend.__class__.__name__
                raise TypeError(
                    f"Backend '{backend_class}' does not implement a recognized extraction protocol. "
                    "Expected either a VLM or LLM backend."
                )
        except Exception as e:
            rich_print(f"[red][OneToOneStrategy][/red] Extraction error: {e}")
            return [], None

    def _extract_with_vlm(
        self, backend: ExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """VLM path: delegate to document-level extraction."""
        models = backend.extract_from_document(source, template)
        return models, None

    def _extract_with_llm(
        self, backend: TextExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[List[BaseModel], DoclingDocument | None]:
        """LLM path: convert to markdown and process per page."""
        document = self.doc_processor.convert_to_docling_doc(source)
        page_markdowns = self.doc_processor.extract_page_markdowns(document)

        extracted_models: List[BaseModel] = []
        total_pages = len(page_markdowns)

        # Import for trace data capture
        import time

        from ....pipeline.trace import ExtractionData

        extraction_id = 0
        for page_num, page_md in enumerate(page_markdowns, start=1):
            rich_print(f"[blue][OneToOneStrategy][/blue] Processing page {page_num}/{total_pages}")

            start_time = time.time()
            error = None

            try:
                model = backend.extract_from_markdown(
                    markdown=page_md,
                    template=template,
                    context=f"page {page_num}",
                    is_partial=True,
                )
            except Exception as e:
                error = str(e)
                model = None

            extraction_time = time.time() - start_time

            # Capture trace data if enabled
            if hasattr(self, "trace_data") and self.trace_data:
                extraction_data = ExtractionData(
                    extraction_id=extraction_id,
                    source_type="page",
                    source_id=page_num - 1,  # 0-indexed
                    parsed_model=model,
                    extraction_time=extraction_time,
                    error=error,
                )
                self.trace_data.extractions.append(extraction_data)
                extraction_id += 1

            if model:
                extracted_models.append(model)
            else:
                rich_print(
                    f"[yellow][OneToOneStrategy][/yellow] No model extracted from page {page_num}"
                )

        rich_print(
            f"[green][OneToOneStrategy][/green] Successfully extracted {len(extracted_models)} model(s)"
        )
        return extracted_models, document
