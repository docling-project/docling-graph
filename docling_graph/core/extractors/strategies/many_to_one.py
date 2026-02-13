"""
Many-to-one extraction strategy.

Extracts one consolidated model from an entire document.

For LLM backend, extraction behavior is driven by the configured
extraction contract (e.g., direct single-pass or staged multi-pass).
"""

import logging
import time
from typing import Any, Tuple, Type, cast

from docling_core.types.doc import DoclingDocument
from pydantic import BaseModel

from ....protocols import (
    Backend,
    ExtractionBackendProtocol,
    TextExtractionBackendProtocol,
    get_backend_type,
    is_llm_backend,
    is_vlm_backend,
)
from ...utils.dict_merger import merge_pydantic_models
from ..document_processor import DocumentProcessor
from ..extractor_base import BaseExtractor

# Initialize logger
logger = logging.getLogger(__name__)


class ManyToOneStrategy(BaseExtractor):
    """
    Many-to-one extraction strategy.

    Extracts one consolidated model from an entire document.
    """

    def __init__(
        self,
        backend: Backend,
        docling_config: str = "ocr",
    ) -> None:
        """
        Initialize extraction strategy.

        Args:
            backend: Extraction backend (VLM or LLM)
            docling_config: Docling pipeline config ("ocr" or "vision")
        """
        super().__init__()
        self.backend = backend

        # Cache protocol checks
        self._is_llm = is_llm_backend(self.backend)
        self._is_vlm = is_vlm_backend(self.backend)
        self._backend_type = get_backend_type(self.backend)

        # No chunking for direct-only extraction
        self.doc_processor = DocumentProcessor(
            docling_config=docling_config,
            chunker_config=None,
        )

        logger.info(
            f"Initialized with {self._backend_type.upper()} backend: "
            f"Backend={self.backend.__class__.__name__}"
        )

    def extract(
        self, source: str, template: Type[BaseModel]
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """
        Extract structured data using many-to-one strategy.

        Returns:
            Tuple containing:
                - List with single merged model, or empty list on failure
                - DoclingDocument object (or None if extraction failed)
        """
        try:
            if self._is_vlm:
                logger.info("Using VLM backend")
                return self._extract_with_vlm(
                    cast(ExtractionBackendProtocol, self.backend), source, template
                )

            elif self._is_llm:
                logger.info("Using LLM backend (direct extraction)")
                return self._extract_with_llm(
                    cast(TextExtractionBackendProtocol, self.backend), source, template
                )

            else:
                backend_class = self.backend.__class__.__name__
                raise TypeError(
                    f"Backend '{backend_class}' does not implement a recognized extraction protocol"
                )

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            import traceback

            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return [], None

    def _extract_with_vlm(
        self, backend: ExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """Extract using VLM backend."""
        try:
            logger.info("Running VLM extraction...")
            models = backend.extract_from_document(source, template)

            if not models:
                logger.warning("No models extracted by VLM")
                return [], None

            if len(models) == 1:
                logger.info("Single-page document extracted")
                return models, None

            logger.info(f"Merging {len(models)} page models...")
            merged_model = merge_pydantic_models(models, template)

            if merged_model:
                logger.info("Successfully merged VLM page models")
                return [merged_model], None
            else:
                logger.warning("Merge failed - returning all page models (zero data loss)")
                return models, None

        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            import traceback

            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return [], None

    def _extract_with_llm(
        self, backend: TextExtractionBackendProtocol, source: str, template: Type[BaseModel]
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """Extract using LLM backend (contract-driven full-document extraction)."""
        try:
            document = self.doc_processor.convert_to_docling_doc(source)
            return self._extract_direct_mode(backend, document, template)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            import traceback

            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return [], None

    def _extract_with_llm_from_text(
        self,
        backend: TextExtractionBackendProtocol,
        text: str,
        template: Type[BaseModel],
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """
        Extract using LLM backend from raw text/markdown input.

        Handles TEXT, TEXT_FILE, and MARKDOWN inputs that don't have a DoclingDocument.
        """
        try:
            return self._extract_direct_mode_from_text(backend, text, template)
        except Exception as e:
            logger.error(f"LLM text extraction failed: {e}")
            import traceback

            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return [], None

    def _extract_direct_mode_from_text(
        self,
        backend: TextExtractionBackendProtocol,
        text: str,
        template: Type[BaseModel],
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """Contract-driven extraction from raw text."""
        logger.info("Contract-driven mode: full-text extraction")

        try:
            if (
                hasattr(self, "trace_data")
                and self.trace_data is not None
                and hasattr(backend, "trace_data")
            ):
                backend.trace_data = self.trace_data

            start_time = time.time()
            model = backend.extract_from_markdown(
                markdown=text,
                template=template,
                context="text input",
                is_partial=False,
            )
            extraction_time = time.time() - start_time

            if hasattr(self, "trace_data") and self.trace_data:
                from ....pipeline.trace import ExtractionData

                extraction_metadata: dict[str, Any] = {}
                backend_diag = getattr(backend, "last_call_diagnostics", None)
                if isinstance(backend_diag, dict) and backend_diag:
                    extraction_metadata.update(backend_diag)
                self.trace_data.extractions.append(
                    ExtractionData(
                        extraction_id=0,
                        source_type="chunk",
                        source_id=0,
                        parsed_model=model,
                        extraction_time=extraction_time,
                        error=None,
                        metadata=extraction_metadata,
                    )
                )

            if model:
                logger.info("Direct text extraction successful")
                return [model], None
            else:
                logger.warning("Direct text extraction returned no model")
                return [], None

        except Exception as e:
            logger.error(f"Direct text extraction failed: {e}")
            if hasattr(self, "trace_data") and self.trace_data:
                from ....pipeline.trace import ExtractionData

                self.trace_data.extractions.append(
                    ExtractionData(
                        extraction_id=0,
                        source_type="chunk",
                        source_id=0,
                        parsed_model=None,
                        extraction_time=0.0,
                        error=str(e),
                    )
                )
            return [], None

    def _extract_direct_mode(
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """Contract-driven full-document extraction."""
        logger.info("Contract-driven mode: full-document extraction")

        try:
            if hasattr(self, "trace_data") and self.trace_data:
                from ....pipeline.trace import PageData

                page_markdowns = self.doc_processor.extract_page_markdowns(document)
                for page_num, page_md in enumerate(page_markdowns, start=1):
                    self.trace_data.pages.append(
                        PageData(page_number=page_num, text_content=page_md, metadata={})
                    )

            full_markdown = self.doc_processor.extract_full_markdown(document)

            if (
                hasattr(self, "trace_data")
                and self.trace_data is not None
                and hasattr(backend, "trace_data")
            ):
                backend.trace_data = self.trace_data

            start_time = time.time()
            model = backend.extract_from_markdown(
                markdown=full_markdown,
                template=template,
                context="full document",
                is_partial=False,
            )
            extraction_time = time.time() - start_time

            if hasattr(self, "trace_data") and self.trace_data:
                from ....pipeline.trace import ExtractionData

                extraction_metadata: dict[str, Any] = {}
                backend_diag = getattr(backend, "last_call_diagnostics", None)
                if isinstance(backend_diag, dict) and backend_diag:
                    extraction_metadata.update(backend_diag)
                if getattr(self.trace_data, "staged_trace", None):
                    extraction_metadata["extraction_contract"] = "staged"
                    extraction_metadata["staged_passes_count"] = 3
                elif hasattr(self.trace_data, "staged_passes") and self.trace_data.staged_passes:
                    extraction_metadata["extraction_contract"] = "staged"
                    extraction_metadata["staged_passes_count"] = len(self.trace_data.staged_passes)
                self.trace_data.extractions.append(
                    ExtractionData(
                        extraction_id=0,
                        source_type="chunk",
                        source_id=0,
                        parsed_model=model,
                        extraction_time=extraction_time,
                        error=None,
                        metadata=extraction_metadata,
                    )
                )

            if model:
                logger.info("Direct extraction successful")
                return [model], document
            else:
                logger.warning("Direct extraction returned no model")
                return [], document

        except Exception as e:
            logger.error(f"Direct extraction failed: {e}")
            if hasattr(self, "trace_data") and self.trace_data:
                from ....pipeline.trace import ExtractionData

                self.trace_data.extractions.append(
                    ExtractionData(
                        extraction_id=0,
                        source_type="chunk",
                        source_id=0,
                        parsed_model=None,
                        extraction_time=0.0,
                        error=str(e),
                    )
                )
            return [], document
