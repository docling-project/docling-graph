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
from ..contracts.delta.strategy_ops import extract_delta_from_document, extract_delta_from_text
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
        extraction_contract: str = "direct",
        use_chunking: bool = True,
        chunk_max_tokens: int | None = None,
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
        self._extraction_contract = extraction_contract

        if extraction_contract == "delta" and not use_chunking:
            raise ValueError("Delta extraction requires use_chunking=True.")

        chunker_config: dict[str, Any] | None = None
        if use_chunking and extraction_contract == "delta":
            chunker_config = {"chunk_max_tokens": int(chunk_max_tokens or 512)}

        self.doc_processor = DocumentProcessor(
            docling_config=docling_config,
            chunker_config=chunker_config,
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
                logger.info("Using LLM backend (%s extraction)", self._extraction_contract)
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
            merged_model = merge_pydantic_models(
                models,
                template,
                description_merge_fields={"description", "summary"},
                description_merge_max_length=4096,
            )

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
            conversion_started_at = time.time()
            document = self.doc_processor.convert_to_docling_doc(source)
            conversion_runtime_seconds = time.time() - conversion_started_at
            return self._extract_direct_mode(
                backend,
                document,
                template,
                conversion_runtime_seconds=conversion_runtime_seconds,
            )
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
            # Emit one docling_conversion event so the step appears (conversion skipped / text input)
            if hasattr(self, "trace_data") and self.trace_data:
                self.trace_data.emit(
                    "page_markdown_extracted",
                    "extraction",
                    {
                        "page_number": 1,
                        "text_content": text or "",
                        "metadata": {"source": "text_input"},
                    },
                )
                self.trace_data.emit(
                    "docling_conversion_completed",
                    "extraction",
                    {"runtime_seconds": 0.0, "page_count": 1, "source": "text_input"},
                )
            if (
                hasattr(self, "trace_data")
                and self.trace_data is not None
                and hasattr(backend, "trace_data")
            ):
                backend.trace_data = self.trace_data

            if self._extraction_contract == "delta" and hasattr(
                backend, "extract_from_chunk_batches"
            ):
                model, extraction_time = extract_delta_from_text(
                    backend=backend,
                    doc_processor=self.doc_processor,
                    text=text,
                    template=template,
                )
            else:
                start_time = time.time()
                model = backend.extract_from_markdown(
                    markdown=text,
                    template=template,
                    context="text input",
                    is_partial=False,
                )
                extraction_time = time.time() - start_time

            if hasattr(self, "trace_data") and self.trace_data:
                extraction_metadata: dict[str, Any] = {}
                backend_diag = getattr(backend, "last_call_diagnostics", None)
                if isinstance(backend_diag, dict) and backend_diag:
                    extraction_metadata.update(backend_diag)
                self.trace_data.emit(
                    "extraction_completed",
                    "extraction",
                    {
                        "extraction_id": 0,
                        "source_type": "chunk",
                        "source_id": 0,
                        "parsed_model": model,
                        "extraction_time": extraction_time,
                        "error": None,
                        "metadata": extraction_metadata,
                    },
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
                self.trace_data.emit(
                    "extraction_failed",
                    "extraction",
                    {
                        "extraction_id": 0,
                        "source_type": "chunk",
                        "source_id": 0,
                        "parsed_model": None,
                        "extraction_time": 0.0,
                        "error": str(e),
                        "metadata": {},
                    },
                )
            return [], None

    def _extract_direct_mode(
        self,
        backend: TextExtractionBackendProtocol,
        document: DoclingDocument,
        template: Type[BaseModel],
        conversion_runtime_seconds: float = 0.0,
    ) -> Tuple[list[BaseModel], DoclingDocument | None]:
        """Contract-driven full-document extraction."""
        logger.info("Contract-driven mode: full-document extraction")

        try:
            full_markdown = self.doc_processor.extract_full_markdown(document)
            if self._extraction_contract == "delta" and hasattr(
                backend, "extract_from_chunk_batches"
            ):
                model, extraction_time = extract_delta_from_document(
                    backend=backend,
                    doc_processor=self.doc_processor,
                    document=document,
                    template=template,
                    trace_data=self.trace_data if hasattr(self, "trace_data") else None,
                )

                if hasattr(self, "trace_data") and self.trace_data:
                    extraction_metadata: dict[str, Any] = {}
                    backend_diag = getattr(backend, "last_call_diagnostics", None)
                    if isinstance(backend_diag, dict) and backend_diag:
                        extraction_metadata.update(backend_diag)
                    self.trace_data.emit(
                        "extraction_completed",
                        "extraction",
                        {
                            "extraction_id": 0,
                            "source_type": "chunk_batch",
                            "source_id": 0,
                            "parsed_model": model,
                            "extraction_time": extraction_time,
                            "error": None,
                            "metadata": extraction_metadata,
                        },
                    )

                if model:
                    logger.info("Delta extraction successful")
                    return [model], document
                logger.warning(
                    "Delta extraction returned no model; falling back to direct extraction"
                )
                if hasattr(self, "trace_data") and self.trace_data:
                    delta_trace_payload = (
                        self.trace_data.latest_payload("delta_trace_emitted")
                        if hasattr(self.trace_data, "latest_payload")
                        else None
                    )
                    self.trace_data.emit(
                        "delta_failed_then_direct_fallback",
                        "extraction",
                        {
                            "reason": "delta_returned_no_model",
                            "delta_quality_gate": (
                                delta_trace_payload.get("quality_gate")
                                if isinstance(delta_trace_payload, dict)
                                else None
                            ),
                            "delta_merge_stats": (
                                delta_trace_payload.get("merge_stats")
                                if isinstance(delta_trace_payload, dict)
                                else None
                            ),
                            "delta_normalizer_stats": (
                                delta_trace_payload.get("normalizer_stats")
                                if isinstance(delta_trace_payload, dict)
                                else None
                            ),
                        },
                    )
                fallback_model = self._extract_direct_fallback_model(
                    backend, full_markdown, template
                )
                if fallback_model:
                    logger.info("Direct fallback after delta extraction succeeded")
                    if hasattr(self, "trace_data") and self.trace_data:
                        self.trace_data.emit(
                            "delta_failed_then_direct_fallback",
                            "extraction",
                            {"reason": "direct_fallback_succeeded"},
                        )
                    return [fallback_model], document
                logger.warning("Direct fallback after delta extraction returned no model")
                if hasattr(self, "trace_data") and self.trace_data:
                    self.trace_data.emit(
                        "delta_failed_then_direct_fallback",
                        "extraction",
                        {"reason": "direct_fallback_returned_no_model"},
                    )
                return [], document

            if hasattr(self, "trace_data") and self.trace_data:
                page_markdown_started_at = time.time()
                page_markdowns = self.doc_processor.extract_page_markdowns(document)
                page_markdown_runtime_seconds = time.time() - page_markdown_started_at
                if len(page_markdowns) == 0:
                    # Ensure docling_conversion step appears (e.g. single-page or non-paged doc)
                    self.trace_data.emit(
                        "page_markdown_extracted",
                        "extraction",
                        {"page_number": 1, "text_content": full_markdown or "", "metadata": {}},
                    )
                else:
                    for page_num, page_md in enumerate(page_markdowns, start=1):
                        self.trace_data.emit(
                            "page_markdown_extracted",
                            "extraction",
                            {"page_number": page_num, "text_content": page_md, "metadata": {}},
                        )
                self.trace_data.emit(
                    "docling_conversion_completed",
                    "extraction",
                    {
                        "runtime_seconds": conversion_runtime_seconds
                        + page_markdown_runtime_seconds,
                        "page_count": len(page_markdowns) if page_markdowns else 1,
                        "source": "docling_document_conversion",
                    },
                )

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
                direct_extraction_metadata: dict[str, Any] = {}
                backend_diag = getattr(backend, "last_call_diagnostics", None)
                if isinstance(backend_diag, dict) and backend_diag:
                    direct_extraction_metadata.update(backend_diag)
                if self.trace_data.find_events("staged_trace_emitted"):
                    direct_extraction_metadata["extraction_contract"] = "staged"
                    direct_extraction_metadata["staged_passes_count"] = len(
                        self.trace_data.find_events("staged_trace_emitted")
                    )
                self.trace_data.emit(
                    "extraction_completed",
                    "extraction",
                    {
                        "extraction_id": 0,
                        "source_type": "chunk",
                        "source_id": 0,
                        "parsed_model": model,
                        "extraction_time": extraction_time,
                        "error": None,
                        "metadata": direct_extraction_metadata,
                    },
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
                self.trace_data.emit(
                    "extraction_failed",
                    "extraction",
                    {
                        "extraction_id": 0,
                        "source_type": "chunk",
                        "source_id": 0,
                        "parsed_model": None,
                        "extraction_time": 0.0,
                        "error": str(e),
                        "metadata": {},
                    },
                )
            return [], document

    def _extract_direct_fallback_model(
        self,
        backend: TextExtractionBackendProtocol,
        markdown: str,
        template: Type[BaseModel],
    ) -> BaseModel | None:
        """Run one best-effort direct extraction fallback after delta returns no model."""
        original_contract = getattr(backend, "extraction_contract", None)
        switched_contract = False
        try:
            # Force direct mode when backend supports contract switching.
            if isinstance(original_contract, str) and original_contract != "direct":
                try:
                    backend.extraction_contract = "direct"
                    switched_contract = True
                except Exception:
                    switched_contract = False

            return backend.extract_from_markdown(
                markdown=markdown,
                template=template,
                context="full document (delta fallback)",
                is_partial=False if switched_contract else True,
            )
        except Exception as e:
            logger.warning("Direct fallback after delta failed: %s", e)
            return None
        finally:
            if switched_contract and original_contract is not None:
                try:
                    backend.extraction_contract = original_contract
                except Exception:
                    pass
