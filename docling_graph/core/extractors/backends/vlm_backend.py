"""
VLM (Vision-Language Model) extraction backend.
Handles document extraction using local VLM models via Docling.
"""

import gc
from typing import List, Type

import torch
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor, ExtractionFormatOption
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline
from pydantic import BaseModel, ValidationError

from ....logging_utils import get_component_logger

logger = get_component_logger("VlmBackend", __name__)


class VlmBackend:
    """Backend for VLM-based extraction (local only)."""

    doc_extractor: DocumentExtractor | None

    def __init__(self, model_name: str) -> None:
        """
        Initialize VLM backend with specified model.

        Args:
            model_name (str): HuggingFace model repository ID (e.g., 'numind/NuExtract-2.0-2B')
        """
        self.model_name = model_name
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Initialize Docling's VLM extractor with custom settings."""
        try:
            # Get default VLM pipeline options
            pipeline_options = ExtractionVlmPipeline.get_default_options()
            # Use getattr guards to avoid static typing issues with mypy
            vlm_opts = getattr(pipeline_options, "vlm_options", None)
            if vlm_opts is not None and hasattr(vlm_opts, "repo_id"):
                vlm_opts.repo_id = self.model_name

            # Define custom format options - MUST include backend parameter
            custom_format_options = {
                InputFormat.PDF: ExtractionFormatOption(
                    pipeline_cls=ExtractionVlmPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: ExtractionFormatOption(
                    pipeline_cls=ExtractionVlmPipeline,
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pipeline_options,
                ),
            }

            # Create extractor
            self.doc_extractor = DocumentExtractor(
                allowed_formats=[InputFormat.IMAGE, InputFormat.PDF],
                extraction_format_options=custom_format_options,
            )

            logger.info("Initialized with model: %s", self.model_name)

        except Exception as e:
            logger.error("Error initializing VLM backend: %s", e)
            raise

    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        """
        Extract structured data from entire document using VLM.

        Args:
            source (str): Path to source document.
            template (Type[BaseModel]): Pydantic model template.

        Returns:
            List[BaseModel]: List of extracted model instances (one per page/item).
        """
        logger.info("Extracting from: %s", source)

        if self.doc_extractor is None:
            raise RuntimeError("DocumentExtractor is not initialized")

        try:
            # Extract using VLM
            extraction_result = self.doc_extractor.extract(source=source, template=template)

            extracted_objects = []

            # Process each page's extracted data
            if extraction_result.pages:
                for page_num, page in enumerate(extraction_result.pages, 1):
                    if page.extracted_data:
                        try:
                            # Use model_validate for proper Pydantic validation
                            validated_model = template.model_validate(page.extracted_data)
                            extracted_objects.append(validated_model)
                        except ValidationError as e:
                            details = "; ".join(
                                f"{' -> '.join(map(str, error['loc']))}: {error['msg']}"
                                for error in e.errors()
                            )
                            logger.warning(
                                "Validation error on page %s: the data extracted by the VLM "
                                "does not match your Pydantic template. Details: %s",
                                page_num,
                                details,
                            )
                            continue

            if extracted_objects:
                logger.info("Extracted %s valid items", len(extracted_objects))
            else:
                logger.warning("No valid data extracted")

            return extracted_objects

        except Exception as e:
            logger.error("Error during VLM extraction: %s: %s", type(e).__name__, e)
            return []

    def cleanup(self) -> None:
        """
        Enhanced GPU cleanup with memory tracking.

        Performs:
        1. Memory tracking (before cleanup)
        2. Model-to-CPU transfer (if model exists)
        3. Resource deletion
        4. CUDA cache clearing
        5. Garbage collection
        6. Memory tracking (after cleanup)
        """
        try:
            # Track memory before cleanup
            memory_before = 0.0
            try:
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
                    logger.info("GPU memory before cleanup: %.2f MB", memory_before)
            except Exception:
                # Torch operations may fail in test environments
                pass

            # Move models to CPU before deletion (if accessible)
            if hasattr(self, "doc_extractor") and self.doc_extractor is not None:
                # Try to access the model through the extractor's pipeline
                try:
                    # Docling's extractor may have models in pipelines
                    if hasattr(self.doc_extractor, "_pipelines"):
                        for pipeline in self.doc_extractor._pipelines.values():
                            if hasattr(pipeline, "model") and pipeline.model is not None:
                                if hasattr(pipeline.model, "to"):
                                    logger.info("Moving model to CPU...")
                                    pipeline.model.to("cpu")
                except Exception as e:
                    # Model access may fail, continue with cleanup
                    try:
                        logger.warning("Could not move model to CPU: %s", e)
                    except Exception:
                        # Even logging might fail with mocks
                        pass

                # Clear the extractor reference
                self.doc_extractor = None
                logger.info("Extractor deleted")

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared")
            except Exception:
                # Torch operations may fail in test environments
                pass

            # Track memory after cleanup
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_freed = memory_before - memory_after
                logger.info(
                    "GPU cleanup complete (freed %.2f MB, remaining %.2f MB)",
                    memory_freed,
                    memory_after,
                )
            else:
                logger.info("Cleanup complete (no GPU detected)")

        except Exception as e:
            logger.warning("Warning during cleanup: %s", e)

    def cleanup_all_gpus(self) -> None:
        """
        Cleanup all available GPUs.

        Useful for multi-GPU setups or when model was distributed across devices.
        """
        if not torch.cuda.is_available():
            logger.info("No CUDA devices available")
            return

        device_count = torch.cuda.device_count()
        logger.info("Cleaning up %s GPU(s)...", device_count)

        for device_id in range(device_count):
            try:
                # Try to clear cache for this device
                memory_before = 0.0
                try:
                    memory_before = torch.cuda.memory_allocated(device_id) / 1024**2
                except Exception:
                    pass

                # Clear cache - this is the critical operation
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                memory_after = 0.0
                try:
                    memory_after = torch.cuda.memory_allocated(device_id) / 1024**2
                except Exception:
                    pass

                memory_freed = memory_before - memory_after
                logger.info("GPU %s: freed %.2f MB", device_id, memory_freed)
            except Exception as e:
                logger.warning("Could not clear GPU %s: %s", device_id, e)

        logger.info("Multi-GPU cleanup complete")
