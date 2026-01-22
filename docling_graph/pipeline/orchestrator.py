"""
Pipeline orchestrator for coordinating stage execution.

This module provides the main orchestrator that coordinates the execution
of pipeline stages, handles errors, and manages resource cleanup.
"""

import gc
import logging
from typing import Any, Dict, Union

from ..core import PipelineConfig
from ..exceptions import PipelineError
from .context import PipelineContext
from .stages import (
    DoclingExportStage,
    ExportStage,
    ExtractionStage,
    GraphConversionStage,
    PipelineStage,
    TemplateLoadingStage,
    VisualizationStage,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates pipeline execution through stages.

    The orchestrator manages the execution flow, passing context between
    stages, handling errors, and ensuring proper resource cleanup.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize orchestrator with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stages: list[PipelineStage] = [
            TemplateLoadingStage(),
            ExtractionStage(),
            DoclingExportStage(),
            GraphConversionStage(),
            ExportStage(),
            VisualizationStage(),
        ]

    def run(self) -> PipelineContext:
        """
        Execute all pipeline stages.

        Returns:
            Final pipeline context with all results

        Raises:
            PipelineError: If any stage fails
        """
        context = PipelineContext(config=self.config)
        current_stage = None

        logger.info("--- Starting Docling-Graph Pipeline ---")

        try:
            for stage in self.stages:
                current_stage = stage
                logger.info(f">>> Stage: {stage.name()}")
                context = stage.execute(context)

            logger.info("--- Pipeline Completed Successfully ---")
            return context

        except Exception as e:
            stage_name = current_stage.name() if current_stage else "Unknown"
            logger.error(f"Pipeline failed at stage: {stage_name}")

            if isinstance(e, PipelineError):
                raise

            raise PipelineError(
                f"Pipeline failed at stage '{stage_name}': {type(e).__name__}",
                details={"stage": stage_name, "error": str(e), "error_type": type(e).__name__},
            ) from e

        finally:
            self._cleanup(context)

    def _cleanup(self, context: PipelineContext) -> None:
        """
        Clean up resources after pipeline execution.

        Args:
            context: Pipeline context with resources to clean
        """
        logger.info("Cleaning up resources...")

        if context.extractor:
            if hasattr(context.extractor, "backend"):
                backend = context.extractor.backend
                if hasattr(backend, "cleanup"):
                    backend.cleanup()

            if hasattr(context.extractor, "doc_processor"):
                doc_processor = context.extractor.doc_processor
                if hasattr(doc_processor, "cleanup"):
                    doc_processor.cleanup()

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> None:
    """
    Run the extraction and graph conversion pipeline.

    This is the main entry point for pipeline execution. It accepts either
    a PipelineConfig object or a dictionary of configuration parameters.

    Args:
        config: Pipeline configuration as PipelineConfig or dict

    Raises:
        PipelineError: If pipeline execution fails

    Example:
        >>> from docling_graph import run_pipeline
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "backend": "llm",
        ...     "inference": "remote"
        ... }
        >>> run_pipeline(config)
    """
    if isinstance(config, dict):
        config = PipelineConfig(**config)

    orchestrator = PipelineOrchestrator(config)
    orchestrator.run()
