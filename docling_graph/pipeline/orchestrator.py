"""
Pipeline orchestrator for coordinating stage execution.

This module provides the main orchestrator that coordinates the execution
of pipeline stages, handles errors, and manages resource cleanup.
"""

import gc
import logging
from typing import Any, Dict, Literal, Union

from ..core import PipelineConfig
from ..exceptions import PipelineError
from .context import PipelineContext
from .stages import (
    DoclingExportStage,
    ExportStage,
    ExtractionStage,
    GraphConversionStage,
    InputNormalizationStage,
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

    def __init__(self, config: PipelineConfig, mode: Literal["cli", "api"] = "api") -> None:
        """
        Initialize orchestrator with configuration.

        Args:
            config: Pipeline configuration
            mode: Execution mode - "cli" or "api"
        """
        self.config = config
        self.mode = mode

        # Auto-detect dump_to_disk based on mode if not explicitly set
        if config.dump_to_disk is None:
            # CLI mode: dump by default
            # API mode: don't dump by default
            self.dump_to_disk = mode == "cli"
        else:
            # User explicitly set dump_to_disk
            self.dump_to_disk = config.dump_to_disk

        # Core stages (always executed)
        self.stages: list[PipelineStage] = [
            InputNormalizationStage(mode=mode),
            TemplateLoadingStage(),
            ExtractionStage(),
            GraphConversionStage(),
        ]

        # Export stages (conditional based on dump_to_disk)
        if self.dump_to_disk:
            self.stages.extend(
                [
                    DoclingExportStage(),
                    ExportStage(),
                    VisualizationStage(),
                ]
            )

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


def run_pipeline(
    config: Union[PipelineConfig, Dict[str, Any]], mode: Literal["cli", "api"] = "api"
) -> PipelineContext:
    """
    Run the extraction and graph conversion pipeline.

    This is the main entry point for pipeline execution. It accepts either
    a PipelineConfig object or a dictionary of configuration parameters.

    Args:
        config: Pipeline configuration as PipelineConfig or dict
        mode: Execution mode - "cli" for CLI invocations, "api" for Python API (default: "api")

    Returns:
        PipelineContext containing:
            - knowledge_graph: NetworkX DiGraph with extracted entities and relationships
            - extracted_models: List of Pydantic models from extraction
            - graph_metadata: Statistics about the generated graph
            - docling_document: Original DoclingDocument (if available)

    Raises:
        PipelineError: If pipeline execution fails

    Note:
        File exports are controlled by the dump_to_disk parameter:
        - None (default): CLI mode exports files, API mode doesn't
        - True: Force file exports regardless of mode
        - False: Disable file exports regardless of mode

    Example (API mode - no exports):
        >>> from docling_graph import run_pipeline
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "backend": "llm",
        ...     "inference": "remote"
        ... }
        >>> context = run_pipeline(config)
        >>> graph = context.knowledge_graph
        >>> models = context.extracted_models

    Example (API mode - with exports):
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "dump_to_disk": True,
        ...     "output_dir": "my_exports"
        ... }
        >>> context = run_pipeline(config)

    Example (CLI mode):
        >>> # Called internally by CLI
        >>> run_pipeline(config, mode="cli")
    """
    if isinstance(config, dict):
        config = PipelineConfig(**config)

    orchestrator = PipelineOrchestrator(config, mode=mode)
    return orchestrator.run()
