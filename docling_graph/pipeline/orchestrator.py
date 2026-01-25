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

        # Auto-detect include_trace based on mode if not explicitly set
        if config.include_trace is None:
            # CLI mode: include trace by default
            # API mode: don't include trace by default (memory-efficient)
            self.include_trace = mode == "cli"
        else:
            # User explicitly set include_trace
            self.include_trace = config.include_trace

        # Core stages (always executed)
        self.stages: list[PipelineStage] = [
            InputNormalizationStage(mode=mode),
            TemplateLoadingStage(),
            ExtractionStage(),
            GraphConversionStage(),
        ]

        # Export stages (conditional based on dump_to_disk)
        if self.dump_to_disk:
            # Add TraceExportStage before other exports if trace is enabled
            if self.include_trace:
                from .stages import TraceExportStage

                self.stages.append(TraceExportStage())

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

        # Initialize OutputDirectoryManager if dumping to disk
        if self.dump_to_disk:
            from pathlib import Path

            from ..core.utils.output_manager import OutputDirectoryManager

            source_filename = Path(self.config.source).name if self.config.source else "output"
            context.output_manager = OutputDirectoryManager(
                base_output_dir=Path(self.config.output_dir), source_filename=source_filename
            )
            logger.info(f"Output directory: {context.output_manager.get_document_dir()}")

        # Initialize TraceData if trace is enabled
        if self.include_trace:
            from .context import TraceData

            context.trace_data = TraceData()
            logger.info("Trace data collection enabled")

        logger.info("--- Starting Docling-Graph Pipeline ---")

        try:
            for stage in self.stages:
                current_stage = stage
                logger.info(f">>> Stage: {stage.name()}")
                context = stage.execute(context)

            # Log trace data summary if enabled
            if self.include_trace and context.trace_data:
                logger.info("Trace data collection complete")
                logger.info(f"  Pages: {len(context.trace_data.pages)}")
                logger.info(
                    f"  Chunks: {len(context.trace_data.chunks) if context.trace_data.chunks else 0}"
                )
                logger.info(f"  Extractions: {len(context.trace_data.extractions)}")
                logger.info(f"  Intermediate graphs: {len(context.trace_data.intermediate_graphs)}")

            # Save metadata.json if output_manager is available
            if context.output_manager and self.dump_to_disk:
                from datetime import datetime
                
                metadata = {
                    "pipeline_version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "source": str(self.config.source),
                    "template": str(self.config.template),
                    "processing_mode": self.config.processing_mode,
                    "backend": self.config.backend,
                    "docling_config": self.config.docling_config,
                    "use_chunking": self.config.use_chunking,
                    "llm_consolidation": self.config.llm_consolidation,
                    "include_trace": self.include_trace,
                    "results": {
                        "nodes": context.graph_metadata.node_count if context.graph_metadata else 0,
                        "edges": context.graph_metadata.edge_count if context.graph_metadata else 0,
                        "extracted_models": len(context.extracted_models) if context.extracted_models else 0,
                    }
                }
                
                if self.include_trace and context.trace_data:
                    metadata["trace_summary"] = {
                        "pages": len(context.trace_data.pages),
                        "chunks": len(context.trace_data.chunks) if context.trace_data.chunks else 0,
                        "extractions": len(context.trace_data.extractions),
                        "intermediate_graphs": len(context.trace_data.intermediate_graphs),
                    }
                
                context.output_manager.save_metadata(metadata)
                logger.info(f"Saved metadata to {context.output_manager.get_document_dir() / 'metadata.json'}")

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
