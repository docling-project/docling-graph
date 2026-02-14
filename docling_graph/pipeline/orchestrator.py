"""
Pipeline orchestrator for coordinating stage execution.

This module provides the main orchestrator that coordinates the execution
of pipeline stages, handles errors, and manages resource cleanup.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal, Union

from .. import __version__
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
from .trace import EventTrace

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
        # Start timing the entire pipeline
        pipeline_start_time = time.time()

        context = PipelineContext(config=self.config)
        current_stage = None

        # Initialize in-memory trace when debug is enabled (CLI --debug or PipelineConfig(debug=True))
        if self.config.debug:
            context.trace_data = EventTrace()
            context.trace_data.emit(
                "pipeline_started",
                "pipeline",
                {
                    "mode": self.mode,
                    "source": str(self.config.source),
                    "processing_mode": self.config.processing_mode,
                    "backend": self.config.backend,
                    "inference": self.config.inference,
                    "debug": self.config.debug,
                },
            )

        # Initialize OutputDirectoryManager if dumping to disk
        if self.dump_to_disk:
            from ..core.utils.output_manager import OutputDirectoryManager

            source_filename = Path(self.config.source).name if self.config.source else "output"
            context.output_manager = OutputDirectoryManager(
                base_output_dir=Path(self.config.output_dir), source_filename=source_filename
            )
            logger.info(f"Output directory: {context.output_manager.get_document_dir()}")

        logger.info("--- Starting Docling-Graph Pipeline ---")

        try:
            for stage in self.stages:
                current_stage = stage
                logger.info(f">>> Stage: {stage.name()}")
                context = stage.execute(context)

            # Calculate total processing time
            pipeline_end_time = time.time()
            pipeline_processing_time = pipeline_end_time - pipeline_start_time

            # Save metadata.json if output_manager is available
            if context.output_manager and self.dump_to_disk:
                from datetime import datetime

                # Determine actual model and provider used
                actual_model = self.config.model_override
                actual_provider = self.config.provider_override

                # If no overrides, get from models config based on backend and inference
                if not actual_model or not actual_provider:
                    if self.config.backend == "llm":
                        if self.config.inference == "local":
                            actual_model = actual_model or self.config.models.llm.local.model
                            actual_provider = (
                                actual_provider or self.config.models.llm.local.provider
                            )
                        else:  # remote
                            actual_model = actual_model or self.config.models.llm.remote.model
                            actual_provider = (
                                actual_provider or self.config.models.llm.remote.provider
                            )
                    else:  # vlm
                        actual_model = actual_model or self.config.models.vlm.local.model
                        actual_provider = actual_provider or self.config.models.vlm.local.provider

                metadata = {
                    "pipeline_version": __version__,
                    "timestamp": datetime.now().isoformat(),
                    "input": {
                        "source": str(self.config.source),
                        "template": str(self.config.template),
                    },
                    "config": {
                        "pipeline": {
                            "processing_mode": self.config.processing_mode,
                            "debug": self.config.debug,
                            "reverse_edges": self.config.reverse_edges,
                            "docling": self.config.docling_config,
                        },
                        "extraction": {
                            "backend": self.config.backend,
                            "inference": self.config.inference,
                            "model": actual_model,
                            "provider": actual_provider,
                            "use_chunking": self.config.use_chunking,
                            "max_batch_size": self.config.max_batch_size,
                            "extraction_contract": getattr(
                                self.config, "extraction_contract", "direct"
                            ),
                            "staged_tuning_preset": getattr(
                                self.config, "staged_tuning_preset", "standard"
                            ),
                            "staged_pass_retries": getattr(
                                self.config, "staged_pass_retries", None
                            ),
                            "parallel_workers": getattr(self.config, "parallel_workers", None),
                            "staged_nodes_fill_cap": getattr(
                                self.config, "staged_nodes_fill_cap", None
                            ),
                            "staged_id_shard_size": getattr(
                                self.config, "staged_id_shard_size", None
                            ),
                        },
                    },
                    "processing_time_seconds": round(pipeline_processing_time, 2),
                    "results": {
                        "nodes": context.graph_metadata.node_count if context.graph_metadata else 0,
                        "edges": context.graph_metadata.edge_count if context.graph_metadata else 0,
                        "extracted_models": len(context.extracted_models)
                        if context.extracted_models
                        else 0,
                        "staged_passes_count": (
                            len(context.trace_data.find_events("staged_trace_emitted"))
                            if context.trace_data is not None
                            else 0
                        ),
                    },
                }

                context.output_manager.save_metadata(metadata)
                logger.info(
                    f"Saved metadata to {context.output_manager.get_document_dir() / 'metadata.json'}"
                )

            if context.trace_data is not None:
                context.trace_data.emit(
                    "pipeline_finished",
                    "pipeline",
                    {
                        "processing_time_seconds": round(pipeline_processing_time, 2),
                        "nodes": context.graph_metadata.node_count if context.graph_metadata else 0,
                        "edges": context.graph_metadata.edge_count if context.graph_metadata else 0,
                        "extracted_models": len(context.extracted_models)
                        if context.extracted_models
                        else 0,
                    },
                )

            # Export trace data to debug dir when debug is on and we're writing to disk
            if context.trace_data and context.output_manager and self.dump_to_disk:
                import json

                from .trace import event_trace_to_jsonable

                debug_dir = context.output_manager.get_debug_dir()
                trace_path = debug_dir / "trace_data.json"
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(
                        event_trace_to_jsonable(context.trace_data),
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                logger.info(f"Saved trace data to {trace_path}")

            # Log final output directory after all exports complete
            if context.output_manager and self.dump_to_disk:
                from rich import print as rich_print

                output_dir = context.output_manager.get_document_dir()
                rich_print(
                    f"[green]→[/green] Saved conversion results to [green]{output_dir}[/green]"
                )

            logger.info(
                f"--- Pipeline Completed Successfully (took {pipeline_processing_time:.2f}s) ---"
            )
            return context

        except Exception as e:
            stage_name = current_stage.name() if current_stage else "Unknown"
            logger.error(f"Pipeline failed at stage: {stage_name}")
            if context.trace_data is not None:
                context.trace_data.emit(
                    "pipeline_failed",
                    "pipeline",
                    {
                        "stage": stage_name,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )

            # Cleanup empty output directory if dump_to_disk was enabled
            if self.dump_to_disk and context.output_manager:
                try:
                    if context.output_manager.cleanup_if_empty():
                        logger.info(
                            f"Removed empty output directory: {context.output_manager.get_document_dir()}"
                        )
                    else:
                        logger.info(
                            f"Kept output directory with partial results: {context.output_manager.get_document_dir()}"
                        )
                except Exception as cleanup_error:
                    # Don't let cleanup errors mask the original error
                    logger.warning(f"Failed to cleanup output directory: {cleanup_error}")

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
