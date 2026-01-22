"""
Main extraction and graph conversion pipeline.

This module provides the public API for running the document extraction
and graph conversion pipeline. The actual implementation has been refactored
into a modular stage-based architecture in the pipeline/ submodule.

For the new modular implementation, see:
- pipeline/context.py: PipelineContext dataclass
- pipeline/stages.py: Individual pipeline stages
- pipeline/orchestrator.py: Pipeline orchestration logic
"""

from typing import Any, Dict, Union

from .core import PipelineConfig
from .pipeline.orchestrator import run_pipeline as _run_pipeline


def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> None:
    """
    Run the extraction and graph conversion pipeline.

    This is the main entry point for pipeline execution. The pipeline has been
    refactored into modular stages for better maintainability and testability.

    Pipeline stages:
    1. Template Loading: Load and validate Pydantic templates
    2. Extraction: Extract structured data from documents
    3. Docling Export: Export Docling document outputs (optional)
    4. Graph Conversion: Convert extracted data to knowledge graphs
    5. Export: Export graphs in multiple formats (CSV, Cypher, JSON)
    6. Visualization: Generate reports and interactive visualizations

    Args:
        config: Pipeline configuration as PipelineConfig or dict

    Raises:
        PipelineError: If pipeline execution fails
        ConfigurationError: If configuration is invalid
        ExtractionError: If document extraction fails

    Example:
        >>> from docling_graph import run_pipeline
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "backend": "llm",
        ...     "inference": "remote",
        ...     "output_dir": "outputs"
        ... }
        >>> run_pipeline(config)
    """
    _run_pipeline(config)


__all__ = ["run_pipeline"]

# Made with Bob
