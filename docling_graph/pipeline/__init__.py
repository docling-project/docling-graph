"""
Pipeline module for document extraction and graph conversion.

This module provides a modular, stage-based pipeline architecture for
processing documents, extracting structured data, and converting to
knowledge graphs.

The pipeline consists of independent stages:
- Template Loading: Load and validate Pydantic templates
- Extraction: Extract structured data from documents
- Docling Export: Export Docling document outputs
- Graph Conversion: Convert extracted data to knowledge graphs
- Export: Export graphs in multiple formats
- Visualization: Generate reports and interactive visualizations

Example:
    >>> from docling_graph.pipeline import run_pipeline
    >>> config = {
    ...     "source": "document.pdf",
    ...     "template": "my_templates.MyTemplate",
    ...     "backend": "llm",
    ...     "inference": "remote"
    ... }
    >>> run_pipeline(config)
"""

from .context import PipelineContext
from .orchestrator import PipelineOrchestrator, run_pipeline
from .stages import (
    DoclingExportStage,
    ExportStage,
    ExtractionStage,
    GraphConversionStage,
    PipelineStage,
    TemplateLoadingStage,
    VisualizationStage,
)

__all__ = [
    "DoclingExportStage",
    "ExportStage",
    "ExtractionStage",
    "GraphConversionStage",
    "PipelineContext",
    "PipelineOrchestrator",
    "PipelineStage",
    "TemplateLoadingStage",
    "VisualizationStage",
    "run_pipeline",
]

# Made with Bob
