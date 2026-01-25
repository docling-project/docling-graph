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
from .pipeline.context import PipelineContext
from .pipeline.orchestrator import run_pipeline as _run_pipeline


def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineContext:
    """
    Run the extraction and graph conversion pipeline.

    This is the main entry point for pipeline execution via Python API.
    By default, files are NOT exported to disk - all data is returned
    in the PipelineContext for programmatic use.

    Pipeline stages:
    1. Input Normalization: Detect and validate input type
    2. Template Loading: Load and validate Pydantic templates
    3. Extraction: Extract structured data from documents
    4. Graph Conversion: Convert extracted data to knowledge graphs
    5. Export (optional): Export files if dump_to_disk=True
    6. Visualization (optional): Generate reports if dump_to_disk=True

    Args:
        config: Pipeline configuration as PipelineConfig or dict

    Returns:
        PipelineContext containing:
            - knowledge_graph: NetworkX directed graph
            - extracted_models: List of Pydantic models
            - graph_metadata: Graph statistics and metadata
            - docling_document: Original document (if available)

    Raises:
        PipelineError: If pipeline execution fails
        ConfigurationError: If configuration is invalid
        ExtractionError: If document extraction fails

    Note:
        When using the Python API (this function), files are NOT exported to disk
        by default. Set dump_to_disk=True in the config to enable file exports.

    Example (default - no file exports):
        >>> from docling_graph import run_pipeline
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "backend": "llm",
        ...     "inference": "remote"
        ... }
        >>> context = run_pipeline(config)
        >>>
        >>> # Access the knowledge graph
        >>> graph = context.knowledge_graph
        >>> print(f"Nodes: {graph.number_of_nodes()}")
        >>> print(f"Edges: {graph.number_of_edges()}")
        >>>
        >>> # Access extracted models
        >>> for model in context.extracted_models:
        ...     print(model)

    Example (with file exports):
        >>> config = {
        ...     "source": "document.pdf",
        ...     "template": "my_templates.MyTemplate",
        ...     "dump_to_disk": True,  # Enable file exports
        ...     "output_dir": "my_exports"
        ... }
        >>> context = run_pipeline(config)
        >>> # Files written to my_exports/ AND data returned
    """
    return _run_pipeline(config, mode="api")


__all__ = ["run_pipeline"]
