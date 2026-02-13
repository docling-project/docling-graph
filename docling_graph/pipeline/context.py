"""
Pipeline context for sharing state between stages.

This module defines the PipelineContext dataclass that carries state
through the pipeline execution, allowing stages to communicate and
share data without tight coupling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import networkx as nx
from pydantic import BaseModel

from ..core import PipelineConfig
from ..core.converters.models import GraphMetadata
from ..core.extractors.extractor_base import BaseExtractor
from .trace import EventTrace


@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline stages.

    This context object carries all state through the pipeline execution,
    allowing stages to access and modify shared data. Each stage receives
    the context, performs its work, and returns the updated context.

    Attributes:
        config: Pipeline configuration
        template: Loaded Pydantic template class
        extractor: Document extractor instance
        extracted_models: List of extracted Pydantic models
        docling_document: Original Docling document
        knowledge_graph: Generated NetworkX graph
        graph_metadata: Graph statistics and metadata
        output_dir: Output directory path
        node_registry: Shared node ID registry for deterministic IDs
        normalized_source: Normalized input ready for extraction
        input_metadata: Processing hints from input normalization
        input_type: Detected input type
        output_manager: Output directory manager for unified structure
        trace_data: Event trace for debugging (populated only when config.debug is True)
    """

    config: PipelineConfig
    template: type[BaseModel] | None = None
    extractor: BaseExtractor | None = None
    extracted_models: list[BaseModel] | None = None
    docling_document: Any | None = None
    knowledge_graph: nx.DiGraph | None = None
    graph_metadata: GraphMetadata | None = None
    output_dir: Path | None = None
    node_registry: Any | None = None

    # Input normalization fields
    normalized_source: Union[str, Path, Any] | None = None
    input_metadata: Dict[str, Any] | None = None
    input_type: Any | None = None  # InputType enum, but avoid circular import

    # Output management and trace data (NEW)
    output_manager: Any | None = None  # OutputDirectoryManager, avoid circular import
    trace_data: EventTrace | None = None

    def __post_init__(self) -> None:
        """Initialize node registry if not provided."""
        if self.node_registry is None:
            from ..core.converters.node_id_registry import NodeIDRegistry

            self.node_registry = NodeIDRegistry()
