"""
Trace data classes for capturing intermediate pipeline data.

This module defines dataclasses for capturing detailed trace information
during pipeline execution, useful for debugging and analysis.
"""

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
from pydantic import BaseModel


@dataclass
class PageData:
    """Data captured for a single page during document processing."""

    page_number: int
    text_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkData:
    """Data captured for a single chunk during document chunking."""

    chunk_id: int
    page_numbers: list[int]
    text_content: str
    token_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionData:
    """Data captured for a single extraction operation."""

    extraction_id: int
    source_type: Literal["page", "chunk"]
    source_id: int
    parsed_model: BaseModel | None
    extraction_time: float
    error: str | None = None


@dataclass
class GraphData:
    """Data captured for an intermediate graph (per-page or per-chunk)."""

    graph_id: int
    source_type: Literal["page", "chunk"]
    source_id: int
    graph: nx.DiGraph
    pydantic_model: BaseModel
    node_count: int
    edge_count: int


@dataclass
class ConsolidationData:
    """Data captured during graph consolidation/merging."""

    strategy: Literal["llm", "programmatic"]
    input_graph_ids: list[int]
    merge_conflicts: list[dict] | None = None


@dataclass
class TraceData:
    """
    Complete trace data for pipeline execution.

    This contains all intermediate data captured during pipeline execution,
    useful for debugging, analysis, and understanding the extraction process.
    """

    pages: list[PageData] = field(default_factory=list)
    chunks: list[ChunkData] | None = None
    extractions: list[ExtractionData] = field(default_factory=list)
    intermediate_graphs: list[GraphData] = field(default_factory=list)
    consolidation: ConsolidationData | None = None
