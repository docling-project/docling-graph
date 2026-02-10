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
    metadata: dict = field(default_factory=dict)


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
class StagedPassData:
    """Data captured for a single staged extraction pass."""

    pass_id: int
    stage_name: str
    stage_type: Literal["skeleton", "group", "repair"]
    success: bool
    attempts: int
    errors: list[str]
    duration_seconds: float
    fields_requested: list[str]
    fields_returned: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class ConflictResolutionData:
    """Data captured for an LLM conflict-resolution fallback."""

    trigger: Literal["unresolved_conflicts", "failed_pass_partial", "quality_stalled"]
    conflict_fields: list[str]
    success: bool
    duration_seconds: float
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TraceData:
    """
    Complete trace data for pipeline execution.

    This contains all intermediate data captured during pipeline execution,
    useful for debugging, analysis, and understanding the extraction process.
    Populated only when config.debug is True.
    """

    pages: list[PageData] = field(default_factory=list)
    chunks: list[ChunkData] | None = None
    extractions: list[ExtractionData] = field(default_factory=list)
    intermediate_graphs: list[GraphData] = field(default_factory=list)
    consolidation: ConsolidationData | None = None
    staged_passes: list[StagedPassData] = field(default_factory=list)
    conflict_resolutions: list[ConflictResolutionData] = field(default_factory=list)


def trace_data_to_jsonable(trace_data: TraceData, max_text_len: int = 2000) -> dict:
    """
    Convert TraceData to a JSON-serializable dict for export.

    Large text fields are truncated; graphs are exported as summaries only
    (node_count, edge_count) to keep file size manageable.
    """

    def _truncate(s: str) -> str:
        if len(s) <= max_text_len:
            return s
        return s[:max_text_len] + f"... [truncated, total {len(s)} chars]"

    pages_out = [
        {
            "page_number": p.page_number,
            "text_content": _truncate(p.text_content),
            "metadata": p.metadata,
        }
        for p in trace_data.pages
    ]

    chunks_out: list[dict] | None = None
    if trace_data.chunks:
        chunks_out = [
            {
                "chunk_id": c.chunk_id,
                "page_numbers": c.page_numbers,
                "text_content": _truncate(c.text_content),
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in trace_data.chunks
        ]

    extractions_out = []
    for e in trace_data.extractions:
        parsed = None
        if e.parsed_model is not None and hasattr(e.parsed_model, "model_dump"):
            parsed = e.parsed_model.model_dump()
        extractions_out.append(
            {
                "extraction_id": e.extraction_id,
                "source_type": e.source_type,
                "source_id": e.source_id,
                "parsed_model": parsed,
                "extraction_time": e.extraction_time,
                "error": e.error,
                "metadata": e.metadata,
            }
        )

    intermediate_graphs_out = [
        {
            "graph_id": g.graph_id,
            "source_type": g.source_type,
            "source_id": g.source_id,
            "node_count": g.node_count,
            "edge_count": g.edge_count,
        }
        for g in trace_data.intermediate_graphs
    ]

    consolidation_out: dict | None = None
    if trace_data.consolidation is not None:
        c = trace_data.consolidation
        consolidation_out = {
            "strategy": c.strategy,
            "input_graph_ids": c.input_graph_ids,
            "merge_conflicts": c.merge_conflicts,
        }

    staged_passes_out = [
        {
            "pass_id": p.pass_id,
            "stage_name": p.stage_name,
            "stage_type": p.stage_type,
            "success": p.success,
            "attempts": p.attempts,
            "errors": p.errors,
            "duration_seconds": p.duration_seconds,
            "fields_requested": p.fields_requested,
            "fields_returned": p.fields_returned,
            "metadata": p.metadata,
        }
        for p in trace_data.staged_passes
    ]

    conflict_resolutions_out = [
        {
            "trigger": r.trigger,
            "conflict_fields": r.conflict_fields,
            "success": r.success,
            "duration_seconds": r.duration_seconds,
            "error": r.error,
            "metadata": r.metadata,
        }
        for r in trace_data.conflict_resolutions
    ]

    return {
        "pages": pages_out,
        "chunks": chunks_out,
        "extractions": extractions_out,
        "intermediate_graphs": intermediate_graphs_out,
        "consolidation": consolidation_out,
        "staged_passes": staged_passes_out,
        "conflict_resolutions": conflict_resolutions_out,
    }
