"""Data models for graph components."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class Edge(BaseModel):
    """Model representing a graph edge with metadata."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: str = Field(..., description="Edge label/relationship type")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional edge properties"
    )


class GraphMetadata(BaseModel):
    """Metadata about a generated graph."""

    model_config = ConfigDict(frozen=True)

    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    node_types: Dict[str, int] = Field(
        default_factory=dict, description="Count of nodes by type/label"
    )
    edge_types: Dict[str, int] = Field(
        default_factory=dict, description="Count of edges by type/label"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Graph creation timestamp"
    )
    source_models: int = Field(..., description="Number of source Pydantic models")
    # Average degree of nodes in the graph
    average_degree: float | None = Field(
        default=None, description="Average degree of nodes in the graph"
    )
