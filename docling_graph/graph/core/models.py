"""Data models for graph components."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class Edge(BaseModel):
    """Model representing a graph edge with metadata."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: str = Field(..., description="Edge label/relationship type")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge properties"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False


class GraphMetadata(BaseModel):
    """Metadata about a generated graph."""

    node_count: int = Field(..., description="Total number of nodes")
    edge_count: int = Field(..., description="Total number of edges")
    node_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of nodes by type/label"
    )
    edge_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of edges by type/label"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Graph creation timestamp"
    )
    source_models: int = Field(
        ...,
        description="Number of source Pydantic models"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
