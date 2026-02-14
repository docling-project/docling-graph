"""Pydantic models for the delta extraction contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_ids_to_str(value: Any) -> dict[str, str]:
    """Coerce id values to string so LLM-returned numbers (e.g. line_number: 1) validate."""
    if not isinstance(value, dict):
        return {}
    return {k: str(v) for k, v in value.items()}


class DeltaParentRef(BaseModel):
    """Reference to a parent node instance in flat graph IR."""

    path: str = ""
    ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}


class DeltaNode(BaseModel):
    """Node payload extracted per batch."""

    path: str
    node_type: str | None = None
    ids: dict[str, str] = Field(default_factory=dict)
    parent: DeltaParentRef | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parent", mode="before")
    @classmethod
    def _parent_from_string_or_dict(cls, v: Any) -> Any:
        """Accept parent as string (e.g. \"BillingDocument\") and coerce to DeltaParentRef shape."""
        if v is None:
            return None
        if isinstance(v, str):
            return {"path": v.strip() or "", "ids": {}}
        return v

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}


class DeltaRelationship(BaseModel):
    """Optional explicit relationship extracted per batch."""

    edge_label: str
    source_path: str
    source_ids: dict[str, str] = Field(default_factory=dict)
    target_path: str
    target_ids: dict[str, str] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}


class DeltaGraph(BaseModel):
    """Flat graph IR returned by each delta extraction batch."""

    nodes: list[DeltaNode] = Field(default_factory=list)
    relationships: list[DeltaRelationship] = Field(default_factory=list)
