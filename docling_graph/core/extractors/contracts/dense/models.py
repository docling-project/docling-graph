"""Pydantic models for dense extraction Phase 1 (skeleton) LLM output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_ids_to_str(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: str(v) for k, v in value.items()}


class DenseParentRef(BaseModel):
    """Parent reference in skeleton output."""

    path: str = ""
    ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}


class DenseSkeletonNode(BaseModel):
    """Skeleton node: path, ids, parent, optional ancestry. No properties."""

    path: str
    ids: dict[str, str] = Field(default_factory=dict)
    parent: DenseParentRef | None = None
    ancestry: list[DenseParentRef] | None = None

    @field_validator("parent", mode="before")
    @classmethod
    def _parent_from_string_or_dict(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return {"path": v.strip() or "", "ids": {}}
        return v

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}

    @field_validator("ancestry", mode="before")
    @classmethod
    def _ancestry_to_list(cls, v: Any) -> list[dict[str, Any]] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return v
        return None


class DenseSkeletonGraph(BaseModel):
    """Phase 1 LLM output: list of skeleton nodes."""

    nodes: list[DenseSkeletonNode] = Field(default_factory=list)
