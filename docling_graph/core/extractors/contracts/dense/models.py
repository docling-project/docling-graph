"""Pydantic models for dense extraction Phase 1 (skeleton) LLM output.

The skeleton contract uses batch-local integer handles: each node carries a
handle ``i`` and references its parent by ``p`` (the parent's handle in the
same response). Copying a one-or-two digit integer is far more reliable for
small models than re-writing parent identifier strings, and it removes the
repeated {path, ids} parent/ancestry objects that used to dominate output
tokens. Handles are resolved to (path, ids) parent references immediately
after parsing; across batches, nodes are still linked by canonical ids.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_ids_to_str(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {k: str(v) for k, v in value.items()}


def _coerce_handle(value: Any) -> int | None:
    """Accept integer handles emitted as strings (e.g. '2')."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


class DenseParentRef(BaseModel):
    """Explicit parent reference; tolerated fallback when a model emits one instead of a handle."""

    path: str = ""
    ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}


class DenseSkeletonNode(BaseModel):
    """Skeleton node: handle ``i``, catalog path, short ids, parent handle ``p``.

    ``c`` is an optional self-reported chunk attribution: the ``--- CHUNK N ---``
    marker number the entity was found under. When it maps to a chunk of the
    current batch it narrows the node's source-chunk provenance from batch
    granularity to one chunk, sharpening the merge rescue rungs that reason
    about locality; an absent or out-of-batch value falls back to the batch set.
    """

    i: int | None = None
    path: str
    ids: dict[str, str] = Field(default_factory=dict)
    p: int | None = None
    c: int | None = None
    parent: DenseParentRef | None = None

    @field_validator("i", "p", "c", mode="before")
    @classmethod
    def _handles_to_int(cls, v: Any) -> int | None:
        return _coerce_handle(v)

    @field_validator("parent", mode="before")
    @classmethod
    def _parent_from_string_or_dict(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return {"path": v.strip() or "", "ids": {}}
        if isinstance(v, dict):
            return v
        return None

    @field_validator("ids", mode="before")
    @classmethod
    def _ids_to_str(cls, v: Any) -> dict[str, str]:
        return _coerce_ids_to_str(v) if isinstance(v, dict) else {}
