"""Merge policy — the single source of truth for ``docling-graph merge`` defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class MergePolicy(BaseModel):
    """Deterministic knobs of a graph merge (CLI flags map 1:1 onto fields).

    Attributes:
        precedence: Duplicate-group fold order. ``input-order`` folds in argv
            order (first graph is base); ``richest`` pre-sorts each duplicate
            group by (attribute richness desc, provenance weight desc, input
            index asc). Both are total orders, so both are deterministic.
        conflicts: Scalar conflict policy. ``keep-first`` keeps the survivor
            value and records the conflict; ``keep-all`` additionally stores
            suppressed values in a ``__conflicts__`` node attribute.
        combine_fields: Text fields merged with sentence-level dedup instead
            of first-wins (the exact set the many-to-one merge passes).
        description_max_length: Truncation bound for combined text fields.
        rekey: Recompute node IDs from identity attributes before folding.
            ``None`` resolves automatically: on iff an id-fields source is
            available (template, format-v2 export, or dense ledger).
        alias_decisions: JSON file with human-confirmed alias candidates from
            a previous merge report (replaces the LLM as confirmer).
        export_format: Extra export written beside graph.json.
        strict_template_check: Refuse to merge inputs whose template schema
            hashes differ (downgrade to a warning when False).
        dry_run: Compute the full merge plan but write only merge_report.json.
    """

    precedence: Literal["input-order", "richest"] = "input-order"
    conflicts: Literal["keep-first", "keep-all"] = "keep-first"
    combine_fields: set[str] = Field(default_factory=lambda: {"description", "summary"})
    description_max_length: int = 4096
    rekey: bool | None = None
    alias_decisions: Path | None = None
    export_format: Literal["csv", "cypher"] | None = None
    strict_template_check: bool = True
    dry_run: bool = False
