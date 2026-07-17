"""
Template-generation settings: the optional ``templategen:`` config block.

Template generation is a **tool, not a pipeline stage** (design §8): these
settings deliberately live outside :class:`~docling_graph.config.PipelineConfig`,
which would leak them into every convert run's ``metadata.json`` via
``to_metadata_config_dict``. The CLI reads ``config.yaml`` tolerantly and hands
the raw mapping to :func:`load_templategen_settings`; an absent or partial
block falls back to defaults, and unknown keys fail loudly with the list of
valid keys (silent typos would silently un-configure a run).

.. code-block:: yaml

    templategen:
      input_budget_chars: 24000   # cap; effective budget also derived from context_limit
      max_models: 30              # induction cap; overflow reported, never truncated
      max_enum_members: 24
      ontology_depth: 4
      llm_gap_fill: false
      strict: false
      strategy: one-shot          # one-shot (1 LLM call/unit, plain JSON) | three-pass
      workers: 4                  # concurrent induction LLM calls (documents x pass-2 batches)
      max_units: 24               # hard cap on induction units (documents/windows); 0 = unlimited
      max_windows_per_doc: 6      # windows an oversized document may split into
      saturation_stop: true       # stop inducing once the schema stops changing (large corpora)
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TemplateGenSettings", "load_templategen_settings"]


class TemplateGenSettings(BaseModel):
    """Defaults for the ``docling-graph template`` commands (design §8)."""

    model_config = ConfigDict(extra="forbid")

    input_budget_chars: int = Field(
        default=24_000,
        ge=1,
        description="Per-document sampler budget cap in characters (design §4.1); the "
        "effective budget is min(this, budget derived from the model's context_limit).",
    )
    max_models: int = Field(
        default=30,
        ge=1,
        description="Cap on induced/compiled models; overflow is reported, never "
        "silently truncated.",
    )
    max_enum_members: int = Field(
        default=24,
        ge=1,
        description="Enums wider than this demote to str with top values in the description.",
    )
    ontology_depth: int = Field(
        default=4,
        ge=1,
        description="BFS depth bound for the ontology closure (--depth default).",
    )
    llm_gap_fill: bool = Field(
        default=False,
        description="Run the one optional docstrings/examples-only gap-fill LLM call.",
    )
    strict: bool = Field(
        default=False,
        description="Fail instead of auto-repairing lint violations (repairs are "
        "printed either way).",
    )
    strategy: Literal["one-shot", "three-pass"] = Field(
        default="one-shot",
        description="from-docs induction strategy. 'one-shot': one LLM call per unit "
        "returning the full ontology, plain-JSON decoding (maximum model "
        "compatibility, minimum spend). 'three-pass': focused inventory/fields/"
        "relationships passes under strict structured output (more per-field "
        "evidence; needs a model that handles guided decoding). Both run the same "
        "evidence gates, merge, and rulebook repair.",
    )
    workers: int = Field(
        default=4,
        ge=1,
        description="Concurrent LLM calls during from-docs induction (documents in "
        "parallel, pass-2 field batches in parallel within each). 1 disables "
        "concurrency; results are deterministic either way.",
    )
    max_units: int = Field(
        default=24,
        ge=0,
        description="Hard cap on from-docs induction units (a document = 1 unit; an "
        "oversized document splits into up to max_windows_per_doc units). Excess "
        "units are skipped and reported. 0 = unlimited.",
    )
    max_windows_per_doc: int = Field(
        default=6,
        ge=1,
        description="Windows an oversized document (text beyond input_budget_chars) "
        "splits into — evenly spread slices, each inducted as its own unit and "
        "merged back as one document.",
    )
    saturation_stop: bool = Field(
        default=True,
        description="Stop inducing a large corpus (> 10 units) once 6 consecutive "
        "units add no new classes and ~no new fields; skipped units are reported. "
        "The from-docs --exhaustive flag disables this and the max_units cap.",
    )


def load_templategen_settings(config: dict[str, Any] | None) -> TemplateGenSettings:
    """Read the optional ``templategen:`` block from a loaded config mapping.

    Tolerant by design: ``config`` may be ``None`` (no ``config.yaml``), miss
    the ``templategen`` key entirely, or carry a partial block — all yield
    defaults for whatever is absent. Unknown keys raise a ``ValueError``
    listing the valid keys, and wrong-typed values propagate pydantic's
    ``ValidationError``.

    Args:
        config: The full loaded config mapping (e.g. from ``load_config``), or
            ``None`` when no config file exists.

    Returns:
        The validated settings.

    Raises:
        ValueError: The ``templategen`` block is not a mapping, or carries
            unknown keys.
    """
    block = None if config is None else config.get("templategen")
    if block is None:
        return TemplateGenSettings()
    if not isinstance(block, Mapping):
        raise ValueError(
            f"The 'templategen' config block must be a mapping, got {type(block).__name__}"
        )
    valid_keys = sorted(TemplateGenSettings.model_fields)
    unknown = sorted(set(block) - set(valid_keys))
    if unknown:
        raise ValueError(
            f"Unknown templategen setting(s) {unknown}. Valid keys: {', '.join(valid_keys)}"
        )
    return TemplateGenSettings(**dict(block))
