"""Tests for delta runtime module (contract-local implementation).

Covers DeltaOrchestrator and DeltaOrchestratorConfig from runtime.py so that
patch coverage includes this module when it is added or changed.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.runtime import (
    DeltaOrchestrator,
    DeltaOrchestratorConfig,
)


class _Child(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class _Root(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    children: list[_Child] = Field(default_factory=list)


def _dummy_llm(**kwargs: Any) -> dict[str, Any]:
    return {"nodes": [], "relationships": []}


def test_runtime_config_from_dict() -> None:
    """DeltaOrchestratorConfig.from_dict from runtime uses delta_ keys."""
    conf = DeltaOrchestratorConfig.from_dict(
        {
            "delta_quality_min_instances": 2,
            "delta_resolvers_mode": "fuzzy",
            "delta_resolver_fuzzy_threshold": 0.9,
        }
    )
    assert conf.quality_min_instances == 2
    assert conf.resolvers.mode == "fuzzy"
    assert conf.resolvers.fuzzy_threshold == 0.9


def test_runtime_config_from_dict_invalid_resolver_mode_falls_back_to_off() -> None:
    """Invalid delta_resolvers_mode becomes 'off'."""
    conf = DeltaOrchestratorConfig.from_dict({"delta_resolvers_mode": "invalid"})
    assert conf.resolvers.mode == "off"


def test_runtime_quality_gate_adaptive_parent_lookup() -> None:
    """DeltaOrchestrator from runtime: _quality_gate with adaptive parent lookup."""
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=_Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_max_parent_lookup_miss=0,
            quality_adaptive_parent_lookup=True,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "D1"},
        path_counts={"": 1, "children[]": 3},
        merge_stats={"parent_lookup_miss": 2, "attached_node_count": 4},
        normalizer_stats={},
        property_sparsity={},
    )
    assert ok is True
    assert "parent_lookup_miss" not in reasons


def test_runtime_extract_returns_merged_root_when_llm_returns_valid_graph() -> None:
    """DeltaOrchestrator.extract from runtime returns merged root when LLM returns valid delta."""
    def llm_return_one_root(**kwargs: Any) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "path": "",
                    "ids": {"document_number": "R1"},
                    "properties": {"document_number": "R1"},
                }
            ],
            "relationships": [],
        }

    orchestrator = DeltaOrchestrator(
        llm_call_fn=llm_return_one_root,
        template=_Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
        ),
    )
    result = orchestrator.extract(
        chunks=["Chunk text."],
        chunk_metadata=None,
        context="test",
    )
    assert result is not None
    assert result.get("document_number") == "R1"


def test_runtime_extract_returns_none_when_no_batch_results() -> None:
    """DeltaOrchestrator.extract from runtime returns None when LLM returns non-dict (no batches)."""
    def llm_return_non_dict(**kwargs: Any) -> list:
        return []  # Not a dict -> batch result not added

    orchestrator = DeltaOrchestrator(
        llm_call_fn=llm_return_non_dict,
        template=_Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
        ),
    )
    result = orchestrator.extract(
        chunks=["Chunk."],
        chunk_metadata=None,
        context="test",
    )
    assert result is None
