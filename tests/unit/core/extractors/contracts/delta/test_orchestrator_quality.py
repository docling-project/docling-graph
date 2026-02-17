import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.orchestrator import (
    DeltaOrchestrator,
    DeltaOrchestratorConfig,
)


class Child(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class Root(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    children: list[Child] = Field(default_factory=list)


def _dummy_llm(**kwargs: Any) -> dict[str, Any]:
    return {"nodes": [], "relationships": []}


def test_quality_gate_adaptive_parent_lookup_tolerance() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_max_parent_lookup_miss=0,
            quality_adaptive_parent_lookup=True,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1, "children[]": 5},
        merge_stats={"parent_lookup_miss": 2, "attached_node_count": 6},
        normalizer_stats={},
        property_sparsity={},
    )
    assert ok is True
    assert "parent_lookup_miss" not in reasons


def test_quality_gate_strict_parent_lookup_without_adaptive_mode() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_max_parent_lookup_miss=0,
            quality_adaptive_parent_lookup=False,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1, "children[]": 2},
        merge_stats={"parent_lookup_miss": 1, "attached_node_count": 2},
        normalizer_stats={},
        property_sparsity={},
    )
    assert ok is False
    assert "parent_lookup_miss" in reasons


def test_from_dict_uses_delta_quality_keys() -> None:
    conf = DeltaOrchestratorConfig.from_dict(
        {
            "delta_quality_require_root": False,
            "delta_quality_min_instances": 3,
            "delta_quality_max_parent_lookup_miss": 7,
            "delta_quality_adaptive_parent_lookup": False,
            "delta_quality_min_non_empty_properties": 5,
            "delta_quality_min_root_non_empty_fields": 2,
            "delta_quality_min_non_empty_by_path": {"children[]": 2},
            "delta_quality_max_orphan_ratio": 0.2,
            "delta_quality_max_canonical_duplicates": 3,
            "delta_batch_split_max_retries": 2,
        }
    )
    assert conf.quality_require_root is False
    assert conf.quality_min_instances == 3
    assert conf.quality_max_parent_lookup_miss == 7
    assert conf.quality_adaptive_parent_lookup is False
    assert conf.quality_min_non_empty_properties == 5
    assert conf.quality_min_root_non_empty_fields == 2
    assert conf.quality_min_non_empty_by_path == {"children[]": 2}
    assert conf.quality_max_orphan_ratio == 0.2
    assert conf.quality_max_canonical_duplicates == 3
    assert conf.batch_split_max_retries == 2


def test_quality_gate_requires_structural_attachments_when_enabled() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_require_structural_attachments=True,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1, "children[]": 1},
        merge_stats={
            "attached_list_items": 0,
            "attached_scalar_items": 0,
            "parent_lookup_miss": 1,
            "missing_parent_descriptor": 0,
            "attached_node_count": 2,
        },
        normalizer_stats={},
        property_sparsity={},
    )
    assert ok is False
    assert "missing_structural_attachments" in reasons


def test_quality_gate_requires_relationships_when_enabled() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_require_relationships=True,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1},
        merge_stats={
            "attached_list_items": 1,
            "attached_scalar_items": 0,
            "attached_node_count": 2,
        },
        normalizer_stats={},
        property_sparsity={},
        relationship_count=0,
    )
    assert ok is False
    assert "missing_relationships" in reasons


def test_trace_includes_diagnostic_samples() -> None:
    traces: list[dict] = []

    def llm_with_mixed_paths(**kwargs: Any) -> dict[str, Any]:
        return {
            "nodes": [
                {"path": "document", "ids": {}, "properties": {"document_number": "INV-1"}},
                {
                    "path": "Root/children/0",
                    "ids": {"name": "Alice"},
                    "properties": {"name": "Alice"},
                },
            ],
            "relationships": [],
        }

    orchestrator = DeltaOrchestrator(
        llm_call_fn=llm_with_mixed_paths,
        template=Root,
        config=DeltaOrchestratorConfig(quality_require_root=False, quality_min_instances=0),
        on_trace=traces.append,
    )
    _ = orchestrator.extract(
        chunks=["chunk"],
        chunk_metadata=[{"token_count": 10, "page_numbers": [1]}],
        context="test",
    )
    assert traces
    diagnostics = traces[-1]["diagnostics"]
    assert "unknown_path_examples" in diagnostics
    assert "top_missing_id_paths" in diagnostics
    assert "property_sparsity" in diagnostics


def test_quality_gate_rejects_sparse_outputs_when_thresholds_enabled() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_min_non_empty_properties=3,
            quality_min_root_non_empty_fields=2,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1},
        merge_stats={
            "parent_lookup_miss": 0,
            "attached_list_items": 0,
            "attached_scalar_items": 0,
            "attached_node_count": 1,
        },
        normalizer_stats={},
        property_sparsity={
            "total_non_empty_properties": 1,
            "root_non_empty_fields": 1,
            "non_empty_properties_by_path": {"": 1},
        },
        relationship_count=0,
    )
    assert ok is False
    assert "insufficient_non_empty_properties" in reasons
    assert "insufficient_root_fields" in reasons


def test_quality_gate_rejects_when_path_coverage_below_threshold() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_min_non_empty_by_path={"children[]": 2},
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1, "children[]": 1},
        merge_stats={
            "parent_lookup_miss": 0,
            "attached_list_items": 0,
            "attached_scalar_items": 0,
            "attached_node_count": 2,
        },
        normalizer_stats={},
        property_sparsity={
            "total_non_empty_properties": 3,
            "root_non_empty_fields": 1,
            "non_empty_properties_by_path": {"": 1, "children[]": 1},
        },
        relationship_count=0,
    )
    assert ok is False
    assert "insufficient_path_fields:children[]" in reasons


def test_quality_gate_rejects_orphan_ratio_and_duplicate_canonical_ids() -> None:
    orchestrator = DeltaOrchestrator(
        llm_call_fn=_dummy_llm,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=True,
            quality_min_instances=1,
            quality_max_orphan_ratio=0.2,
            quality_max_canonical_duplicates=0,
        ),
    )
    ok, reasons = orchestrator._quality_gate(
        merged_root={"document_number": "INV-1"},
        path_counts={"": 1, "children[]": 2},
        merge_stats={
            "parent_lookup_miss": 0,
            "attached_list_items": 1,
            "attached_scalar_items": 0,
            "attached_node_count": 2,
        },
        normalizer_stats={},
        property_sparsity={
            "total_non_empty_properties": 3,
            "root_non_empty_fields": 1,
            "non_empty_properties_by_path": {"": 1, "children[]": 2},
        },
        relationship_count=1,
        orphan_ratio=0.5,
        canonical_duplicates_total=1,
    )
    assert ok is False
    assert "orphan_ratio_exceeded" in reasons
    assert "canonical_identity_duplicates" in reasons


def test_extract_retries_failed_batch_by_splitting() -> None:
    traces: list[dict[str, Any]] = []

    def llm_fail_large_batch_then_pass(**kwargs: Any) -> dict[str, Any] | list[Any]:
        prompt = kwargs.get("prompt", {})
        user_prompt = ""
        if isinstance(prompt, dict):
            user_prompt = str(prompt.get("user", ""))
        if "chunk-alpha" in user_prompt and "chunk-beta" in user_prompt:
            return []
        return {
            "nodes": [
                {
                    "path": "",
                    "ids": {"document_number": "INV-1"},
                    "properties": {"document_number": "INV-1"},
                }
            ],
            "relationships": [],
        }

    orchestrator = DeltaOrchestrator(
        llm_call_fn=llm_fail_large_batch_then_pass,
        template=Root,
        config=DeltaOrchestratorConfig(
            quality_require_root=False,
            quality_min_instances=0,
            batch_split_max_retries=1,
        ),
        on_trace=traces.append,
    )
    output = orchestrator.extract(
        chunks=["chunk-alpha", "chunk-beta"],
        chunk_metadata=[
            {"token_count": 5, "page_numbers": [1]},
            {"token_count": 5, "page_numbers": [2]},
        ],
        context="split-retry-test",
    )
    assert output is not None
    diagnostics = traces[-1]["diagnostics"]
    assert diagnostics["batch_split_retries"] == 1
    assert diagnostics["split_batch_sizes"] == [[1, 1]]
    # With global_context, the second sub-batch prompt may still contain the first chunk, so mock can fail it (split_failures 1).
    assert diagnostics["split_failures"] in (0, 1)


def test_extract_writes_debug_json_when_debug_dir_set() -> None:
    """When debug_dir is set, extract() writes delta_trace.json, delta_merged_graph.json, delta_merged_output.json."""

    def llm_ok(**kwargs: Any) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "path": "",
                    "ids": {"document_number": "D1"},
                    "properties": {"document_number": "D1"},
                }
            ],
            "relationships": [],
        }

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = DeltaOrchestrator(
            llm_call_fn=llm_ok,
            template=Root,
            config=DeltaOrchestratorConfig(
                quality_require_root=True,
                quality_min_instances=1,
            ),
            debug_dir=tmp,
        )
        result = orchestrator.extract(
            chunks=["chunk"],
            chunk_metadata=None,
            context="debug-test",
        )
        assert result is not None
        assert (Path(tmp) / "delta_trace.json").exists()
        assert (Path(tmp) / "delta_merged_graph.json").exists()
        assert (Path(tmp) / "delta_merged_output.json").exists()
