"""
Unit tests for dense extraction orchestrator (Phase 1 and Phase 2, including parallel).
"""

import os
from typing import Any

import pytest

from docling_graph.core.extractors.contracts.dense.catalog import NodeCatalog, NodeSpec
from docling_graph.core.extractors.contracts.dense.models import DenseSkeletonNode
from docling_graph.core.extractors.contracts.dense.orchestrator import (
    DenseOrchestrator,
    DenseOrchestratorConfig,
    chunk_batches_by_token_limit,
    normalize_skeleton_batch,
    prune_barren_branches,
)
from tests.fixtures.sample_templates.test_template import SampleInvoice


def _skeleton(raw: dict) -> DenseSkeletonNode:
    return DenseSkeletonNode.model_validate(raw)


def test_normalize_skeleton_batch_resolves_parent_handles():
    """A node's p handle resolves to the referenced node's (path, ids)."""
    allowed = {"", "studies[]", "studies[].experiments[]"}
    nodes = [
        _skeleton({"i": 1, "path": "", "ids": {}}),
        _skeleton({"i": 2, "path": "studies[]", "ids": {"study_id": "S1"}, "p": 1}),
        _skeleton({"i": 3, "path": "studies[].experiments[]", "ids": {"exp_id": "E1"}, "p": 2}),
    ]
    out = normalize_skeleton_batch(nodes, allowed)
    assert out[0]["parent"] is None
    assert out[1]["parent"] == {"path": "", "ids": {}}
    assert out[2]["parent"] == {"path": "studies[]", "ids": {"study_id": "S1"}}


def test_normalize_skeleton_batch_accepts_string_handles_and_parent_fallback():
    """String handles are coerced; an explicit parent object works when p is absent."""
    allowed = {"", "studies[]"}
    nodes = [
        _skeleton({"i": "1", "path": "", "ids": {}}),
        _skeleton({"i": "2", "path": "studies[]", "ids": {"study_id": "S1"}, "p": "1"}),
        _skeleton(
            {
                "path": "studies[]",
                "ids": {"study_id": "S2"},
                "parent": {"path": "", "ids": {}},
            }
        ),
    ]
    out = normalize_skeleton_batch(nodes, allowed)
    assert out[1]["parent"] == {"path": "", "ids": {}}
    assert out[2]["parent"] == {"path": "", "ids": {}}


def test_normalize_skeleton_batch_ignores_dangling_and_self_handles():
    """Unknown or self-referencing p handles yield parent=None instead of garbage."""
    allowed = {"", "studies[]"}
    nodes = [
        _skeleton({"i": 5, "path": "studies[]", "ids": {"study_id": "S1"}, "p": 5}),
        _skeleton({"i": 6, "path": "studies[]", "ids": {"study_id": "S2"}, "p": 99}),
    ]
    out = normalize_skeleton_batch(nodes, allowed)
    assert out[0]["parent"] is None
    assert out[1]["parent"] is None


def test_normalize_skeleton_batch_output_shape_and_source_batch_index():
    """Normalized nodes contain only path/ids/parent (+ _source_batch_index when given)."""
    allowed = {""}
    nodes = [_skeleton({"i": 1, "path": "", "ids": {}})]
    out = normalize_skeleton_batch(nodes, allowed, source_batch_index=2)
    assert set(out[0].keys()) == {"path", "ids", "parent", "_source_batch_index"}
    assert out[0]["_source_batch_index"] == 2
    out_no_idx = normalize_skeleton_batch(nodes, allowed)
    assert set(out_no_idx[0].keys()) == {"path", "ids", "parent"}


def test_normalize_skeleton_batch_canonicalizes_paths_missing_brackets():
    """Model paths without [] map onto catalog paths for both node and handle parents."""
    allowed = {"", "studies[]", "studies[].experiments[]"}
    nodes = [
        _skeleton({"i": 1, "path": "studies", "ids": {"study_id": "S1"}}),
        _skeleton({"i": 2, "path": "studies.experiments", "ids": {"exp_id": "E1"}, "p": 1}),
    ]
    out = normalize_skeleton_batch(nodes, allowed)
    assert out[0]["path"] == "studies[]"
    assert out[1]["path"] == "studies[].experiments[]"
    assert out[1]["parent"]["path"] == "studies[]"


def test_chunk_batches_produces_multiple_batches_for_parallel():
    """With small max_batch_tokens, multiple chunks yield multiple batches."""
    chunks = ["chunk one", "chunk two", "chunk three", "chunk four", "chunk five"]
    token_counts = [40, 40, 40, 40, 40]
    batches = chunk_batches_by_token_limit(chunks, token_counts, max_batch_tokens=50)
    assert len(batches) >= 2


def test_orchestrator_parallel_workers_returns_non_null_root():
    """With parallel_workers=2 and multiple skeleton batches, orchestrator runs and returns a non-None root."""
    # Build chunks so we get multiple skeleton batches (small token limit)
    chunks = ["Document chunk A. " * 20, "Document chunk B. " * 20, "Document chunk C. " * 20]
    token_counts = [50, 50, 50]
    full_markdown = "\n\n".join(chunks)

    skeleton_responses = [
        {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
        for _ in range(5)
    ]
    fill_responses = [
        {
            "items": [
                {
                    "invoice_number": "INV-1",
                    "date": "2024-01-01",
                    "total_amount": 100.0,
                    "vendor_name": "Acme",
                    "items": [],
                }
            ]
        }
    ]
    skeleton_idx = [0]
    fill_idx = [0]

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            i = skeleton_idx[0]
            skeleton_idx[0] += 1
            return skeleton_responses[i % len(skeleton_responses)]
        if "dense_fill" in context:
            i = fill_idx[0]
            fill_idx[0] += 1
            return fill_responses[i % len(fill_responses)]
        return None

    config = DenseOrchestratorConfig(
        parallel_workers=2,
        skeleton_batch_tokens=50,
        fill_nodes_cap=5,
    )
    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=config,
    )
    root = orch.run(
        chunks=chunks,
        chunk_metadata=[{"token_count": tc} for tc in token_counts],
        full_markdown=full_markdown,
        context="test",
    )
    assert root is not None
    assert "invoice_number" in root or "vendor_name" in root


def test_prune_barren_branches_removes_barren_branch_keeps_non_barren():
    """prune_barren_branches removes branch nodes that are childless and barren; leaves non-barren and leaf paths unchanged."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(
                path="",
                node_type="Root",
                id_fields=[],
                parent_path="",
                field_name="",
                is_list=False,
            ),
            NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="",
                field_name="studies",
                is_list=True,
            ),
            NodeSpec(
                path="studies[].experiments[]",
                node_type="Experiment",
                id_fields=["exp_id"],
                parent_path="studies[]",
                field_name="experiments",
                is_list=True,
            ),
        ]
    )
    root = {
        "studies": [
            {"study_id": "S1", "objective": None, "experiments": []},
            {"study_id": "S2", "objective": "Real study", "experiments": []},
        ]
    }
    out = prune_barren_branches(root, catalog)
    assert len(out["studies"]) == 1
    assert out["studies"][0]["study_id"] == "S2"
    assert out["studies"][0]["objective"] == "Real study"


def test_merge_skeleton_batches_accumulates_source_batches():
    """The same node seen in several batches keeps the union of source batch indexes."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_skeleton_batches

    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=["invoice_number"]),
        ]
    )
    node_a = {
        "path": "",
        "ids": {"invoice_number": "INV-1"},
        "parent": None,
        "_source_batch_index": 0,
    }
    node_b = {
        "path": "",
        "ids": {"invoice_number": "INV-1"},
        "parent": None,
        "_source_batch_index": 2,
    }
    merged = merge_skeleton_batches([[node_a], [node_b]], catalog)
    assert len(merged) == 1
    assert merged[0]["_source_batch_indexes"] == [0, 2]
    assert "_source_batch_index" not in merged[0]


def _make_orchestrator(fill_context_mode: str) -> DenseOrchestrator:
    config = DenseOrchestratorConfig(fill_context_mode=fill_context_mode)
    return DenseOrchestrator(
        llm_call_fn=lambda **kwargs: None,
        template=SampleInvoice,
        config=config,
    )


def test_build_fill_context_scoped_uses_only_source_batches():
    """Scoped fill context contains the node's source batches plus the document head."""
    orch = _make_orchestrator("scoped")
    batch_texts = ["BATCH-ZERO " * 50, "BATCH-ONE " * 50, "BATCH-TWO " * 50]
    full_markdown = "\n\n".join(batch_texts)
    descriptors = [{"path": "items[]", "ids": {}, "_source_batch_indexes": [1]}]
    ctx = orch._build_fill_context("items[]", descriptors, batch_texts, full_markdown, "HEAD")
    assert "BATCH-ONE" in ctx
    assert "BATCH-TWO" not in ctx
    assert ctx.startswith("HEAD")
    assert len(ctx) < len(full_markdown)


def test_build_fill_context_root_and_full_mode_use_full_markdown():
    """Root path and full mode always receive the whole document."""
    batch_texts = ["A" * 100, "B" * 100]
    full_markdown = "\n\n".join(batch_texts)
    descriptors = [{"path": "", "ids": {}, "_source_batch_indexes": [1]}]
    scoped = _make_orchestrator("scoped")
    assert (
        scoped._build_fill_context("", descriptors, batch_texts, full_markdown, "H")
        == full_markdown
    )
    full = _make_orchestrator("full")
    assert (
        full._build_fill_context("items[]", descriptors, batch_texts, full_markdown, "H")
        == full_markdown
    )


def test_build_fill_context_without_provenance_falls_back_to_full():
    """Descriptors without source batch info get the full document."""
    orch = _make_orchestrator("scoped")
    batch_texts = ["A" * 100, "B" * 100]
    full_markdown = "\n\n".join(batch_texts)
    descriptors = [{"path": "items[]", "ids": {}}]
    assert (
        orch._build_fill_context("items[]", descriptors, batch_texts, full_markdown, "H")
        == full_markdown
    )


def test_dense_config_from_dict_parses_fill_context():
    """dense_fill_context is parsed and invalid values fall back to scoped."""
    assert (
        DenseOrchestratorConfig.from_dict({"dense_fill_context": "full"}).fill_context_mode
        == "full"
    )
    assert DenseOrchestratorConfig.from_dict({}).fill_context_mode == "scoped"
    assert (
        DenseOrchestratorConfig.from_dict({"dense_fill_context": "bogus"}).fill_context_mode
        == "scoped"
    )


def test_fill_pads_missing_items_so_skeleton_instances_survive():
    """When the fill LLM returns fewer items than instances, missing ones keep their ids."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    skeleton_response = {
        "nodes": [
            {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
            {
                "path": "employees[]",
                "ids": {"email": "a@acme.com"},
                "parent": {"path": "", "ids": {"company_name": "Acme"}},
            },
            {
                "path": "employees[]",
                "ids": {"email": "b@acme.com"},
                "parent": {"path": "", "ids": {"company_name": "Acme"}},
            },
        ]
    }

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return skeleton_response
        if "dense_fill_employees[]" in context:
            # Only one of the two requested instances comes back.
            return {
                "items": [
                    {"email": "a@acme.com", "first_name": "Alice", "last_name": "A"},
                ]
            }
        if "dense_fill" in context:
            return {"items": [{"company_name": "Acme", "industry": "Tools", "founded_year": 1999}]}
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    root = orch.run(
        chunks=["Acme employs Alice and Bob."],
        chunk_metadata=[{"token_count": 10}],
        full_markdown="Acme employs Alice and Bob.",
        context="test",
    )
    assert root is not None
    employees = root.get("employees") or []
    emails = {e.get("email") for e in employees}
    assert emails == {"a@acme.com", "b@acme.com"}


def _company_skeleton_args() -> tuple[Any, set[str], dict[str, Any]]:
    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from tests.fixtures.sample_templates.test_template import SampleCompany

    catalog = build_node_catalog(SampleCompany)
    allowed = set(catalog.paths())
    spec_by_path = {s.path: s for s in catalog.nodes}
    return SampleCompany, allowed, spec_by_path


def test_skeleton_batch_splits_on_truncation_and_recovers_nodes():
    """A truncated multi-chunk skeleton batch is split so dropped nodes are recovered."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        user = prompt["user"]
        n_chunks = user.count("--- CHUNK")
        if n_chunks >= 2:
            # Full batch overflows the model output budget: only the first node survives.
            if _diagnostics_out is not None:
                _diagnostics_out["truncated"] = True
            return {
                "nodes": [
                    {
                        "path": "employees[]",
                        "ids": {"email": "alice@x.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    }
                ]
            }
        # Single-chunk sub-batches fit and each returns its own node.
        if "bob" in user.lower():
            return {
                "nodes": [
                    {
                        "path": "employees[]",
                        "ids": {"email": "bob@x.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    }
                ]
            }
        return {
            "nodes": [
                {
                    "path": "employees[]",
                    "ids": {"email": "alice@x.com"},
                    "parent": {"path": "", "ids": {"company_name": "Acme"}},
                }
            ]
        }

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(0, "Alice email alice@x.com", 40), (1, "Bob email bob@x.com", 40)]
    _, nodes = orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    emails = {n["ids"].get("email") for n in nodes}
    assert emails == {"alice@x.com", "bob@x.com"}


def test_skeleton_single_chunk_truncation_does_not_split():
    """A single-chunk batch cannot be split further; the partial result is kept once."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    calls = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        if _diagnostics_out is not None:
            _diagnostics_out["truncated"] = True
        return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(0, "single chunk only", 40)]
    _, nodes = orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    # one LLM call (max_pass_retries default 1 -> one validated result, no split)
    assert len(calls) == 1
    assert len(nodes) == 1


def test_multi_chunk_truncation_splits_before_escalating():
    """P2: a multi-chunk batch splits *before* max_tokens escalation is allowed."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    flags: list[bool] = []
    state = {"n": 0}

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        allow_truncation_retry: bool = True,
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        flags.append(allow_truncation_retry)
        state["n"] += 1
        if state["n"] == 1:  # the initial multi-chunk batch truncates
            if _diagnostics_out is not None:
                _diagnostics_out["truncated"] = True
            return {"nodes": []}
        return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(0, "chunk a", 40), (1, "chunk b", 40)]
    orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    # Initial multi-chunk call must NOT permit escalation (split first); the
    # single-chunk sub-batches may escalate.
    assert flags[0] is False
    assert all(f is True for f in flags[1:])
    assert orch._counters.get("split_count", 0) == 1


def test_single_chunk_unrecoverable_truncation_records_dropped_chunk():
    """P2/V3: a single chunk that never recovers is recorded as dropped, not silently lost."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    calls: list[str] = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        if _diagnostics_out is not None:
            _diagnostics_out["truncated"] = True
        return {"nodes": []}  # nothing usable, ever

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(7, "single chunk only", 40)]
    _, nodes = orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    assert nodes == []
    # main skeleton call + one shallow-projection fallback attempt
    assert len(calls) == 2
    assert orch._dropped_chunk_ids == [7]
    assert orch._counters.get("failed_batch_count") == 1


def test_single_chunk_truncation_recovers_via_shallow_fallback():
    """P2: when the full-schema attempt truncates but the shallow (root +
    direct-children) projection succeeds, the recovered node is kept and the
    chunk is NOT recorded as dropped."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    calls: list[str] = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        if len(calls) == 1:  # main (full-schema) attempt truncates, nothing usable
            if _diagnostics_out is not None:
                _diagnostics_out["truncated"] = True
            return {"nodes": []}
        # shallow fallback attempt succeeds
        return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(3, "single chunk only", 40)]
    _, nodes = orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    assert len(nodes) == 1
    assert len(calls) == 2
    assert orch._dropped_chunk_ids == []
    assert orch._counters.get("failed_batch_count", 0) == 0


def test_shallow_skeleton_artifacts_none_without_root_spec():
    """A catalog with no root spec has nothing to fall back to; the shallow
    retry is a no-op rather than raising."""
    from docling_graph.core.extractors.contracts.dense.catalog import NodeSpec

    template_cls, _allowed, _spec_by_path = _company_skeleton_args()
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: {"nodes": []},
        template=template_cls,
        config=DenseOrchestratorConfig(),
    )
    # Simulate a catalog with only nested paths (no root "" spec).
    orch._catalog.nodes = [n for n in orch._catalog.nodes if n.path != ""]
    assert isinstance(orch._catalog.nodes[0], NodeSpec)  # sanity: nodes remain
    assert orch._shallow_skeleton_artifacts() is None
    # Cached: a second call returns the same (still None) result without rebuilding.
    assert orch._shallow_skeleton_artifacts() is None


def test_skeleton_no_split_when_not_truncated():
    """A non-truncated multi-chunk batch is not split."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    calls = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    batch = [(0, "chunk a", 40), (1, "chunk b", 40)]
    orch._run_one_skeleton_batch(
        batch_idx=0,
        batch=batch,
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
        already_found_str=None,
    )
    assert len(calls) == 1


def _linkage_catalog() -> NodeCatalog:
    return NodeCatalog(
        nodes=[
            NodeSpec(
                path="",
                node_type="Root",
                id_fields=["title"],
                parent_path="",
                field_name="",
                is_list=False,
            ),
            NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="",
                field_name="studies",
                is_list=True,
            ),
            NodeSpec(
                path="studies[].experiments[]",
                node_type="Experiment",
                id_fields=["exp_id"],
                parent_path="studies[]",
                field_name="experiments",
                is_list=True,
            ),
        ]
    )


def _desc(path: str, ids: dict, parent: dict | None) -> dict:
    return {"path": path, "ids": ids, "parent": parent}


def test_merge_skeleton_batches_collapses_duplicate_roots():
    """Paraphrased root ids across batches collapse into a single root."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_skeleton_batches

    catalog = _linkage_catalog()
    r1 = {"path": "", "ids": {"title": "Version A"}, "parent": None, "_source_batch_index": 0}
    r2 = {
        "path": "",
        "ids": {"title": "A completely different phrasing"},
        "parent": None,
        "_source_batch_index": 3,
    }
    study = {
        "path": "studies[]",
        "ids": {"study_id": "S1"},
        "parent": {"path": "", "ids": {}},
        "_source_batch_index": 1,
    }
    merged = merge_skeleton_batches([[r1, study], [r2]], catalog)
    roots = [n for n in merged if n["path"] == ""]
    assert len(roots) == 1
    assert roots[0]["_source_batch_indexes"] == [0, 3]
    assert any(n["path"] == "studies[]" for n in merged)


def test_merge_filled_recovers_child_via_single_parent_instance():
    """A drifted parent reference resolves to the only existing parent instance."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = _linkage_catalog()
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "S1"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [_desc("studies[]", {"study_id": "S1"}, {"path": "", "ids": {}})],
        "studies[].experiments[]": [
            _desc(
                "studies[].experiments[]",
                {"exp_id": "E1"},
                {"path": "studies[]", "ids": {"study_id": "the main study"}},
            )
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    assert root["studies"][0]["experiments"][0]["exp_id"] == "E1"
    assert stats["recovered_single_parent"] == 1
    assert stats["dropped"] == 0


def test_merge_filled_recovers_child_via_fuzzy_parent_match():
    """With several parents, a drifted reference attaches to the unique canonical match."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = _linkage_catalog()
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha-Study"}, {"study_id": "Beta-Study"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            _desc("studies[]", {"study_id": "Alpha-Study"}, {"path": "", "ids": {}}),
            _desc("studies[]", {"study_id": "Beta-Study"}, {"path": "", "ids": {}}),
        ],
        "studies[].experiments[]": [
            _desc(
                "studies[].experiments[]",
                {"exp_id": "E1"},
                {"path": "studies[]", "ids": {"study_id": "alpha"}},
            )
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    alpha = root["studies"][0]
    beta = root["studies"][1]
    assert alpha["experiments"][0]["exp_id"] == "E1"
    assert "experiments" not in beta
    assert stats["recovered_fuzzy"] == 1


def test_merge_filled_creates_placeholder_parent_for_unmatched_reference():
    """An unresolvable-but-identified parent is materialized so the subtree survives."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = _linkage_catalog()
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha-Study"}, {"study_id": "Beta-Study"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            _desc("studies[]", {"study_id": "Alpha-Study"}, {"path": "", "ids": {}}),
            _desc("studies[]", {"study_id": "Beta-Study"}, {"path": "", "ids": {}}),
        ],
        "studies[].experiments[]": [
            _desc(
                "studies[].experiments[]",
                {"exp_id": "E1"},
                {"path": "studies[]", "ids": {"study_id": "Gamma"}},
            )
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    assert len(root["studies"]) == 3
    placeholder = root["studies"][2]
    assert placeholder["study_id"] == "Gamma"
    assert placeholder["experiments"][0]["exp_id"] == "E1"
    assert stats["recovered_placeholder"] == 1
    assert stats["dropped"] == 0


def test_merge_filled_rescues_idless_orphans_into_shared_bucket():
    """Children with no parent ids and ambiguous candidates land in a shared bucket parent."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = _linkage_catalog()
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha-Study"}, {"study_id": "Beta-Study"}],
        "studies[].experiments[]": [{"exp_id": "E1"}, {"exp_id": "E2"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            _desc("studies[]", {"study_id": "Alpha-Study"}, {"path": "", "ids": {}}),
            _desc("studies[]", {"study_id": "Beta-Study"}, {"path": "", "ids": {}}),
        ],
        "studies[].experiments[]": [
            _desc("studies[].experiments[]", {"exp_id": "E1"}, None),
            _desc("studies[].experiments[]", {"exp_id": "E2"}, None),
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    assert stats["dropped"] == 0
    # First orphan creates the bucket; the second reuses it via the lookup hit.
    assert stats["recovered_bucket"] == 1
    assert stats["attached_exact"] >= 1
    bucket = root["studies"][2]
    assert "study_id" not in bucket
    assert {e["exp_id"] for e in bucket["experiments"]} == {"E1", "E2"}


def _linkage_template() -> type:
    """Pydantic counterpart of _linkage_catalog with a REQUIRED study_id."""
    from pydantic import BaseModel, ConfigDict, Field

    class _Experiment(BaseModel):
        model_config = ConfigDict(graph_id_fields=["exp_id"])
        exp_id: str

    class _Study(BaseModel):
        model_config = ConfigDict(graph_id_fields=["study_id"])
        study_id: str
        experiments: list[_Experiment] = Field(default_factory=list)

    class _Doc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["title"])
        title: str | None = None
        studies: list[_Study] = Field(default_factory=list)

    return _Doc


def test_merge_filled_locality_adopts_unique_co_chunk_parent():
    """A child with a dangling parent handle attaches to the single parent
    discovered in the same source chunk — before any bucket is considered."""
    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    template = _linkage_template()
    catalog = build_node_catalog(template)
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha"}, {"study_id": "Beta"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            {**_desc("studies[]", {"study_id": "Alpha"}, None), "_source_chunk_ids": [1]},
            {**_desc("studies[]", {"study_id": "Beta"}, None), "_source_chunk_ids": [5, 6]},
        ],
        "studies[].experiments[]": [
            {
                **_desc("studies[].experiments[]", {"exp_id": "E1"}, None),
                "_source_chunk_ids": [5],
            }
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(
        path_filled, path_descriptors, catalog, stats_out=stats, template=template
    )
    beta = next(s for s in root["studies"] if s["study_id"] == "Beta")
    assert [e["exp_id"] for e in beta.get("experiments", [])] == ["E1"]
    assert stats["recovered_locality"] == 1
    assert stats["dropped"] == 0
    assert stats["recovered_bucket"] == 0


def test_merge_filled_bucket_gated_when_identity_required():
    """With a template whose parent identity is required, an id-less bucket
    parent is never materialized (it would be deleted or blanked downstream);
    the orphan is dropped honestly and counted."""
    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    template = _linkage_template()
    catalog = build_node_catalog(template)
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha"}, {"study_id": "Beta"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            _desc("studies[]", {"study_id": "Alpha"}, None),
            _desc("studies[]", {"study_id": "Beta"}, None),
        ],
        # No parent ids, no source-locality data -> nothing can resolve it.
        "studies[].experiments[]": [_desc("studies[].experiments[]", {"exp_id": "E1"}, None)],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(
        path_filled, path_descriptors, catalog, stats_out=stats, template=template
    )
    assert stats["dropped"] == 1
    assert stats["recovered_bucket"] == 0
    assert len(root["studies"]) == 2  # no phantom third study


def test_merge_filled_bucket_still_allowed_when_identity_optional():
    """Bucket rescue is preserved for templates whose identity fields are
    optional — there the bucket survives validation and keeps the subtree."""
    from pydantic import BaseModel, ConfigDict, Field

    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    class _Experiment(BaseModel):
        model_config = ConfigDict(graph_id_fields=["exp_id"])
        exp_id: str

    class _Study(BaseModel):
        model_config = ConfigDict(graph_id_fields=["study_id"])
        study_id: str | None = None
        experiments: list[_Experiment] = Field(default_factory=list)

    class _Doc(BaseModel):
        title: str | None = None
        studies: list[_Study] = Field(default_factory=list)

    catalog = build_node_catalog(_Doc)
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha"}, {"study_id": "Beta"}],
        "studies[].experiments[]": [{"exp_id": "E1"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "studies[]": [
            _desc("studies[]", {"study_id": "Alpha"}, None),
            _desc("studies[]", {"study_id": "Beta"}, None),
        ],
        "studies[].experiments[]": [_desc("studies[].experiments[]", {"exp_id": "E1"}, None)],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(
        path_filled, path_descriptors, catalog, stats_out=stats, template=_Doc
    )
    assert stats["recovered_bucket"] == 1
    assert stats["dropped"] == 0
    assert len(root["studies"]) == 3


def test_merge_filled_distinct_siblings_with_value_as_key_ids_do_not_collapse():
    """Siblings whose ids use the value as the key (a small-model shape error,
    e.g. {"Alpha": "alpha"} instead of {"study_id": "Alpha"}) must stay distinct
    parents so each keeps its own children — not collapse onto the last sibling.
    Regression for _canonical_lookup_key lacking the raw-id fallback that
    _skeleton_identity_key uses, which silently misattached every child.
    """
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = _linkage_catalog()
    path_filled = {
        "": [{"title": "T"}],
        "studies[]": [{"study_id": "Alpha"}, {"study_id": "Beta"}],
        "studies[].experiments[]": [{"exp_id": "E1"}, {"exp_id": "E2"}],
    }
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        # Malformed: identifier value emitted as the key (no "study_id" key).
        "studies[]": [
            _desc("studies[]", {"Alpha": "Alpha"}, {"path": "", "ids": {}}),
            _desc("studies[]", {"Beta": "Beta"}, {"path": "", "ids": {}}),
        ],
        "studies[].experiments[]": [
            _desc(
                "studies[].experiments[]",
                {"exp_id": "E1"},
                {"path": "studies[]", "ids": {"Alpha": "alpha"}},
            ),
            _desc(
                "studies[].experiments[]",
                {"exp_id": "E2"},
                {"path": "studies[]", "ids": {"Beta": "beta"}},
            ),
        ],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    assert stats["dropped"] == 0
    studies = {s["study_id"]: s for s in root["studies"]}
    # Each study keeps its own experiment; nothing collapses onto the last sibling.
    assert [e["exp_id"] for e in studies["Alpha"].get("experiments", [])] == ["E1"]
    assert [e["exp_id"] for e in studies["Beta"].get("experiments", [])] == ["E2"]


def test_skeleton_batch_keeps_valid_nodes_when_some_entries_invalid():
    """Malformed entries (e.g. echoed schema fragments) are skipped, not the whole batch."""
    template_cls, allowed, _spec_by_path = _company_skeleton_args()

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        return {
            "nodes": [
                {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
                "ids",
                {"type": "string"},
                {
                    "path": "employees[]",
                    "ids": {"email": "a@x.com"},
                    "parent": {"path": "", "ids": {"company_name": "Acme"}},
                },
            ]
        }

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    nodes, truncated = orch._call_skeleton_batch(
        batch_idx=0,
        batch=[(0, "text", 10)],
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        already_found_str=None,
    )
    assert truncated is False
    assert len(nodes) == 2
    assert {n["path"] for n in nodes} == {"", "employees[]"}


def test_apply_skeleton_reconciliation_merges_aliases_and_remaps_parents():
    """Alias instances merge into the specific one; children re-point to it."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        apply_skeleton_reconciliation,
    )

    catalog = _linkage_catalog()
    spec_by_path = {s.path: s for s in catalog.nodes}
    generic = {
        "path": "studies[]",
        "ids": {"study_id": "LFP study"},
        "parent": None,
        "_source_batch_indexes": [0],
    }
    specific = {
        "path": "studies[]",
        "ids": {"study_id": "LFP_20vol_5wtPVDF"},
        "parent": None,
        "_source_batch_indexes": [2],
    }
    child = {
        "path": "studies[].experiments[]",
        "ids": {"exp_id": "E1"},
        "parent": {"path": "studies[]", "ids": {"study_id": "LFP study"}},
        "_source_batch_indexes": [0],
    }
    groups = [{"path": "studies[]", "keep": 1, "merge": [0]}]
    kept, merged_count = apply_skeleton_reconciliation(
        [generic, specific, child], groups, spec_by_path
    )
    assert merged_count == 1
    study_ids = [n["ids"] for n in kept if n["path"] == "studies[]"]
    assert study_ids == [{"study_id": "LFP_20vol_5wtPVDF"}]
    assert specific["_source_batch_indexes"] == [2, 0]
    child_after = next(n for n in kept if n["path"] == "studies[].experiments[]")
    assert child_after["parent"]["ids"] == {"study_id": "LFP_20vol_5wtPVDF"}


def test_apply_skeleton_reconciliation_ignores_invalid_groups():
    """Bad paths, out-of-range indices, and self-merges are skipped harmlessly."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        apply_skeleton_reconciliation,
    )

    catalog = _linkage_catalog()
    spec_by_path = {s.path: s for s in catalog.nodes}
    nodes = [
        {"path": "studies[]", "ids": {"study_id": "S1"}, "parent": None},
        {"path": "studies[]", "ids": {"study_id": "S2"}, "parent": None},
    ]
    groups = [
        {"path": "unknown[]", "keep": 0, "merge": [1]},
        {"path": "studies[]", "keep": 9, "merge": [0]},
        {"path": "studies[]", "keep": 0, "merge": [0, 99]},
        "not a dict",
    ]
    kept, merged_count = apply_skeleton_reconciliation(nodes, groups, spec_by_path)
    assert merged_count == 0
    assert len(kept) == 2


def test_orchestrator_reconciliation_skips_llm_when_no_duplicates():
    """No path with >= 2 instances means no reconciliation LLM call."""
    calls: list[str] = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    skeleton = [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]
    out, merged = orch._run_skeleton_reconciliation(skeleton, spec_by_path, "t")
    assert out is skeleton
    assert merged == 0
    assert calls == []


def test_orchestrator_reconciliation_applies_llm_merge_groups():
    """A valid merges response collapses alias instances."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        assert "dense_reconcile" in context
        return {"merges": [{"path": "employees[]", "keep": 0, "merge": [1]}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    skeleton = [
        {"path": "employees[]", "ids": {"email": "alice.smith@x.com"}, "parent": None},
        {"path": "employees[]", "ids": {"email": "alice@x.com"}, "parent": None},
    ]
    out, merged = orch._run_skeleton_reconciliation(skeleton, spec_by_path, "t")
    assert merged == 1
    assert len(out) == 1
    assert out[0]["ids"] == {"email": "alice.smith@x.com"}


def test_sanitize_filled_restores_unusable_id_values():
    """Skeleton-known identity values replace null/empty/object fill values."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import _sanitize_filled

    spec = NodeSpec(
        path="studies[]",
        node_type="Study",
        id_fields=["study_id"],
        parent_path="",
        field_name="studies",
        is_list=True,
    )
    descriptors = [
        {"ids": {"study_id": "INV-000004"}},
        {"ids": {"study_id": "S2"}},
        {"ids": {"study_id": "S3"}},
    ]
    items = [
        {"study_id": {"value": "INV-000004"}},  # object instead of scalar
        {"study_id": ""},  # empty string
        {"study_id": "s3-refined"},  # usable value from fill is kept
    ]
    out = _sanitize_filled(items, descriptors, spec, None)
    assert out[0]["study_id"] == "INV-000004"
    assert out[1]["study_id"] == "S2"
    assert out[2]["study_id"] == "s3-refined"


def test_run_stats_published_after_run():
    """The orchestrator exposes observability counters after a run."""

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return {"nodes": [{"i": 1, "path": "", "ids": {"invoice_number": "INV-1"}}]}
        if "dense_fill" in context:
            return {
                "items": [
                    {
                        "invoice_number": "INV-1",
                        "date": "d",
                        "total_amount": 1.0,
                        "vendor_name": "v",
                        "items": [],
                    }
                ]
            }
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
    )
    root = orch.run(
        chunks=["doc text"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc text",
        context="t",
    )
    assert root is not None
    stats = orch.last_run_stats
    assert stats["skeleton_nodes"] == 1
    assert stats["truncation_count"] == 0
    assert stats["split_count"] == 0
    assert "retention_pct" in stats
    # V3: honest source-coverage signals alongside merge-only retention.
    assert stats["skeleton_batches_failed"] == 0
    assert stats["dropped_chunk_ids"] == []
    assert stats["chunk_coverage_pct"] == 100.0
    assert stats["parallel_workers"] == 1
    assert "phase1_seconds" in stats and "phase2_seconds" in stats


def test_dense_config_from_dict_parses_dedupe_mode():
    """dense_dedupe maps to dedupe_mode with a safe default for invalid values."""
    assert DenseOrchestratorConfig.from_dict({"dense_dedupe": "off"}).dedupe_mode == "off"
    assert (
        DenseOrchestratorConfig.from_dict({"dense_dedupe": "aggressive"}).dedupe_mode
        == "aggressive"
    )
    assert DenseOrchestratorConfig.from_dict({}).dedupe_mode == "standard"
    assert DenseOrchestratorConfig.from_dict({"dense_dedupe": "bogus"}).dedupe_mode == "standard"


def test_dense_config_from_dict_parses_provenance_mode():
    """provenance maps to provenance_mode; an invalid value falls back to 'standard'."""
    assert DenseOrchestratorConfig.from_dict({"provenance": "off"}).provenance_mode == "off"
    assert (
        DenseOrchestratorConfig.from_dict({"provenance": "detailed"}).provenance_mode == "detailed"
    )
    assert DenseOrchestratorConfig.from_dict({}).provenance_mode == "standard"
    assert (
        DenseOrchestratorConfig.from_dict({"provenance": "nonsense"}).provenance_mode == "standard"
    )


def _run_dedupe_mode(mode: str) -> list[str]:
    """Run a two-employee SampleCompany extraction; return the LLM contexts seen."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    calls: list[str] = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls.append(context)
        if "dense_skeleton" in context:
            return {
                "nodes": [
                    {"i": 1, "path": "", "ids": {"company_name": "Acme"}},
                    {"i": 2, "path": "employees[]", "ids": {"email": "a@x.com"}, "p": 1},
                    {"i": 3, "path": "employees[]", "ids": {"email": "b@x.com"}, "p": 1},
                ]
            }
        if "dense_reconcile" in context:
            return {"merges": []}
        if "dense_fill" in context:
            return {"items": []}
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(dedupe_mode=mode),
    )
    orch.run(
        chunks=["doc"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc",
        context="t",
    )
    return calls


def test_dedupe_off_skips_reconciliation_llm_call():
    """dedupe_mode='off' must not spend an LLM call on reconciliation; 'standard' does."""
    assert not any("dense_reconcile" in c for c in _run_dedupe_mode("off"))
    assert any("dense_reconcile" in c for c in _run_dedupe_mode("standard"))


def test_fill_schema_hoists_defs_to_wrapper_root():
    """The wrapped fill schema must keep $ref pointers valid (defs hoisted to root)."""
    import json

    from tests.fixtures.sample_templates.test_template import SampleCompany

    captured: dict[str, Any] = {}

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        if "dense_fill" in context:
            captured["schema"] = json.loads(schema_json)
        return {"items": []}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec = next(s for s in orch._catalog.nodes if s.path == "employees[]")
    sub_schema = json.dumps(
        {
            "type": "object",
            "properties": {"boss": {"$ref": "#/$defs/Ref"}},
            "$defs": {"Ref": {"type": "string"}},
        }
    )
    orch._run_one_fill_batch(
        path="employees[]",
        spec=spec,
        batch_descriptors=[{"ids": {"email": "a@x.com"}}],
        batch_index=0,
        sub_schema=sub_schema,
        fill_markdown="doc",
        context="t",
    )
    schema = captured["schema"]
    assert "Ref" in schema.get("$defs", {})
    assert "$defs" not in schema["properties"]["items"]["items"]


def test_strip_mislabeled_root_ids_clears_prose_number_field():
    """A root *_number field holding multi-word digit-free prose is cleared."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        strip_mislabeled_root_ids,
    )

    nodes = [
        {"path": "", "ids": {"document_number": "Zylker PC Builds"}, "parent": None},
        {"path": "items[]", "ids": {"line_no": "Some Long Label"}, "parent": {"path": ""}},
    ]
    out = strip_mislabeled_root_ids(nodes)
    # Root mis-capture cleared...
    assert "document_number" not in out[0]["ids"]
    # ...but non-root nodes are never touched (parent linkage must stay intact).
    assert out[1]["ids"]["line_no"] == "Some Long Label"


def test_strip_mislabeled_root_ids_keeps_real_identifiers():
    """Digit-bearing, single-token, or non-number-named ids are preserved."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        strip_mislabeled_root_ids,
    )

    nodes = [
        {
            "path": "",
            "ids": {
                "invoice_number": "INV-2024-001",  # has digits -> keep
                "reference_no": "Contract",  # single token -> keep
                "vendor_name": "Zylker PC Builds",  # not a number field -> keep
            },
            "parent": None,
        }
    ]
    out = strip_mislabeled_root_ids(nodes)
    assert out[0]["ids"] == {
        "invoice_number": "INV-2024-001",
        "reference_no": "Contract",
        "vendor_name": "Zylker PC Builds",
    }


# ============================================================================
# Additional coverage: defensive branches, edge cases, and less-exercised paths
# ============================================================================


def test_strip_mislabeled_root_ids_skips_non_dict_ids():
    """A root node whose 'ids' is not a dict (malformed) is left untouched."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        strip_mislabeled_root_ids,
    )

    nodes = [{"path": "", "ids": "not-a-dict", "parent": None}]
    out = strip_mislabeled_root_ids(nodes)
    assert out[0]["ids"] == "not-a-dict"


def test_skeleton_identity_key_handles_non_dict_ids():
    """A node with non-dict 'ids' falls back to an empty-ids identity key."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        _skeleton_identity_key,
    )

    spec_by_path: dict[str, NodeSpec] = {}
    node = {"path": "studies[]", "ids": "garbage"}
    path, pairs = _skeleton_identity_key(node, spec_by_path)
    assert path == "studies[]"
    # No spec and no usable ids -> falls back to the process-unique key.
    assert pairs[0][0] == "__key"


def test_skeleton_ledger_key_handles_non_dict_ids():
    """_skeleton_ledger_key tolerates a non-dict 'ids' the same way."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        _skeleton_ledger_key,
    )

    spec_by_path: dict[str, NodeSpec] = {}
    node = {"path": "studies[]", "ids": 12345}
    # With no spec and no id fields, identity_key degrades to None (unkeyable).
    assert _skeleton_ledger_key(node, spec_by_path) is None


def test_chunk_batches_rejects_non_positive_max_tokens():
    """max_batch_tokens <= 0 is an invariant violation, not a silent no-op."""
    with pytest.raises(ValueError, match="max_batch_tokens must be > 0"):
        chunk_batches_by_token_limit(["a chunk"], None, max_batch_tokens=0)


def test_chunk_batches_falls_back_to_word_count_when_counts_misaligned():
    """A token_counts list of the wrong length is ignored in favor of a word-count estimate."""
    chunks = ["one two three four", "five six"]
    # Mismatched length (3 counts for 2 chunks) triggers the fallback branch.
    batches = chunk_batches_by_token_limit(chunks, [10, 10, 10], max_batch_tokens=1000)
    assert sum(len(b) for b in batches) == 2


def test_canonical_catalog_path_returns_none_when_no_candidate_matches():
    """A path with no bracket-stripped match against any allowed path is unresolvable."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        _canonical_catalog_path,
    )

    assert _canonical_catalog_path("totally.unknown.path", {"", "studies[]"}) is None


def test_normalize_skeleton_batch_drops_node_with_unresolvable_path():
    """A node whose path cannot be mapped onto the catalog is silently dropped."""
    allowed = {"", "studies[]"}
    nodes = [
        _skeleton({"i": 1, "path": "totally_unknown", "ids": {"a": "b"}}),
        _skeleton({"i": 2, "path": "studies[]", "ids": {"study_id": "S1"}}),
    ]
    out = normalize_skeleton_batch(nodes, allowed)
    assert len(out) == 1
    assert out[0]["path"] == "studies[]"


def test_apply_skeleton_reconciliation_skips_group_whose_keep_node_already_removed():
    """A later group targeting an already-merged-away keep node is skipped, not double-applied."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        apply_skeleton_reconciliation,
    )

    catalog = _linkage_catalog()
    spec_by_path = {s.path: s for s in catalog.nodes}
    a = {"path": "studies[]", "ids": {"study_id": "A"}, "parent": None}
    b = {"path": "studies[]", "ids": {"study_id": "B"}, "parent": None}
    c = {"path": "studies[]", "ids": {"study_id": "C"}, "parent": None}
    # First group merges b into a (removing b). Second group tries to use the
    # now-removed b (index 1) as its "keep" node; it must be skipped harmlessly.
    groups = [
        {"path": "studies[]", "keep": 0, "merge": [1]},
        {"path": "studies[]", "keep": 1, "merge": [2]},
    ]
    kept, merged_count = apply_skeleton_reconciliation([a, b, c], groups, spec_by_path)
    assert merged_count == 1
    ids_left = {n["ids"]["study_id"] for n in kept}
    assert ids_left == {"A", "C"}


def test_merge_filled_into_root_skips_node_whose_catalog_parent_path_is_unknown():
    """A NodeSpec whose declared parent_path is absent from the catalog is skipped entirely."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=["title"]),
            NodeSpec(
                path="orphan[]",
                node_type="Orphan",
                id_fields=["oid"],
                # Declares a parent path that has no corresponding NodeSpec at all.
                parent_path="missing_parent[]",
                field_name="orphans",
                is_list=True,
            ),
        ]
    )
    path_filled = {"": [{"title": "T"}], "orphan[]": [{"oid": "O1"}]}
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "orphan[]": [_desc("orphan[]", {"oid": "O1"}, {"path": "", "ids": {}})],
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    # "parent_path not in spec_by_path" guard skips the whole path -> never attached.
    assert "orphans" not in root
    assert stats.get("dropped", 0) == 0


def test_merge_filled_into_root_depth_guard_stops_infinite_placeholder_recursion():
    """A pathologically deep parent chain hits the recursion depth guard and drops cleanly."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    # Build a chain of 10 nested paths so the placeholder-creation recursion
    # exceeds the depth=8 guard before reaching the root.
    nodes = [
        NodeSpec(path="", node_type="Root", id_fields=["title"], parent_path="", field_name="")
    ]
    parent_path = ""
    for level in range(10):
        path = "/".join(["lvl"] * (level + 1)) + "[]"
        nodes.append(
            NodeSpec(
                path=path,
                node_type=f"Level{level}",
                id_fields=["id"],
                parent_path=parent_path,
                field_name="child",
                is_list=True,
            )
        )
        parent_path = path
    catalog = NodeCatalog(nodes=nodes)
    deepest_path = parent_path
    path_filled = {deepest_path: [{"id": "leaf"}]}
    path_descriptors = {
        deepest_path: [
            _desc(deepest_path, {"id": "leaf"}, {"path": "/".join(["lvl"] * 9) + "[]", "ids": {}})
        ]
    }
    stats: dict[str, int] = {}
    root = merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    # The deep chain of id-less placeholders exceeds depth 8: unresolvable, dropped.
    assert stats["dropped"] == 1
    assert root == {}


def test_merge_filled_into_root_skips_instance_with_empty_field_name():
    """A NodeSpec with an empty field_name (malformed catalog) is skipped, not attached."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import merge_filled_into_root

    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=["title"]),
            NodeSpec(
                path="orphan[]",
                node_type="Orphan",
                id_fields=["oid"],
                parent_path="",
                field_name="",  # malformed: no attachment field
                is_list=True,
            ),
        ]
    )
    path_filled = {"": [{"title": "T"}], "orphan[]": [{"oid": "O1"}]}
    path_descriptors = {
        "": [_desc("", {"title": "T"}, None)],
        "orphan[]": [_desc("orphan[]", {"oid": "O1"}, {"path": "", "ids": {}})],
    }
    root = merge_filled_into_root(path_filled, path_descriptors, catalog)
    assert "orphan" not in root


def test_prune_barren_branches_keeps_branch_with_children_even_if_scalars_empty():
    """A branch node that HAS children is never pruned, regardless of its own scalar fields."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[]),
            NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="",
                field_name="studies",
                is_list=True,
            ),
            NodeSpec(
                path="studies[].experiments[]",
                node_type="Experiment",
                id_fields=["exp_id"],
                parent_path="studies[]",
                field_name="experiments",
                is_list=True,
            ),
        ]
    )
    root = {
        "studies": [
            {
                "study_id": "S1",
                "objective": None,
                "experiments": [{"exp_id": "E1"}],
            }
        ]
    }
    out = prune_barren_branches(root, catalog)
    # Has a child -> kept even though "objective" is None (barren-looking otherwise).
    assert len(out["studies"]) == 1
    assert out["studies"][0]["experiments"][0]["exp_id"] == "E1"


def test_prune_barren_branches_records_pruned_event():
    """events_out receives a 'pruned' event with the id fields of the removed branch."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[]),
            NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="",
                field_name="studies",
                is_list=True,
            ),
            NodeSpec(
                path="studies[].experiments[]",
                node_type="Experiment",
                id_fields=["exp_id"],
                parent_path="studies[]",
                field_name="experiments",
                is_list=True,
            ),
        ]
    )
    root = {"studies": [{"study_id": "S1", "objective": None, "experiments": []}]}
    events: list[dict[str, Any]] = []
    out = prune_barren_branches(root, catalog, events_out=events)
    assert out["studies"] == []
    assert len(events) == 1
    assert events[0]["event"] == "pruned"
    assert events[0]["path"] == "studies[]"
    assert events[0]["ids"] == {"study_id": "S1"}


def test_prune_barren_branches_recurses_into_non_list_dict_child():
    """A singular (non-list) branch child is recursed into via the dict-value path."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[]),
            NodeSpec(
                path="summary",
                node_type="Summary",
                id_fields=[],
                parent_path="",
                field_name="summary",
                is_list=False,
            ),
            NodeSpec(
                path="summary.detail[]",
                node_type="Detail",
                id_fields=["detail_id"],
                parent_path="summary",
                field_name="detail",
                is_list=True,
            ),
            # A grandchild makes "summary.detail[]" itself a branch path, so
            # pruning actually applies to its (barren) list entries.
            NodeSpec(
                path="summary.detail[].sub[]",
                node_type="Sub",
                id_fields=["sub_id"],
                parent_path="summary.detail[]",
                field_name="sub",
                is_list=True,
            ),
        ]
    )
    root = {"summary": {"detail": [{"detail_id": "D1", "note": None, "sub": []}]}}
    out = prune_barren_branches(root, catalog)
    # "summary" is recursed into via the non-list dict-value path (elif branch);
    # its "detail" list is then pruned because the entry is barren and childless.
    assert out["summary"]["detail"] == []


def test_is_usable_id_value_accepts_bool_and_numeric():
    """Booleans and numeric types count as usable identity values (not just strings)."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import _is_usable_id_value

    assert _is_usable_id_value(True) is True
    assert _is_usable_id_value(0) is True
    assert _is_usable_id_value(3.14) is True
    assert _is_usable_id_value("  ") is False
    assert _is_usable_id_value(None) is False


def test_call_skeleton_batch_retries_when_every_entry_invalid():
    """When every node entry fails validation, the pass is retried before giving up."""
    template_cls, allowed, _spec_by_path = _company_skeleton_args()
    calls = {"n": 0}

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        calls["n"] += 1
        # Every entry is malformed (missing required 'path'/'ids' shape).
        return {"nodes": ["garbage", 123]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=template_cls,
        config=DenseOrchestratorConfig(max_pass_retries=1),
    )
    nodes, truncated = orch._call_skeleton_batch(
        batch_idx=0,
        batch=[(0, "text", 10)],
        total_batches=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        already_found_str=None,
    )
    assert nodes == []
    assert truncated is False
    # Initial attempt + one retry (max_pass_retries=1) = 2 calls.
    assert calls["n"] == 2


def test_shallow_skeleton_retry_returns_empty_without_root_spec():
    """The shallow terminal fallback is a clean no-op when there is no root to fall back to."""
    template_cls, _allowed, _spec_by_path = _company_skeleton_args()
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: {"nodes": []},
        template=template_cls,
        config=DenseOrchestratorConfig(),
    )
    orch._catalog.nodes = [n for n in orch._catalog.nodes if n.path != ""]
    result = orch._shallow_skeleton_retry(0, [(0, "text", 10)], "t")
    assert result == []


def test_dedupe_skeleton_proposes_containment_and_keeps_both_without_llm_confirm():
    """Containment pairs are traced as PROPOSALS and reach the reconciliation
    prompt as candidates; when the (mocked) LLM declines, both instances survive
    — the tier-destroying auto-merge is gone."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    traced: list[dict[str, Any]] = []
    prompts_seen: list[dict[str, str]] = []

    def fake_llm(**kwargs: Any) -> dict[str, Any]:
        prompts_seen.append(kwargs.get("prompt") or {})
        return {"merges": []}

    orch = DenseOrchestrator(
        llm_call_fn=fake_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(dedupe_mode="standard"),
        on_trace=traced.append,
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    # Two employees whose ids are a containment pair (base vs. superset string).
    skeleton = [
        {"path": "employees[]", "ids": {"email": "alice@x.com"}, "parent": None},
        {"path": "employees[]", "ids": {"email": "alice@x.com extra"}, "parent": None},
    ]
    kept, merged = orch._dedupe_skeleton(skeleton, spec_by_path, "t")
    assert len(kept) == 2  # LLM declined -> nothing merged
    assert merged == 0
    kinds = {key for t in traced for key in t if key != "contract"}
    assert "phase1_containment_proposals" in kinds
    assert any("CONTAINMENT CANDIDATES" in (p.get("user") or "") for p in prompts_seen)


def test_dedupe_skeleton_applies_llm_confirmed_containment_merge():
    """A containment candidate the LLM confirms is merged via reconciliation."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def fake_llm(**_kwargs: Any) -> dict[str, Any]:
        return {"merges": [{"path": "employees[]", "keep": 0, "merge": [1]}]}

    orch = DenseOrchestrator(
        llm_call_fn=fake_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(dedupe_mode="standard"),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    skeleton = [
        {"path": "employees[]", "ids": {"email": "alice@x.com"}, "parent": None},
        {"path": "employees[]", "ids": {"email": "alice@x.com extra"}, "parent": None},
    ]
    kept, merged = orch._dedupe_skeleton(skeleton, spec_by_path, "t")
    assert merged == 1
    assert len(kept) == 1
    assert kept[0]["ids"] == {"email": "alice@x.com"}


def test_write_debug_writes_json_file(tmp_path):
    """_write_debug creates the debug dir and writes JSON when debug_dir is set."""
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
        debug_dir=str(tmp_path / "debug_out"),
    )
    orch._write_debug("sample.json", {"a": 1})
    written = tmp_path / "debug_out" / "sample.json"
    assert written.exists()
    assert written.read_text(encoding="utf-8") == '{\n  "a": 1\n}'


def test_write_debug_is_noop_without_debug_dir():
    """_write_debug does nothing when no debug_dir was configured."""
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
    )
    # Should not raise even though no debug_dir exists.
    orch._write_debug("sample.json", {"a": 1})


def test_freeze_ledger_assigns_unkeyed_key_for_idless_node():
    """A node with no usable ids gets a positional '#unkeyedN' ledger key, never a false binding."""
    from docling_graph.core.provenance import ChunkRecord
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    merged_skeleton = [
        {"path": "employees[]", "ids": {}, "parent": None, "_source_chunk_ids": [0]},
    ]
    chunk_records = {
        0: ChunkRecord(chunk_id=0, batch_index=0, token_count=5, text_hash="h", char_length=4)
    }
    ledger = orch._freeze_ledger(merged_skeleton, chunk_records, spec_by_path)
    assert len(ledger.nodes) == 1
    entry = next(iter(ledger.nodes.values()))
    assert entry.identity_key.startswith("employees[]#unkeyed")
    assert "identity:unkeyed" in entry.notes


def test_freeze_ledger_falls_back_to_batch_level_chunks_when_source_chunks_missing():
    """When _source_chunk_ids is absent, anchors fall back to all chunks in the source batches."""
    from docling_graph.core.provenance import ChunkRecord
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    merged_skeleton = [
        {
            "path": "employees[]",
            "ids": {"email": "a@x.com"},
            "parent": None,
            "_source_batch_indexes": [0],
            # No _source_chunk_ids: must fall back to batch-level chunk lookup.
        },
    ]
    chunk_records = {
        0: ChunkRecord(chunk_id=0, batch_index=0, token_count=5, text_hash="h", char_length=4),
        1: ChunkRecord(chunk_id=1, batch_index=1, token_count=5, text_hash="h", char_length=4),
    }
    ledger = orch._freeze_ledger(merged_skeleton, chunk_records, spec_by_path)
    entry = next(iter(ledger.nodes.values()))
    anchored_chunk_ids = {a.chunk_id for a in entry.anchors}
    assert anchored_chunk_ids == {0}


def test_freeze_ledger_records_reconciled_anchors_and_merged_from():
    """Reconciled chunk ids become 'reconciled' anchors; merged_from lineage is recorded."""
    from docling_graph.core.provenance import ChunkRecord
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    merged_skeleton = [
        {
            "path": "employees[]",
            "ids": {"email": "a@x.com"},
            "parent": None,
            "_source_chunk_ids": [0],
            "_reconciled_chunk_ids": [1],
            "_merged_from": ["employees[]#email=alt@x.com"],
        },
    ]
    chunk_records = {
        0: ChunkRecord(chunk_id=0, batch_index=0, token_count=5, text_hash="h", char_length=4),
        1: ChunkRecord(chunk_id=1, batch_index=0, token_count=5, text_hash="h", char_length=4),
    }
    ledger = orch._freeze_ledger(merged_skeleton, chunk_records, spec_by_path)
    entry = next(iter(ledger.nodes.values()))
    kinds_by_chunk = {a.chunk_id: a.kind for a in entry.anchors}
    assert kinds_by_chunk == {0: "observed", 1: "reconciled"}
    assert entry.merged_from == ["employees[]#email=alt@x.com"]


def test_apply_merge_events_marks_placeholder_reused_when_entry_has_no_anchors():
    """A second 'synthetic' event for an existing, still-anchorless entry appends a reuse note."""
    from docling_graph.core.provenance import NodeProvenance, ProvenanceLedger, identity_key
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    ids = {"email": "a@x.com"}
    key = identity_key("employees[]", ids, spec_by_path["employees[]"].id_fields)
    assert key is not None
    entry = NodeProvenance(
        identity_key=key,
        catalog_path="employees[]",
        ids=ids,
        anchors=[],  # no anchors yet -> a second synthetic event should append a reuse note
    )
    ledger = ProvenanceLedger(node_level=True, chunks={}, nodes={key: entry})
    events = [{"event": "synthetic", "path": "employees[]", "ids": ids}]
    orch._apply_merge_events(ledger, events, spec_by_path)
    assert "merge:placeholder-reused" in ledger.nodes[key].notes


def test_apply_merge_events_marks_dropped_entry():
    """A 'dropped' merge event flags the matching ledger entry and appends a note."""
    from docling_graph.core.provenance import NodeProvenance, ProvenanceLedger, identity_key
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    ids = {"email": "a@x.com"}
    key = identity_key("employees[]", ids, spec_by_path["employees[]"].id_fields)
    assert key is not None
    entry = NodeProvenance(identity_key=key, catalog_path="employees[]", ids=ids)
    ledger = ProvenanceLedger(node_level=True, chunks={}, nodes={key: entry})
    events = [{"event": "dropped", "path": "employees[]", "ids": ids}]
    orch._apply_merge_events(ledger, events, spec_by_path)
    assert ledger.nodes[key].dropped is True
    assert "merge:dropped" in ledger.nodes[key].notes


def test_apply_merge_events_creates_new_synthetic_entry_when_absent():
    """A 'synthetic' event with no existing ledger entry creates a fresh synthetic one."""
    from docling_graph.core.provenance import ProvenanceLedger
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    ledger = ProvenanceLedger(node_level=True, chunks={}, nodes={})
    events = [{"event": "synthetic", "path": "employees[]", "ids": {"email": "a@x.com"}}]
    orch._apply_merge_events(ledger, events, spec_by_path)
    assert len(ledger.nodes) == 1
    entry = next(iter(ledger.nodes.values()))
    assert entry.synthetic is True
    assert entry.node_type == "SamplePerson"
    assert "merge:placeholder" in entry.notes


def test_apply_merge_events_rescued_grants_derived_anchors_to_synthetic_parent():
    """A 'rescued' event lets a synthetic parent inherit its child's observed anchors as 'derived'."""
    from docling_graph.core.provenance import (
        NodeProvenance,
        ProvenanceLedger,
        SourceAnchor,
        identity_key,
    )

    # "studies[]" needs a real NodeSpec so _key_for can compute the child's key;
    # "employees[]" has no id_fields, so its key falls back to the "#bucket" form.
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[]),
            NodeSpec(
                path="employees[]",
                node_type="Bucket",
                id_fields=[],
                parent_path="",
                field_name="employees",
                is_list=True,
            ),
            NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="employees[]",
                field_name="studies",
                is_list=True,
            ),
        ]
    )
    spec_by_path = {s.path: s for s in catalog.nodes}
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
    )

    child_key = identity_key("studies[]", {"study_id": "S1"}, ["study_id"])
    assert child_key is not None
    child_entry = NodeProvenance(
        identity_key=child_key,
        catalog_path="studies[]",
        ids={"study_id": "S1"},
        anchors=[SourceAnchor(chunk_id=4, kind="observed")],
    )
    parent_key = "employees[]#bucket"
    parent_entry = NodeProvenance(
        identity_key=parent_key,
        catalog_path="employees[]",
        ids={},
        synthetic=True,
        anchors=[],
    )
    ledger = ProvenanceLedger(
        node_level=True, chunks={}, nodes={parent_key: parent_entry, child_key: child_entry}
    )
    events = [
        {
            "event": "rescued",
            "how": "bucket",
            "path": "studies[]",
            "ids": {"study_id": "S1"},
            "parent_path": "employees[]",
            "parent_ids": {},
        }
    ]
    orch._apply_merge_events(ledger, events, spec_by_path)
    parent_after = ledger.nodes[parent_key]
    assert len(parent_after.anchors) == 1
    assert parent_after.anchors[0].chunk_id == 4
    assert parent_after.anchors[0].kind == "derived"


def test_rekey_ledger_to_filled_skips_paths_without_id_fields_or_spec():
    """Paths with no spec or no declared id_fields are left as-is (nothing to re-key)."""
    from docling_graph.core.provenance import ProvenanceLedger
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    ledger = ProvenanceLedger(node_level=True, chunks={}, nodes={})
    path_descriptors = {"unknown_path[]": [_desc("unknown_path[]", {}, None)]}
    path_filled: dict[str, list[dict[str, Any]]] = {"unknown_path[]": [{}]}
    # Should not raise despite the unknown path having no matching spec.
    orch._rekey_ledger_to_filled(ledger, path_descriptors, path_filled, spec_by_path)
    assert ledger.nodes == {}


def test_rekey_ledger_to_filled_merges_into_existing_target_entry():
    """When two skeleton ids collapse onto the same final id, their anchors/lineage merge."""
    from docling_graph.core.provenance import (
        NodeProvenance,
        ProvenanceLedger,
        SourceAnchor,
        identity_key,
    )
    from tests.fixtures.sample_templates.test_template import SampleCompany

    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec_by_path = {s.path: s for s in orch._catalog.nodes}
    id_fields = spec_by_path["employees[]"].id_fields
    skel_ids = {"email": "skeleton-guess@x.com"}
    final_ids = {"email": "final@x.com"}
    skel_key = identity_key("employees[]", skel_ids, id_fields)
    final_key = identity_key("employees[]", final_ids, id_fields)
    assert skel_key is not None and final_key is not None and skel_key != final_key
    skel_entry = NodeProvenance(
        identity_key=skel_key,
        catalog_path="employees[]",
        ids=skel_ids,
        anchors=[SourceAnchor(chunk_id=0, kind="observed")],
        merged_from=["some#other"],
    )
    target_entry = NodeProvenance(
        identity_key=final_key,
        catalog_path="employees[]",
        ids=final_ids,
        anchors=[SourceAnchor(chunk_id=1, kind="observed")],
    )
    ledger = ProvenanceLedger(
        node_level=True, chunks={}, nodes={skel_key: skel_entry, final_key: target_entry}
    )
    path_descriptors = {"employees[]": [_desc("employees[]", skel_ids, None)]}
    path_filled = {"employees[]": [dict(final_ids)]}
    orch._rekey_ledger_to_filled(ledger, path_descriptors, path_filled, spec_by_path)
    # The skeleton-keyed entry is gone; its anchor/lineage merged into the target.
    assert skel_key not in ledger.nodes
    target = ledger.nodes[final_key]
    chunk_ids = {a.chunk_id for a in target.anchors}
    assert chunk_ids == {0, 1}
    assert "some#other" in target.merged_from


def test_run_one_fill_batch_accepts_bare_list_output():
    """A fill LLM response that is a bare list (not wrapped in {'items': ...}) is accepted."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"email": "a@x.com", "first_name": "Alice", "last_name": "A"}]

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec = next(s for s in orch._catalog.nodes if s.path == "employees[]")
    _path, _bi, sanitized = orch._run_one_fill_batch(
        path="employees[]",
        spec=spec,
        batch_descriptors=[{"ids": {"email": "a@x.com"}}],
        batch_index=0,
        sub_schema=('{"type": "object", "properties": {"email": {"type": "string"}}, "$defs": {}}'),
        fill_markdown="doc",
        context="t",
    )
    assert sanitized[0]["email"] == "a@x.com"


def test_run_one_fill_batch_truncates_extra_items_without_matching_descriptor():
    """Extra fill items beyond the requested instance count are discarded."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(*, prompt: Any, schema_json: str, context: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "items": [
                {"email": "a@x.com", "first_name": "Alice", "last_name": "A"},
                {"email": "b@x.com", "first_name": "Bob", "last_name": "B"},
            ]
        }

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec = next(s for s in orch._catalog.nodes if s.path == "employees[]")
    _path, _bi, sanitized = orch._run_one_fill_batch(
        path="employees[]",
        spec=spec,
        batch_descriptors=[{"ids": {"email": "a@x.com"}}],  # only one requested
        batch_index=0,
        sub_schema=('{"type": "object", "properties": {"email": {"type": "string"}}, "$defs": {}}'),
        fill_markdown="doc",
        context="t",
    )
    assert len(sanitized) == 1
    assert sanitized[0]["email"] == "a@x.com"


def test_run_one_fill_batch_bumps_truncation_counter():
    """A fill response flagged as truncated increments the shared truncation counter."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if _diagnostics_out is not None:
            _diagnostics_out["truncated"] = True
        return {"items": [{"email": "a@x.com"}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(),
    )
    spec = next(s for s in orch._catalog.nodes if s.path == "employees[]")
    orch._run_one_fill_batch(
        path="employees[]",
        spec=spec,
        batch_descriptors=[{"ids": {"email": "a@x.com"}}],
        batch_index=0,
        sub_schema=('{"type": "object", "properties": {"email": {"type": "string"}}, "$defs": {}}'),
        fill_markdown="doc",
        context="t",
    )
    assert orch._counters.get("truncation_count") == 1


def test_run_skeleton_phase_parallel_records_empty_on_batch_exception():
    """A batch whose worker raises is recorded as an empty result, not crashing the run."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(*, prompt: Any, schema_json: str, context: str, **kwargs: Any) -> dict[str, Any]:
        if "_dense_skeleton_1" in context:
            raise RuntimeError("boom")
        return {"nodes": [{"path": "", "ids": {"company_name": "Acme"}, "parent": None}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(parallel_workers=2),
    )
    catalog = orch._catalog
    spec_by_path = {s.path: s for s in catalog.nodes}
    batches = [
        [(0, "chunk a", 10)],
        [(1, "chunk b", 10)],
    ]
    results = orch._run_skeleton_phase(
        batches=batches,
        workers=2,
        catalog_block="",
        allowed_paths=set(catalog.paths()),
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
    )
    assert len(results) == 2
    # One batch failed outright (exception path) -> empty list, not a crash.
    assert any(r == [] for r in results)


def test_run_emits_parallel_workers_warning_with_single_batch(caplog):
    """Configuring parallel_workers > 1 with only one skeleton batch logs a warning."""
    import logging

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
        if "dense_fill" in context:
            return {
                "items": [
                    {
                        "invoice_number": "INV-1",
                        "date": "d",
                        "total_amount": 1.0,
                        "vendor_name": "v",
                        "items": [],
                    }
                ]
            }
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        # Large skeleton_batch_tokens -> a short doc always fits one batch.
        config=DenseOrchestratorConfig(parallel_workers=3, skeleton_batch_tokens=4096),
    )
    with caplog.at_level(logging.WARNING):
        root = orch.run(
            chunks=["short doc"],
            chunk_metadata=[{"token_count": 5}],
            full_markdown="short doc",
            context="t",
        )
    assert root is not None
    assert any("parallel workers will not be used" in r.message for r in caplog.records)


def test_run_writes_debug_artifacts_when_debug_dir_set(tmp_path):
    """A configured debug_dir causes the skeleton graph, merge stats, and run stats to be written."""

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
        if "dense_fill" in context:
            return {
                "items": [
                    {
                        "invoice_number": "INV-1",
                        "date": "d",
                        "total_amount": 1.0,
                        "vendor_name": "v",
                        "items": [],
                    }
                ]
            }
        return None

    debug_dir = str(tmp_path / "debug")
    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
        debug_dir=debug_dir,
    )
    root = orch.run(
        chunks=["doc text"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc text",
        context="t",
    )
    assert root is not None
    assert os.path.exists(os.path.join(debug_dir, "dense_skeleton_graph.json"))
    assert os.path.exists(os.path.join(debug_dir, "dense_merge_stats.json"))
    assert os.path.exists(os.path.join(debug_dir, "dense_run_stats.json"))
    assert os.path.exists(os.path.join(debug_dir, "dense_provenance.json"))


def test_run_quality_gate_failure_writes_partial_ledger_and_traces(tmp_path):
    """When the skeleton has no root, a partial ledger is written under debug_dir and on_trace fires."""
    traced: list[dict[str, Any]] = []

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            # No root node at all -> quality gate must fail with "empty_skeleton".
            return {"nodes": []}
        return None

    debug_dir = str(tmp_path / "debug_partial")
    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
        debug_dir=debug_dir,
        on_trace=traced.append,
    )
    root = orch.run(
        chunks=["doc text"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc text",
        context="t",
    )
    assert root is None
    assert orch.last_run_stats.get("quality_gate_failure") == "empty_skeleton"
    assert any(t.get("phase1_quality") is False for t in traced)


def test_run_skips_fill_job_when_spec_missing_for_path():
    """A skeleton path absent from spec_by_path (should not normally happen) is skipped in fill planning."""

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None}]}
        if "dense_fill" in context:
            return {
                "items": [
                    {
                        "invoice_number": "INV-1",
                        "date": "d",
                        "total_amount": 1.0,
                        "vendor_name": "v",
                        "items": [],
                    }
                ]
            }
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
    )
    # Running normally still exercises the "if not spec: continue" guard safely
    # since fill_paths is always drawn from the catalog; this test instead
    # verifies the run completes normally, documenting the invariant that
    # every skeleton path in bottom_up_path_order has a spec.
    root = orch.run(
        chunks=["doc text"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc text",
        context="t",
    )
    assert root is not None


def test_run_parallel_fill_records_exception_without_crashing():
    """A fill job that raises in parallel mode is logged and skipped, not fatal to the run."""
    from tests.fixtures.sample_templates.test_template import SampleCompany

    def mock_llm(
        *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        if "dense_skeleton" in context:
            return {
                "nodes": [
                    {"i": 1, "path": "", "ids": {"company_name": "Acme"}},
                    {"i": 2, "path": "employees[]", "ids": {"email": "a@x.com"}, "p": 1},
                    {"i": 3, "path": "employees[]", "ids": {"email": "b@x.com"}, "p": 1},
                ]
            }
        if "dense_reconcile" in context:
            return {"merges": []}
        if "dense_fill_employees[]_1" in context:
            raise RuntimeError("fill boom")
        if "dense_fill" in context:
            return {"items": [{"email": "a@x.com", "first_name": "Alice", "last_name": "A"}]}
        return None

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=SampleCompany,
        # fill_nodes_cap=1 forces two separate fill batches for the two employees,
        # so the two run in parallel and one of them can be made to raise.
        config=DenseOrchestratorConfig(parallel_workers=2, fill_nodes_cap=1),
    )
    root = orch.run(
        chunks=["doc"],
        chunk_metadata=[{"token_count": 5}],
        full_markdown="doc",
        context="test",
    )
    assert root is not None


def test_write_debug_applies_attempt_suffix(tmp_path):
    """A debug_suffix namespaces artifacts so retries never overwrite them."""
    orch = DenseOrchestrator(
        llm_call_fn=lambda **_kwargs: None,
        template=SampleInvoice,
        config=DenseOrchestratorConfig(),
        debug_dir=str(tmp_path / "debug_out"),
        debug_suffix="_attempt2",
    )
    orch._write_debug("dense_skeleton_graph.json", {"nodes": []})
    assert (tmp_path / "debug_out" / "dense_skeleton_graph_attempt2.json").exists()
    assert not (tmp_path / "debug_out" / "dense_skeleton_graph.json").exists()


# --- P1: stable negative reference handles for already-extracted entities ---


def test_normalize_skeleton_batch_resolves_negative_known_handles():
    """A negative p handle resolves against the advertised already-found map."""
    allowed = {"", "studies[]", "studies[].experiments[]"}
    known = {
        -1: {"path": "studies[]", "ids": {"study_id": "S1"}},
        -2: {"path": "", "ids": {"title": "Doc"}},
    }
    nodes = [
        _skeleton({"i": 1, "path": "studies[].experiments[]", "ids": {"exp_id": "E9"}, "p": -1}),
        _skeleton({"i": 2, "path": "studies[]", "ids": {"study_id": "S2"}, "p": -2}),
    ]
    stats: dict[str, int] = {}
    out = normalize_skeleton_batch(nodes, allowed, known_handles=known, stats_out=stats)
    assert out[0]["parent"] == {"path": "studies[]", "ids": {"study_id": "S1"}}
    assert out[1]["parent"] == {"path": "", "ids": {"title": "Doc"}}
    assert stats["parents_from_already_found"] == 2


def test_normalize_skeleton_batch_local_handles_win_over_known_handles():
    """A positive handle present in the response resolves locally, never via known map."""
    allowed = {"", "studies[]"}
    known = {-1: {"path": "", "ids": {"title": "Doc"}}}
    nodes = [
        _skeleton({"i": 1, "path": "", "ids": {"title": "Local Root"}}),
        _skeleton({"i": 2, "path": "studies[]", "ids": {"study_id": "S1"}, "p": 1}),
    ]
    stats: dict[str, int] = {}
    out = normalize_skeleton_batch(nodes, allowed, known_handles=known, stats_out=stats)
    assert out[1]["parent"] == {"path": "", "ids": {"title": "Local Root"}}
    assert stats.get("parents_from_already_found", 0) == 0


def test_normalize_skeleton_batch_unknown_negative_handle_yields_no_parent():
    """A negative handle absent from the known map falls back to parent=None."""
    allowed = {"", "studies[]"}
    nodes = [_skeleton({"i": 1, "path": "studies[]", "ids": {"study_id": "S1"}, "p": -7})]
    out = normalize_skeleton_batch(nodes, allowed, known_handles={-1: {"path": "", "ids": {}}})
    assert out[0]["parent"] is None


def test_sequential_skeleton_phase_advertises_and_resolves_negative_handles():
    """Batch N+1 sees already-found entities as negative handles and can parent onto them."""
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    prompts_seen: list[str] = []

    def mock_llm(
        *,
        prompt: Any,
        schema_json: str,
        context: str,
        response_top_level: str = "object",
        response_schema_name: str = "extraction",
        _diagnostics_out: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        prompts_seen.append(prompt["user"])
        if len(prompts_seen) == 1:
            return {
                "nodes": [
                    {"i": 1, "path": "", "ids": {"company_name": "Acme"}},
                    {
                        "i": 2,
                        "path": "employees[]",
                        "ids": {"email": "alice@x.com"},
                        "p": 1,
                    },
                ]
            }
        # Second batch: a NEW employee referencing the already-extracted root
        # via its negative handle instead of re-emitting it. The window is
        # ordered most-recent-first: -1 = alice, -2 = root.
        return {"nodes": [{"i": 1, "path": "employees[]", "ids": {"email": "bob@x.com"}, "p": -2}]}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    results = orch._run_skeleton_phase(
        batches=[[(0, "Alice works at Acme", 40)], [(1, "Bob also works here", 40)]],
        workers=1,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
    )
    assert "ALREADY EXTRACTED" not in prompts_seen[0]
    assert '"i": -1' in prompts_seen[1] and '"i": -2' in prompts_seen[1]
    assert '"alice@x.com"' in prompts_seen[1]
    bob = results[1][0]
    assert bob["ids"] == {"email": "bob@x.com"}
    assert bob["parent"] == {"path": "", "ids": {"company_name": "Acme"}}
    assert orch._counters.get("parents_from_already_found") == 1


# --- P5: class-name-echo guard in strip_mislabeled_root_ids ---


def test_strip_mislabeled_root_ids_clears_class_name_echo():
    """A root id that merely echoes the template class name is schema echo, not data."""
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        strip_mislabeled_root_ids,
    )

    nodes = [
        {"path": "", "ids": {"reference_document": "AssuranceMRH", "version": "2.1"}},
        {"path": "garanties[]", "ids": {"nom": "AssuranceMRH"}},
    ]
    out = strip_mislabeled_root_ids(nodes, template_class_name="AssuranceMRH")
    assert "reference_document" not in out[0]["ids"]
    assert out[0]["ids"]["version"] == "2.1"
    # Only the root is touched; a child legitimately named like the class survives.
    assert out[1]["ids"]["nom"] == "AssuranceMRH"


def test_strip_mislabeled_root_ids_keeps_real_reference_with_class_guard():
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        strip_mislabeled_root_ids,
    )

    nodes = [{"path": "", "ids": {"reference_document": "HABITATION_07-25"}}]
    out = strip_mislabeled_root_ids(nodes, template_class_name="AssuranceMRH")
    assert out[0]["ids"]["reference_document"] == "HABITATION_07-25"


# --- Plan F: Phase 1 coverage second pass over zero-yield chunks ---


def _coverage_orchestrator(mock_llm: Any) -> tuple[DenseOrchestrator, set, dict]:
    template_cls, allowed, spec_by_path = _company_skeleton_args()
    orch = DenseOrchestrator(
        llm_call_fn=mock_llm, template=template_cls, config=DenseOrchestratorConfig()
    )
    return orch, allowed, spec_by_path


def test_coverage_pass_reexamines_zero_yield_chunks():
    """Chunks with no skeleton yield get one focused retry with reference handles."""
    coverage_prompts: list[str] = []

    def mock_llm(*, prompt: Any, context: str, _diagnostics_out: Any = None, **kwargs: Any) -> dict:
        if "_coverage" in context:
            coverage_prompts.append(prompt["user"])
            return {
                "nodes": [{"i": 1, "path": "employees[]", "ids": {"email": "bob@x.com"}, "p": -2}]
            }
        return {"nodes": []}

    orch, allowed, spec_by_path = _coverage_orchestrator(mock_llm)
    batches = [[(0, "covered chunk", 50), (1, "missed chunk", 50)]]
    merged_skeleton = [
        {"path": "", "ids": {"company_name": "Acme"}, "parent": None, "_source_chunk_ids": [0]},
        {
            "path": "employees[]",
            "ids": {"email": "alice@x.com"},
            "parent": {"path": "", "ids": {"company_name": "Acme"}},
            "_source_chunk_ids": [0],
        },
    ]
    cov_batches, cov_results = orch._run_coverage_pass(
        batches=batches,
        merged_skeleton=merged_skeleton,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
    )
    # Only the uncovered chunk 1 is re-examined, with the retry note + handles.
    assert len(cov_batches) == 1
    assert [cid for cid, _, _ in cov_batches[0]] == [1]
    assert "no entities on the first pass" in coverage_prompts[0]
    assert "ALREADY EXTRACTED" in coverage_prompts[0]
    # The recovered node resolved its parent via the negative handle (-2 = root:
    # -1 is the most recent entity, alice).
    bob = cov_results[0][0]
    assert bob["parent"] == {"path": "", "ids": {"company_name": "Acme"}}
    assert orch._counters.get("coverage_pass_recovered") == 1


def test_coverage_pass_skips_small_uncovered_share():
    """Zero-yield chunks below the token-share threshold are treated as boilerplate."""
    calls: list[str] = []

    def mock_llm(*, prompt: Any, context: str, _diagnostics_out: Any = None, **kwargs: Any) -> dict:
        calls.append(context)
        return {"nodes": []}

    orch, allowed, spec_by_path = _coverage_orchestrator(mock_llm)
    # Uncovered chunk holds 5/1005 tokens (<10%).
    batches = [[(0, "big covered chunk", 1000), (1, "tiny footer", 5)]]
    merged_skeleton = [
        {"path": "", "ids": {"company_name": "Acme"}, "parent": None, "_source_chunk_ids": [0]}
    ]
    cov_batches, cov_results = orch._run_coverage_pass(
        batches=batches,
        merged_skeleton=merged_skeleton,
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
    )
    assert cov_batches == [] and cov_results == []
    assert calls == []


def test_coverage_pass_noop_on_empty_skeleton():
    """A failed phase 1 (no skeleton) must not trigger a duplicate full pass."""
    orch, allowed, spec_by_path = _coverage_orchestrator(lambda **_kwargs: {"nodes": []})
    cov_batches, cov_results = orch._run_coverage_pass(
        batches=[[(0, "chunk", 100)]],
        merged_skeleton=[],
        catalog_block="",
        allowed_paths=allowed,
        global_context=None,
        semantic_guide=None,
        schema_json="{}",
        context="t",
        spec_by_path=spec_by_path,
    )
    assert cov_batches == [] and cov_results == []


# --- P2/P3: reconciliation co-occurrence veto + observability ---


def test_reconciliation_vetoes_same_chunk_merge():
    """Instances first emitted from the same chunk are never merged (CONFORT case)."""
    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        apply_skeleton_reconciliation,
    )
    from tests.fixtures.sample_templates.test_template import SampleCompany

    catalog = build_node_catalog(SampleCompany)
    spec_by_path = {s.path: s for s in catalog.nodes}
    # Both offers came from the same page-1 summary-table chunk.
    nodes = [
        {
            "path": "employees[]",
            "ids": {"email": "confort@x.com"},
            "parent": None,
            "_source_chunk_ids": [0],
        },
        {
            "path": "employees[]",
            "ids": {"email": "confort-plus@x.com"},
            "parent": None,
            "_source_chunk_ids": [0, 3],
        },
    ]
    events: list[dict] = []
    kept, merged = apply_skeleton_reconciliation(
        nodes,
        [{"path": "employees[]", "keep": 1, "merge": [0]}],
        spec_by_path,
        events_out=events,
    )
    assert merged == 0
    assert len(kept) == 2
    assert events[0]["action"] == "vetoed_cooccurrence"
    assert events[0]["shared_chunks"] == [0]


def test_reconciliation_merge_from_distinct_chunks_applies_and_logs():
    """A genuine alias (table label chunk vs section title chunk) still merges."""
    from docling_graph.core.extractors.contracts.dense.catalog import build_node_catalog
    from docling_graph.core.extractors.contracts.dense.orchestrator import (
        apply_skeleton_reconciliation,
    )
    from tests.fixtures.sample_templates.test_template import SampleCompany

    catalog = build_node_catalog(SampleCompany)
    spec_by_path = {s.path: s for s in catalog.nodes}
    nodes = [
        {
            "path": "employees[]",
            "ids": {"email": "rc@x.com"},
            "parent": None,
            "_source_chunk_ids": [0],
        },
        {
            "path": "employees[]",
            "ids": {"email": "rc-vie-privee@x.com"},
            "parent": None,
            "_source_chunk_ids": [7],
        },
    ]
    events: list[dict] = []
    kept, merged = apply_skeleton_reconciliation(
        nodes,
        [{"path": "employees[]", "keep": 1, "merge": [0]}],
        spec_by_path,
        events_out=events,
    )
    assert merged == 1
    assert len(kept) == 1
    assert events[0]["action"] == "merged"
    assert events[0]["merge_ids"] == {"email": "rc@x.com"}


def test_reconciliation_writes_debug_artifact(tmp_path):
    """The reconciliation pass dumps proposals, LLM answer, and events to debug."""
    template_cls, _allowed, spec_by_path = _company_skeleton_args()

    def mock_llm(*, prompt: Any, context: str, **kwargs: Any) -> dict:
        if "_dense_reconcile" in context:
            return {"merges": [{"path": "employees[]", "keep": 1, "merge": [0]}]}
        return {"nodes": []}

    orch = DenseOrchestrator(
        llm_call_fn=mock_llm,
        template=template_cls,
        config=DenseOrchestratorConfig(),
        debug_dir=str(tmp_path / "debug"),
    )
    skeleton = [
        {
            "path": "employees[]",
            "ids": {"email": "a@x.com"},
            "parent": None,
            "_source_chunk_ids": [0],
        },
        {
            "path": "employees[]",
            "ids": {"email": "alice-a@x.com"},
            "parent": None,
            "_source_chunk_ids": [4],
        },
    ]
    _, merged = orch._run_skeleton_reconciliation(skeleton, spec_by_path, "t")
    assert merged == 1
    import json as _json

    artifact = _json.loads((tmp_path / "debug" / "dense_reconciliation.json").read_text())
    assert artifact["llm_merges"] == [{"path": "employees[]", "keep": 1, "merge": [0]}]
    assert artifact["events"][0]["action"] == "merged"
