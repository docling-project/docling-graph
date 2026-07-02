"""
Unit tests for dense extraction orchestrator (Phase 1 and Phase 2, including parallel).
"""

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
        quality_min_instances=1,
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


def test_merge_filled_counts_dropped_when_parent_unresolvable():
    """A child with no parent ids and several candidate parents is dropped and counted."""
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
            _desc("studies[].experiments[]", {"exp_id": "E1"}, None),
        ],
    }
    stats: dict[str, int] = {}
    merge_filled_into_root(path_filled, path_descriptors, catalog, stats_out=stats)
    assert stats["dropped"] == 1


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
