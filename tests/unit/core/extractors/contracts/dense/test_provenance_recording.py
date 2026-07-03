"""Tests for dense-contract provenance recording (chunk stamping, ledger freeze,
merge accounting) — spec hooks H2-H7."""

from typing import Any

from docling_graph.core.extractors.contracts.dense.catalog import NodeCatalog, NodeSpec
from docling_graph.core.extractors.contracts.dense.models import DenseSkeletonNode
from docling_graph.core.extractors.contracts.dense.orchestrator import (
    DenseOrchestrator,
    DenseOrchestratorConfig,
    apply_skeleton_reconciliation,
    merge_filled_into_root,
    merge_skeleton_batches,
    normalize_skeleton_batch,
    prune_barren_branches,
)
from docling_graph.core.provenance import (
    ChunkRecord,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
    compact_view,
)
from tests.fixtures.sample_templates.test_template import SampleCompany


def _company_catalog_specs() -> dict[str, NodeSpec]:
    root = NodeSpec(path="", node_type="SampleCompany", id_fields=["company_name"])
    emp = NodeSpec(
        path="employees[]",
        node_type="SamplePerson",
        id_fields=["email"],
        parent_path="",
        field_name="employees",
        is_list=True,
    )
    return {"": root, "employees[]": emp}


class TestChunkIdStamping:
    def test_normalize_skeleton_batch_stamps_source_chunk_ids(self):
        nodes = [DenseSkeletonNode(i=0, path="", ids={"invoice_number": "INV-1"})]
        out = normalize_skeleton_batch(nodes, {""}, source_batch_index=2, source_chunk_ids=[4, 5])
        assert out[0]["_source_batch_index"] == 2
        assert out[0]["_source_chunk_ids"] == [4, 5]

    def test_merge_keeps_first_emission_chunks_but_unions_batches(self):
        # Re-emissions (already_found echo) must NOT accumulate chunks — only
        # the first batch's chunks are the genuine reading — but fill-context
        # batch indexes still union.
        catalog = NodeCatalog(
            nodes=[NodeSpec(path="items[]", node_type="Item", id_fields=["name"])]
        )
        node_a = {
            "path": "items[]",
            "ids": {"name": "X"},
            "parent": None,
            "_source_batch_index": 0,
            "_source_chunk_ids": [0, 1],
        }
        node_b = {
            "path": "items[]",
            "ids": {"name": "X"},
            "parent": None,
            "_source_batch_index": 2,
            "_source_chunk_ids": [1, 5],
        }
        merged = merge_skeleton_batches([[node_a], [node_b]], catalog)
        assert len(merged) == 1
        assert merged[0]["_source_chunk_ids"] == [0, 1]  # first emission only
        assert merged[0]["_source_batch_indexes"] == [0, 2]  # batches still union

    def test_root_collapse_keeps_primary_first_emission(self):
        catalog = NodeCatalog(nodes=[NodeSpec(path="", node_type="Root", id_fields=["name"])])
        root_a = {
            "path": "",
            "ids": {"name": "Title A"},
            "parent": None,
            "_source_batch_index": 0,
            "_source_chunk_ids": [0],
        }
        root_b = {
            "path": "",
            "ids": {"name": "Title A (variant)"},
            "parent": None,
            "_source_batch_index": 1,
            "_source_chunk_ids": [3],
        }
        merged = merge_skeleton_batches([[root_a], [root_b]], catalog)
        assert len(merged) == 1
        assert merged[0]["_source_chunk_ids"] == [0]
        assert merged[0]["_source_batch_indexes"] == [0, 1]


class TestReconciliationLineage:
    def test_reconciliation_absorbs_chunks_and_records_merged_from(self):
        spec_by_path = _company_catalog_specs()
        keeper = {
            "path": "employees[]",
            "ids": {"email": "jane@acme.com"},
            "parent": None,
            "_source_batch_indexes": [0],
            "_source_chunk_ids": [0],
        }
        alias = {
            "path": "employees[]",
            "ids": {"email": "j.doe@acme.com"},
            "parent": None,
            "_source_batch_indexes": [1],
            "_source_chunk_ids": [2],
        }
        kept, merged_count = apply_skeleton_reconciliation(
            [keeper, alias],
            [{"path": "employees[]", "keep": 0, "merge": [1]}],
            spec_by_path,
        )
        assert merged_count == 1
        assert len(kept) == 1
        assert kept[0]["_reconciled_chunk_ids"] == [2]
        assert kept[0]["_source_chunk_ids"] == [0]  # own observations untouched
        assert kept[0]["_merged_from"] == ["employees[]|email=jdoeacmecom"]


class TestMergeEvents:
    def _catalog(self) -> NodeCatalog:
        return NodeCatalog(
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

    def test_placeholder_rescue_emits_synthetic_and_rescued_events(self):
        catalog = self._catalog()
        path_filled = {
            "": [{}],
            "studies[].experiments[]": [{"exp_id": "E1", "note": "data"}],
        }
        path_descriptors = {
            "": [{"path": "", "ids": {}, "parent": None}],
            "studies[].experiments[]": [
                {
                    "path": "studies[].experiments[]",
                    "ids": {"exp_id": "E1"},
                    "parent": {"path": "studies[]", "ids": {"study_id": "S9"}},
                }
            ],
        }
        events: list[dict[str, Any]] = []
        root = merge_filled_into_root(path_filled, path_descriptors, catalog, events_out=events)
        assert root["studies"][0]["study_id"] == "S9"
        synthetic = [e for e in events if e["event"] == "synthetic"]
        rescued = [e for e in events if e["event"] == "rescued"]
        assert synthetic == [{"event": "synthetic", "path": "studies[]", "ids": {"study_id": "S9"}}]
        assert len(rescued) == 1
        assert rescued[0]["how"] == "placeholder"
        assert rescued[0]["path"] == "studies[].experiments[]"
        assert rescued[0]["parent_ids"] == {"study_id": "S9"}

    def test_prune_emits_pruned_events(self):
        catalog = self._catalog()
        root = {
            "studies": [
                {"study_id": "S1", "objective": None, "experiments": []},
                {"study_id": "S2", "objective": "Real", "experiments": []},
            ]
        }
        events: list[dict[str, Any]] = []
        out = prune_barren_branches(root, catalog, events_out=events)
        assert len(out["studies"]) == 1
        assert events == [{"event": "pruned", "path": "studies[]", "ids": {"study_id": "S1"}}]


class TestApplyMergeEvents:
    def test_synthetic_parent_inherits_derived_anchors_and_drop_marks_entry(self):
        spec_by_path = {
            "studies[]": NodeSpec(
                path="studies[]",
                node_type="Study",
                id_fields=["study_id"],
                parent_path="",
                field_name="studies",
                is_list=True,
            ),
            "studies[].experiments[]": NodeSpec(
                path="studies[].experiments[]",
                node_type="Experiment",
                id_fields=["exp_id"],
                parent_path="studies[]",
                field_name="experiments",
                is_list=True,
            ),
        }
        child = NodeProvenance(
            identity_key="studies[].experiments[]|exp_id=e1",
            catalog_path="studies[].experiments[]",
            ids={"exp_id": "E1"},
            anchors=[SourceAnchor(chunk_id=3, kind="observed")],
        )
        dropped = NodeProvenance(
            identity_key="studies[].experiments[]|exp_id=e2",
            catalog_path="studies[].experiments[]",
            ids={"exp_id": "E2"},
        )
        ledger = ProvenanceLedger(
            chunks={3: ChunkRecord(chunk_id=3, batch_index=0)},
            nodes={child.identity_key: child, dropped.identity_key: dropped},
        )
        orch = DenseOrchestrator(
            llm_call_fn=lambda **kwargs: None,
            template=SampleCompany,
            config=DenseOrchestratorConfig(),
        )
        events = [
            {"event": "synthetic", "path": "studies[]", "ids": {"study_id": "S9"}},
            {
                "event": "rescued",
                "how": "placeholder",
                "path": "studies[].experiments[]",
                "ids": {"exp_id": "E1"},
                "parent_path": "studies[]",
                "parent_ids": {"study_id": "S9"},
            },
            {
                "event": "dropped",
                "path": "studies[].experiments[]",
                "ids": {"exp_id": "E2"},
            },
        ]
        orch._apply_merge_events(ledger, events, spec_by_path)

        parent = ledger.nodes["studies[]|study_id=s9"]
        assert parent.synthetic is True
        assert [(a.chunk_id, a.kind) for a in parent.anchors] == [(3, "derived")]
        assert ledger.nodes["studies[].experiments[]|exp_id=e2"].dropped is True


def _company_mock_llm(
    *, prompt: Any, schema_json: str, context: str, **kwargs: Any
) -> dict[str, Any] | None:
    if "dense_skeleton_0" in context:
        return {"nodes": [{"i": 0, "path": "", "ids": {"company_name": "Acme"}, "p": None}]}
    if "dense_skeleton_1" in context:
        return {
            "nodes": [
                {"i": 0, "path": "", "ids": {"company_name": "Acme"}, "p": None},
                {"i": 1, "path": "employees[]", "ids": {"email": "jane@acme.com"}, "p": 0},
            ]
        }
    if "dense_fill_employees[]" in context:
        return {"items": [{"first_name": "Jane", "last_name": "Doe", "email": "jane@acme.com"}]}
    if "dense_fill_" in context:
        return {
            "items": [
                {
                    "company_name": "Acme",
                    "industry": "Robotics",
                    "founded_year": 1999,
                    "employees": [],
                }
            ]
        }
    if "dense_reconcile" in context:
        return {"merges": []}
    return None


def _run_company_orchestrator(**config_kwargs: Any) -> DenseOrchestrator:
    chunks = ["Acme overview text. " * 10, "Jane Doe jane@acme.com works at Acme. " * 5]
    orch = DenseOrchestrator(
        llm_call_fn=_company_mock_llm,
        template=SampleCompany,
        config=DenseOrchestratorConfig(skeleton_batch_tokens=50, **config_kwargs),
    )
    root = orch.run(
        chunks=chunks,
        chunk_metadata=[
            {"token_count": 40, "page_numbers": [1], "doc_item_refs": ["#/texts/0"]},
            {"token_count": 40, "page_numbers": [2], "doc_item_refs": ["#/texts/9"]},
        ],
        full_markdown="\n\n".join(chunks),
        context="test",
    )
    assert root is not None
    return orch


class TestLedgerEndToEnd:
    def test_ledger_records_chunks_and_node_anchors(self):
        orch = _run_company_orchestrator()
        ledger = orch.last_provenance
        assert ledger is not None
        # Chunk index from batch tuples + metadata join
        assert set(ledger.chunks) == {0, 1}
        assert ledger.chunks[0].page_numbers == (1,)
        assert ledger.chunks[1].doc_item_refs == ("#/texts/9",)
        assert ledger.chunks[0].batch_index == 0
        assert ledger.chunks[1].batch_index == 1
        assert len(ledger.chunks[0].text_hash) == 16
        # Chunk text is stored so the ledger is self-contained (issue #3).
        assert ledger.chunks[0].text and ledger.chunks[1].text
        assert ledger.node_level is True
        # Root is document-scoped (first-emission chunk kept, scope note present)
        root_entry = ledger.nodes["|company_name=acme"]
        assert "scope:document" in root_entry.notes
        # Employee: observed in its first-emission chunk (1). The orchestrator no
        # longer runs the verbatim scan (the binder does, with final model ids),
        # so at this stage the anchor is observed-only.
        emp_entry = ledger.nodes["employees[]|email=janeacmecom"]
        assert [(a.chunk_id, a.kind) for a in emp_entry.anchors] == [(1, "observed")]
        assert emp_entry.node_type == "SamplePerson"
        assert emp_entry.ids == {"email": "jane@acme.com"}
        assert ledger.resolution == "chunk"
        # Every anchor points into the chunk index
        for entry in ledger.nodes.values():
            for anchor in entry.anchors:
                assert anchor.chunk_id in ledger.chunks

    def test_parallel_and_sequential_ledgers_agree(self):
        sequential = _run_company_orchestrator(parallel_workers=1).last_provenance
        parallel = _run_company_orchestrator(parallel_workers=2).last_provenance
        assert sequential is not None and parallel is not None
        assert set(sequential.nodes) == set(parallel.nodes)
        for key, seq_entry in sequential.nodes.items():
            par_entry = parallel.nodes[key]
            assert {(a.chunk_id, a.kind) for a in seq_entry.anchors} == {
                (a.chunk_id, a.kind) for a in par_entry.anchors
            }

    def test_provenance_off_records_nothing(self):
        orch = _run_company_orchestrator(provenance_mode="off")
        assert orch.last_provenance is None

    def test_quality_gate_failure_leaves_no_ledger(self):
        def no_root_llm(
            *, prompt: Any, schema_json: str, context: str, **kwargs: Any
        ) -> dict[str, Any] | None:
            if "dense_skeleton" in context:
                return {"nodes": []}
            return None

        orch = DenseOrchestrator(
            llm_call_fn=no_root_llm,
            template=SampleCompany,
            config=DenseOrchestratorConfig(),
        )
        root = orch.run(
            chunks=["some text"],
            chunk_metadata=[{"token_count": 10}],
            full_markdown="some text",
            context="test",
        )
        assert root is None
        assert orch.last_provenance is None

    def test_fill_batches_recorded(self):
        orch = _run_company_orchestrator()
        ledger = orch.last_provenance
        assert ledger is not None
        assert ledger.nodes["employees[]|email=janeacmecom"].fill_batches == [0]


class TestRekeyToFilledIds:
    """Phase 2 often refines a rough skeleton id to the real one; the ledger
    entry must follow so the graph node (final id) can still bind (issue #2)."""

    def _refining_llm(
        self, *, prompt: Any, schema_json: str, context: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        # Skeleton captures a generic placeholder email; fill returns the real one.
        if "dense_skeleton" in context:
            return {
                "nodes": [
                    {"i": 0, "path": "", "ids": {"company_name": "Acme"}, "p": None},
                    {"i": 1, "path": "employees[]", "ids": {"email": "person@placeholder"}, "p": 0},
                ]
            }
        if "dense_fill_employees[]" in context:
            return {"items": [{"first_name": "Jane", "last_name": "Doe", "email": "jane@acme.com"}]}
        if "dense_fill_" in context:
            return {
                "items": [
                    {
                        "company_name": "Acme",
                        "industry": "Robotics",
                        "founded_year": 1999,
                        "employees": [],
                    }
                ]
            }
        if "dense_reconcile" in context:
            return {"merges": []}
        return None

    def test_ledger_entry_rekeyed_from_skeleton_to_fill_id(self):
        orch = DenseOrchestrator(
            llm_call_fn=self._refining_llm,
            template=SampleCompany,
            config=DenseOrchestratorConfig(skeleton_batch_tokens=50),
        )
        chunks = ["Acme overview.", "Jane Doe jane@acme.com at Acme."]
        root = orch.run(
            chunks=chunks,
            chunk_metadata=[
                {"token_count": 5, "page_numbers": [1]},
                {"token_count": 8, "page_numbers": [2]},
            ],
            full_markdown="\n\n".join(chunks),
            context="test",
        )
        assert root is not None
        ledger = orch.last_provenance
        assert ledger is not None
        # Re-keyed to the FINAL fill id; the skeleton placeholder key is gone.
        assert "employees[]|email=janeacmecom" in ledger.nodes
        assert "employees[]|email=personplaceholder" not in ledger.nodes
        # The observed anchor travelled with the re-key (so the node still grounds).
        entry = ledger.nodes["employees[]|email=janeacmecom"]
        assert entry.ids == {"email": "jane@acme.com"}
        assert any(a.kind == "observed" for a in entry.anchors)


class TestConfigParsing:
    def test_from_dict_parses_provenance_mode(self):
        assert DenseOrchestratorConfig.from_dict({}).provenance_mode == "standard"
        assert (
            DenseOrchestratorConfig.from_dict({"provenance": "detailed"}).provenance_mode
            == "detailed"
        )
        assert DenseOrchestratorConfig.from_dict({"provenance": "off"}).provenance_mode == "off"
        assert (
            DenseOrchestratorConfig.from_dict({"provenance": "bogus"}).provenance_mode == "standard"
        )
