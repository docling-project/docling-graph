"""Tests for canonical identity computation and compact views."""

from typing import Any

from docling_graph.core.provenance import (
    ChunkRecord,
    DocumentOrigin,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
    canonical_id_text,
    compact_view,
    identity_key,
    identity_pairs,
    merge_compact_views,
)


class TestIdentityPairs:
    def test_prefers_declared_id_fields(self):
        ids = {"name": "LFP Slurry", "batch_id": "B-1"}
        pairs = identity_pairs(ids, ["name"])
        assert pairs == (("name", "LFP_SLURRY"),)

    def test_falls_back_to_all_ids_when_declared_fields_absent(self):
        ids = {"batch_id": "B-1"}
        pairs = identity_pairs(ids, ["name"])
        assert pairs == (("batch_id", "b1"),)

    def test_order_independence(self):
        a = identity_pairs({"b": "2", "a": "1"}, [])
        b = identity_pairs({"a": "1", "b": "2"}, [])
        assert a == b

    def test_none_values_skipped(self):
        assert identity_pairs({"name": None}, ["name"]) == ()

    def test_casing_and_punctuation_canonicalized(self):
        a = identity_pairs({"run_id": "Run-1"}, ["run_id"])
        b = identity_pairs({"run_id": "run_1"}, ["run_id"])
        assert a == b

    def test_name_field_uses_entity_normalization(self):
        a = identity_pairs({"name": "The John Doe"}, ["name"])
        b = identity_pairs({"name": "john doe"}, ["name"])
        assert a == b


class TestIdentityKey:
    def test_stable_serialization(self):
        key = identity_key("studies[]", {"name": "Study A"}, ["name"])
        assert key == "studies[]|name=STUDY_A"

    def test_unkeyable_when_no_ids(self):
        assert identity_key("studies[]", {}, ["name"]) is None

    def test_unkeyable_when_all_values_canonicalize_empty(self):
        assert identity_key("studies[]", {"name": "The"}, ["name"]) is None

    def test_path_disambiguates(self):
        a = identity_key("a[]", {"name": "X"}, ["name"])
        b = identity_key("b[]", {"name": "X"}, ["name"])
        assert a != b

    def test_matches_skeleton_identity_key_semantics(self):
        """identity_pairs must agree with the orchestrator's dedup key."""
        from docling_graph.core.extractors.contracts.dense.catalog import NodeSpec
        from docling_graph.core.extractors.contracts.dense.orchestrator import (
            _skeleton_identity_key,
        )

        spec = NodeSpec(path="items[]", node_type="Item", id_fields=["name"])
        node = {"path": "items[]", "ids": {"name": "Widget Pro", "sku": "W-1"}}
        tuple_key = _skeleton_identity_key(node, {"items[]": spec})
        assert tuple_key[0] == "items[]"
        assert tuple_key[1] == identity_pairs(node["ids"], spec.id_fields)


class TestCanonicalIdText:
    def test_joins_canonical_values(self):
        assert canonical_id_text({"name": "LFP Slurry", "run_id": "Run-1"}) == "LFP_SLURRY run1"

    def test_empty_for_no_usable_ids(self):
        assert canonical_id_text({"name": None}) == ""


def _ledger_with_entry(**entry_kwargs: Any) -> tuple[ProvenanceLedger, NodeProvenance]:
    entry = NodeProvenance(identity_key="items[]|name=x", catalog_path="items[]", **entry_kwargs)
    ledger = ProvenanceLedger(
        document=DocumentOrigin(document_id="doc1", source="a.pdf"),
        chunks={
            0: ChunkRecord(chunk_id=0, batch_index=0, page_numbers=(1,)),
            1: ChunkRecord(chunk_id=1, batch_index=0, page_numbers=(2, 3)),
        },
        nodes={entry.identity_key: entry},
    )
    return ledger, entry


class TestCompactView:
    def test_observed_only_view_is_approximate(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0), SourceAnchor(chunk_id=1)]
        )
        view = compact_view(entry, ledger)
        assert view["document_id"] == "doc1"
        assert view["chunks"] == [0, 1]
        assert view["pages"] == [1, 2, 3]
        assert view["match"] == "observed"
        assert view["approximate"] is True

    def test_verbatim_leads_with_exact_location_only(self):
        # Verbatim view reports ONLY the exact chunk(s), never the broad
        # observed set, and is not flagged approximate.
        ledger, entry = _ledger_with_entry(
            anchors=[
                SourceAnchor(chunk_id=0),
                SourceAnchor(chunk_id=1, kind="verbatim", span=(5, 12)),
            ]
        )
        view = compact_view(entry, ledger)
        assert view["match"] == "verbatim"
        assert view["chunks"] == [1]
        assert view["pages"] == [2, 3]
        assert "approximate" not in view
        assert "spans" not in view  # spans only with include_spans (detailed)

    def test_verbatim_spans_when_requested(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=1, kind="verbatim", span=(5, 12))]
        )
        view = compact_view(entry, ledger, include_spans=True)
        assert view["spans"] == [{"chunk": 1, "start": 5, "end": 12}]

    def test_caps_chunk_list(self):
        anchors = [SourceAnchor(chunk_id=i) for i in range(12)]
        ledger, entry = _ledger_with_entry(anchors=anchors)
        view = compact_view(entry, ledger, max_anchors=8)
        assert len(view["chunks"]) == 8
        assert view["chunks_omitted"] == 4

    def test_document_scope_note_short_circuits(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0)], notes=["scope:document"]
        )
        view = compact_view(entry, ledger)
        assert view == {"document_id": "doc1", "scope": "document"}

    def test_anchor_dedup(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0), SourceAnchor(chunk_id=0)]
        )
        assert compact_view(entry, ledger)["chunks"] == [0]

    def test_document_scope_note_synthetic_flag(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0)], notes=["scope:document"], synthetic=True
        )
        view = compact_view(entry, ledger)
        assert view == {"document_id": "doc1", "scope": "document", "synthetic": True}

    def test_verbatim_chunks_omitted_when_over_max(self):
        anchors = [SourceAnchor(chunk_id=i, kind="verbatim", span=(0, 1)) for i in range(3)]
        ledger, entry = _ledger_with_entry(anchors=anchors)
        view = compact_view(entry, ledger, max_anchors=2)
        assert view["match"] == "verbatim"
        assert view["chunks"] == [0, 1]
        assert view["chunks_omitted"] == 1

    def test_verbatim_spans_capped_at_max_spans(self):
        anchors = [SourceAnchor(chunk_id=i, kind="verbatim", span=(i, i + 1)) for i in range(6)]
        ledger, entry = _ledger_with_entry(anchors=anchors)
        view = compact_view(entry, ledger, include_spans=True)
        assert len(view["spans"]) == 4
        assert view["spans"][0] == {"chunk": 0, "start": 0, "end": 1}

    def test_verbatim_synthetic_flag(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0, kind="verbatim", span=(0, 1))],
            synthetic=True,
        )
        view = compact_view(entry, ledger)
        assert view["match"] == "verbatim"
        assert view["synthetic"] is True

    def test_approximate_chunks_omitted_when_over_max(self):
        anchors = [SourceAnchor(chunk_id=i) for i in range(10)]
        ledger, entry = _ledger_with_entry(anchors=anchors)
        view = compact_view(entry, ledger, max_anchors=8)
        assert view["approximate"] is True
        assert view["chunks_omitted"] == 2

    def test_approximate_synthetic_flag(self):
        ledger, entry = _ledger_with_entry(anchors=[SourceAnchor(chunk_id=0)], synthetic=True)
        view = compact_view(entry, ledger)
        assert view["approximate"] is True
        assert view["synthetic"] is True


class TestMergeCompactViews:
    def test_none_and_unresolved_yield(self):
        resolved = {"document_id": "d", "match": "observed", "chunks": [1], "pages": []}
        assert merge_compact_views(None, resolved) == resolved
        assert merge_compact_views({"status": "unresolved"}, resolved) == resolved
        assert merge_compact_views(resolved, None) == resolved

    def test_a_unresolved_yields_to_b(self):
        resolved = {"document_id": "d", "match": "observed", "chunks": [1], "pages": []}
        unresolved = {"status": "unresolved"}
        assert merge_compact_views(unresolved, resolved) == resolved

    def test_b_unresolved_yields_to_a(self):
        resolved = {"document_id": "d", "match": "observed", "chunks": [1], "pages": []}
        unresolved = {"status": "unresolved"}
        assert merge_compact_views(resolved, unresolved) == resolved

    def test_union_chunks_pages_strongest_match(self):
        a = {"document_id": "d", "match": "observed", "chunks": [1], "pages": [1]}
        b = {"document_id": "d", "match": "verbatim", "chunks": [2], "pages": [2]}
        merged = merge_compact_views(a, b)
        assert merged["chunks"] == [1, 2]
        assert merged["pages"] == [1, 2]
        assert merged["match"] == "verbatim"

    def test_document_scope_absorbs(self):
        a = {"document_id": "d", "scope": "document"}
        b = {"document_id": "d", "match": "observed", "chunks": [2], "pages": []}
        assert merge_compact_views(a, b) == {"document_id": "d", "scope": "document"}

    def test_document_scope_absorbs_synthetic_when_both_synthetic(self):
        a = {"document_id": "d", "scope": "document", "synthetic": True}
        b = {"document_id": "d", "scope": "document", "synthetic": True}
        merged = merge_compact_views(a, b)
        assert merged == {"document_id": "d", "scope": "document", "synthetic": True}

    def test_document_scope_absorbs_without_synthetic_when_only_one_synthetic(self):
        a = {"document_id": "d", "scope": "document", "synthetic": True}
        b = {"document_id": "d", "match": "observed", "chunks": [2], "pages": []}
        merged = merge_compact_views(a, b)
        assert merged == {"document_id": "d", "scope": "document"}

    def test_approximate_only_when_both_sides_approximate(self):
        a = {
            "document_id": "d",
            "match": "observed",
            "chunks": [1],
            "pages": [],
            "approximate": True,
        }
        b = {"document_id": "d", "match": "verbatim", "chunks": [2], "pages": []}
        merged = merge_compact_views(a, b)
        assert "approximate" not in merged

    def test_approximate_set_when_both_sides_approximate(self):
        a = {
            "document_id": "d",
            "match": "observed",
            "chunks": [1],
            "pages": [],
            "approximate": True,
        }
        b = {
            "document_id": "d",
            "match": "observed",
            "chunks": [2],
            "pages": [],
            "approximate": True,
        }
        merged = merge_compact_views(a, b)
        assert merged["approximate"] is True

    def test_spans_deduped_and_capped(self):
        a = {
            "document_id": "d",
            "match": "verbatim",
            "chunks": [1],
            "pages": [],
            "spans": [
                {"chunk": 1, "start": 0, "end": 1},
                {"chunk": 2, "start": 0, "end": 1},
            ],
        }
        b = {
            "document_id": "d",
            "match": "verbatim",
            "chunks": [2],
            "pages": [],
            "spans": [
                {"chunk": 1, "start": 0, "end": 1},  # duplicate of a's first span
                {"chunk": 3, "start": 0, "end": 1},
                {"chunk": 4, "start": 0, "end": 1},
                {"chunk": 5, "start": 0, "end": 1},
            ],
        }
        merged = merge_compact_views(a, b)
        assert len(merged["spans"]) == 4
        assert merged["spans"][0] == {"chunk": 1, "start": 0, "end": 1}
        assert {"chunk": 1, "start": 0, "end": 1} in merged["spans"]
        # Duplicate collapsed, so 5 unique input spans still cap at 4.
        assert merged["spans"].count({"chunk": 1, "start": 0, "end": 1}) == 1

    def test_chunks_omitted_summed(self):
        a = {
            "document_id": "d",
            "match": "observed",
            "chunks": [1],
            "pages": [],
            "chunks_omitted": 2,
        }
        b = {
            "document_id": "d",
            "match": "observed",
            "chunks": [2],
            "pages": [],
            "chunks_omitted": 3,
        }
        merged = merge_compact_views(a, b)
        assert merged["chunks_omitted"] == 5

    def test_synthetic_combination_in_general_merge(self):
        a = {"document_id": "d", "match": "observed", "chunks": [1], "pages": [], "synthetic": True}
        b = {"document_id": "d", "match": "observed", "chunks": [2], "pages": [], "synthetic": True}
        merged = merge_compact_views(a, b)
        assert merged["synthetic"] is True

    def test_synthetic_not_combined_when_only_one_side(self):
        a = {"document_id": "d", "match": "observed", "chunks": [1], "pages": [], "synthetic": True}
        b = {"document_id": "d", "match": "observed", "chunks": [2], "pages": []}
        merged = merge_compact_views(a, b)
        assert "synthetic" not in merged


class TestLedgerRoundTrip:
    def test_json_round_trip(self):
        ledger, entry = _ledger_with_entry(
            anchors=[SourceAnchor(chunk_id=0, kind="verbatim", span=(1, 4))],
            merged_from=["items[]|name=y"],
            notes=["identity:unkeyed"],
        )
        restored = ProvenanceLedger.model_validate_json(ledger.model_dump_json())
        assert restored.nodes[entry.identity_key].anchors[0].span == (1, 4)
        assert restored.chunks[1].page_numbers == (2, 3)
        assert restored.document is not None
        assert restored.document.document_id == "doc1"
