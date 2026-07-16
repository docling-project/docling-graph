"""Unit tests for cross-document provenance merging (core.merge.provenance_merge)."""

import json

from docling_graph.core.converters.graph_converter import _provenance_weight
from docling_graph.core.merge.provenance_merge import (
    MAX_VIEW_SOURCES,
    merge_node_views,
    write_ledger_sidecars,
)
from docling_graph.core.provenance.identity import (
    PROVENANCE_NODE_ATTR,
    iter_provenance_views,
    merge_compact_views,
)
from docling_graph.core.provenance.models import DocumentOrigin, ProvenanceLedger


def _view(doc: str, chunks: list[int]) -> dict:
    return {"document_id": doc, "match": "verbatim", "chunks": chunks, "pages": [1]}


def test_same_document_delegates_to_merge_compact_views():
    a, b = _view("doc-a", [1, 3]), _view("doc-a", [2])
    assert merge_node_views(a, b) == merge_compact_views(a, b)
    merged = merge_node_views(a, b)
    assert merged is not None
    assert "multi_document" not in merged
    assert merged["chunks"] == [1, 2, 3]


def test_missing_document_id_never_wraps():
    a = _view("", [1])
    b = _view("doc-b", [2])
    merged = merge_node_views(a, b)
    assert merged is not None
    assert "multi_document" not in merged
    assert merged["document_id"] == "doc-b"


def test_cross_document_wraps_without_blending_chunks():
    merged = merge_node_views(_view("doc-a", [3]), _view("doc-b", [3]))
    assert merged is not None
    assert merged["multi_document"] is True
    by_doc = {s["document_id"]: s for s in merged["sources"]}
    # Chunk 3 stays resolvable against the right ledger on each side.
    assert by_doc["doc-a"]["chunks"] == [3]
    assert by_doc["doc-b"]["chunks"] == [3]


def test_wrap_plus_new_document_appends():
    wrapped = merge_node_views(_view("doc-a", [1]), _view("doc-b", [2]))
    merged = merge_node_views(wrapped, _view("doc-c", [5]))
    assert merged is not None
    assert [s["document_id"] for s in merged["sources"]] == ["doc-a", "doc-b", "doc-c"]


def test_wrap_plus_same_document_merges_into_matching_source():
    wrapped = merge_node_views(_view("doc-a", [1]), _view("doc-b", [2]))
    merged = merge_node_views(wrapped, _view("doc-a", [7]))
    assert merged is not None
    by_doc = {s["document_id"]: s for s in merged["sources"]}
    assert by_doc["doc-a"]["chunks"] == [1, 7]
    assert by_doc["doc-b"]["chunks"] == [2]


def test_wrap_plus_wrap_merges_source_wise():
    left = merge_node_views(_view("doc-a", [1]), _view("doc-b", [2]))
    right = merge_node_views(_view("doc-b", [9]), _view("doc-c", [4]))
    merged = merge_node_views(left, right)
    assert merged is not None
    by_doc = {s["document_id"]: s for s in merged["sources"]}
    assert set(by_doc) == {"doc-a", "doc-b", "doc-c"}
    assert by_doc["doc-b"]["chunks"] == [2, 9]


def test_source_cap_applies_and_records_overflow():
    merged = _view("doc-0", [0])
    for i in range(1, MAX_VIEW_SOURCES + 2):  # 10 documents in total
        merged = merge_node_views(merged, _view(f"doc-{i}", [i]))
    assert merged is not None
    assert len(merged["sources"]) == MAX_VIEW_SOURCES
    assert merged["sources_omitted"] == 2


def test_unresolved_view_yields_to_resolved():
    unresolved = {"status": "unresolved"}
    resolved = _view("doc-a", [1])
    assert merge_node_views(unresolved, resolved) == resolved
    assert merge_node_views(resolved, unresolved) == resolved


def test_empty_sides():
    view = _view("doc-a", [1])
    assert merge_node_views(None, None) is None
    assert merge_node_views(view, None) == view
    assert merge_node_views(None, view) == view


def test_iter_provenance_views_handles_both_forms():
    plain = _view("doc-a", [1])
    assert list(iter_provenance_views(plain)) == [plain]
    wrapped = merge_node_views(_view("doc-a", [1]), _view("doc-b", [2]))
    assert [v["document_id"] for v in iter_provenance_views(wrapped)] == ["doc-a", "doc-b"]
    assert list(iter_provenance_views(None)) == []


def test_provenance_weight_sums_wrapped_sources():
    wrapped = merge_node_views(_view("doc-a", [1, 2]), _view("doc-b", [3]))
    assert _provenance_weight({PROVENANCE_NODE_ATTR: wrapped}) == 3
    plain = _view("doc-a", [1, 2])
    assert _provenance_weight({PROVENANCE_NODE_ATTR: plain}) == 2


def test_write_ledger_sidecars_writes_verbatim_ledgers_and_manifest(tmp_path):
    ledger = ProvenanceLedger(
        document=DocumentOrigin(
            document_id="doc-abc",
            source="invoice.pdf",
            template_name="SampleCompany",
            template_schema_hash="hash-a",
        )
    )
    entries = [
        (
            {
                "index": 0,
                "document_id": "doc-abc",
                "source": "outputs/run_a",
                "template_name": "SampleCompany",
                "template_schema_hash": "hash-a",
                "converted_at": "2026-01-01T00:00:00",
                "graph": "outputs/run_a/docling_graph/graph.json",
            },
            ledger,
        ),
        (
            {
                "index": 1,
                "document_id": "",
                "source": "outputs/run_b",
                "template_name": "",
                "template_schema_hash": "",
                "converted_at": None,
                "graph": "outputs/run_b/graph.json",
            },
            None,
        ),
    ]
    provenance_dir = tmp_path / "provenance"
    manifest_path = write_ledger_sidecars(entries, provenance_dir)

    # Ledger written verbatim (round-trippable).
    ledger_path = provenance_dir / "doc-abc.json"
    assert ledger_path.is_file()
    round_tripped = ProvenanceLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
    assert round_tripped.document is not None
    assert round_tripped.document.document_id == "doc-abc"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents"][0]["ledger"] == "doc-abc.json"
    assert manifest["documents"][1]["ledger"] is None
    assert manifest["documents"][1]["source"] == "outputs/run_b"
