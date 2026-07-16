"""Unit tests for GraphMerger orchestration (core.merge.merger)."""

from typing import Any

import networkx as nx
import pytest

from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.merge import MergePolicy, merge_graphs
from docling_graph.core.provenance.models import DocumentOrigin, ProvenanceLedger
from docling_graph.exceptions import ConfigurationError

_AUDIT_ATTRS = {"merged_from", "__conflicts__"}


def _node(cls: str, **attrs: Any) -> dict[str, Any]:
    return {"label": cls, "type": "entity", "__class__": cls, **attrs}


def _graph_a() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("P_x", id="P_x", **_node("Person", name="Marie", role="physicist", city=""))
    g.add_node("O_a", id="O_a", **_node("Org", name="Sorbonne"))
    g.add_edge("P_x", "O_a", label="AFFILIATED", keywords=["research"])
    return g


def _graph_b() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("P_x", id="P_x", **_node("Person", name="Marie", role="chemist", city="Paris"))
    g.add_node("O_b", id="O_b", **_node("Org", name="Nobel Committee"))
    g.add_edge("P_x", "O_b", label="AWARDED_BY", keywords=["prize"])
    return g


def _content_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attrs.items() if k not in _AUDIT_ATTRS}


def _assert_same_content(a: nx.DiGraph, b: nx.DiGraph) -> None:
    assert set(a.nodes) == set(b.nodes)
    for node_id in a.nodes:
        assert _content_attrs(a.nodes[node_id]) == _content_attrs(b.nodes[node_id])
    assert set(a.edges) == set(b.edges)
    for edge in a.edges:
        assert a.edges[edge] == b.edges[edge]


def _write_run(
    tmp_path: Any,
    name: str,
    graph: nx.DiGraph,
    ledger: ProvenanceLedger | None = None,
) -> Any:
    run_dir = tmp_path / name
    graph_dir = run_dir / "docling_graph"
    JSONExporter().export(graph, graph_dir / "graph.json")
    if ledger is not None:
        (graph_dir / "provenance.json").write_text(
            ledger.model_dump_json(indent=2), encoding="utf-8"
        )
    return run_dir


def _ledger(document_id: str, source: str, schema_hash: str = "hash-1") -> ProvenanceLedger:
    return ProvenanceLedger(
        document=DocumentOrigin(
            document_id=document_id,
            source=source,
            template_name="Doc",
            template_schema_hash=schema_hash,
        )
    )


# ------------------------------------------------------------------ folding


def test_union_fills_empty_and_keeps_first_on_conflict():
    merged, report = merge_graphs([_graph_a(), _graph_b()])
    person = merged.nodes["P_x"]
    assert person["role"] == "physicist"  # keep-first
    assert person["city"] == "Paris"  # fill-empty
    assert merged.has_edge("P_x", "O_a") and merged.has_edge("P_x", "O_b")
    assert report.nodes_folded == 1
    assert report.field_conflicts == [
        {
            "node": "P_x",
            "field": "role",
            "kept": "physicist",
            "dropped": "chemist",
            "dropped_source": "graph-object-1",
        }
    ]
    # Folds are recorded, never silent.
    assert [e["source"] for e in person["merged_from"]] == [
        "graph-object-0",
        "graph-object-1",
    ]


def test_merge_is_not_commutative_and_says_so():
    forward, _ = merge_graphs([_graph_a(), _graph_b()])
    backward, _ = merge_graphs([_graph_b(), _graph_a()])
    assert forward.nodes["P_x"]["role"] == "physicist"
    assert backward.nodes["P_x"]["role"] == "chemist"
    assert forward.nodes["P_x"]["role"] != backward.nodes["P_x"]["role"]


def test_single_input_round_trips_content():
    original = _graph_a()
    merged, report = merge_graphs([original])
    _assert_same_content(merged, _graph_a())
    assert report.nodes_folded == 0
    assert report.identity_source == "node_ids"


def test_duplicate_graph_object_absorbed_and_idempotent():
    graph = _graph_a()
    merged_twice, report = merge_graphs([graph, graph])
    merged_once, _ = merge_graphs([graph])
    assert report.duplicates_absorbed == ["graph-object-1"]
    _assert_same_content(merged_twice, merged_once)
    _assert_same_content(merged_twice, _graph_a())  # merge(A, A) == A


def test_duplicate_path_input_absorbed(tmp_path):
    run = _write_run(tmp_path, "run_a", _graph_a())
    merged, report = merge_graphs([run, run])
    assert len(report.duplicates_absorbed) == 1
    assert merged.number_of_nodes() == _graph_a().number_of_nodes()


def test_duplicate_document_id_absorbed_across_paths(tmp_path):
    ledger = _ledger("doc-same", "invoice.pdf")
    run_a = _write_run(tmp_path, "run_a", _graph_a(), ledger)
    run_b = _write_run(tmp_path, "run_b", _graph_a(), ledger)
    merged, report = merge_graphs([run_a, run_b])
    assert report.duplicates_absorbed == [str(run_b)]
    assert merged.number_of_nodes() == _graph_a().number_of_nodes()


def test_left_fold_associativity_modulo_audit_trail():
    g_c = nx.DiGraph()
    g_c.add_node("P_x", id="P_x", **_node("Person", name="Marie", role="laureate", born="1867"))
    g_c.add_node("O_c", id="O_c", **_node("Org", name="Radium Institute"))
    g_c.add_edge("P_x", "O_c", label="FOUNDED")

    all_at_once, _ = merge_graphs([_graph_a(), _graph_b(), g_c])
    intermediate, _ = merge_graphs([_graph_a(), _graph_b()])
    staged, _ = merge_graphs([intermediate, g_c])
    _assert_same_content(all_at_once, staged)


def test_richest_precedence_lets_richer_node_win_conflicts():
    policy = MergePolicy(precedence="richest")
    merged, _ = merge_graphs([_graph_a(), _graph_b()], policy=policy)
    # graph_b's P_x carries more filled attrs (city) -> it becomes the base.
    assert merged.nodes["P_x"]["role"] == "chemist"
    # Deterministic either way: input order breaks exact ties.
    merged_again, _ = merge_graphs([_graph_a(), _graph_b()], policy=policy)
    assert merged.nodes["P_x"] == merged_again.nodes["P_x"]


def test_edge_label_conflict_keeps_first_and_records():
    g1 = nx.DiGraph()
    g1.add_node("A", id="A", **_node("Org", name="Acme"))
    g1.add_node("B", id="B", **_node("Person", name="Ada"))
    g1.add_edge("A", "B", label="EMPLOYS")
    g2 = nx.DiGraph()
    g2.add_node("A", id="A", **_node("Org", name="Acme"))
    g2.add_node("B", id="B", **_node("Person", name="Ada"))
    g2.add_edge("A", "B", label="HIRED")

    merged, report = merge_graphs([g1, g2])
    edge = merged.edges["A", "B"]
    assert edge["label"] == "EMPLOYS"
    assert edge["also_labels"] == ["HIRED"]
    assert report.edge_label_conflicts == [
        {
            "source": "A",
            "kept_label": "EMPLOYS",
            "dropped_labels": ["HIRED"],
            "target": "B",
            "dropped_source": "graph-object-1",
        }
    ]


def test_keep_all_conflicts_land_on_the_node():
    merged, _ = merge_graphs([_graph_a(), _graph_b()], policy=MergePolicy(conflicts="keep-all"))
    assert merged.nodes["P_x"]["__conflicts__"] == [
        {"field": "role", "value": "chemist", "source": "graph-object-1"}
    ]


# ------------------------------------------------------- provenance wrapping


def test_cross_document_provenance_wraps_on_fold(tmp_path):
    from docling_graph.core.provenance.identity import PROVENANCE_NODE_ATTR

    g1 = _graph_a()
    g1.nodes["P_x"][PROVENANCE_NODE_ATTR] = {
        "document_id": "doc-a",
        "match": "verbatim",
        "chunks": [1],
        "pages": [1],
    }
    g2 = _graph_b()
    g2.nodes["P_x"][PROVENANCE_NODE_ATTR] = {
        "document_id": "doc-b",
        "match": "verbatim",
        "chunks": [1],
        "pages": [2],
    }
    run_a = _write_run(tmp_path, "run_a", g1, _ledger("doc-a", "a.pdf"))
    run_b = _write_run(tmp_path, "run_b", g2, _ledger("doc-b", "b.pdf"))

    merged, _ = merge_graphs([run_a, run_b])
    view = merged.nodes["P_x"][PROVENANCE_NODE_ATTR]
    assert view["multi_document"] is True
    assert [s["document_id"] for s in view["sources"]] == ["doc-a", "doc-b"]


# -------------------------------------------------------------- audit trail


def test_merged_aliases_survive_union_fold():
    """Alias audit records from previous merges are unioned on fold, exactly
    like merged_from (both are meta attrs the field fold skips)."""
    g1 = _graph_a()
    g1.nodes["P_x"]["merged_aliases"] = [{"id": "Person_old1", "name": "Marie C."}]
    g2 = _graph_b()
    g2.nodes["P_x"]["merged_aliases"] = [{"id": "Person_old2", "name": "M. Curie"}]
    merged, _report = merge_graphs([g1, g2])
    assert {a["id"] for a in merged.nodes["P_x"]["merged_aliases"]} == {
        "Person_old1",
        "Person_old2",
    }


# -------------------------------------------------------------- schema gate


def test_template_schema_mismatch_raises_by_default(tmp_path):
    run_a = _write_run(tmp_path, "run_a", _graph_a(), _ledger("doc-a", "a.pdf", "hash-a"))
    run_b = _write_run(tmp_path, "run_b", _graph_b(), _ledger("doc-b", "b.pdf", "hash-b"))
    with pytest.raises(ConfigurationError, match="different template schemas"):
        merge_graphs([run_a, run_b])


def test_template_schema_mismatch_downgrades_to_warning(tmp_path):
    run_a = _write_run(tmp_path, "run_a", _graph_a(), _ledger("doc-a", "a.pdf", "hash-a"))
    run_b = _write_run(tmp_path, "run_b", _graph_b(), _ledger("doc-b", "b.pdf", "hash-b"))
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(strict_template_check=False))
    assert merged.number_of_nodes() == 3
    assert any("Template schemas differ" in w for w in report.warnings)


def test_missing_schema_hashes_skip_gate_with_warning():
    _merged, report = merge_graphs([_graph_a(), _graph_b()])
    assert any("compatibility check skipped" in w for w in report.warnings)


def test_explicit_template_joins_the_schema_gate(tmp_path):
    """--template is hash-checked against the inputs like any other source."""
    from docling_graph.core.provenance.models import template_schema_hash
    from tests.fixtures.sample_templates.test_template import SampleCompany

    run = _write_run(tmp_path, "run_a", _graph_a(), _ledger("doc-a", "a.pdf", "hash-other"))
    with pytest.raises(ConfigurationError, match="different template schemas"):
        merge_graphs([run], template=SampleCompany)

    _merged, report = merge_graphs(
        [run], template=SampleCompany, policy=MergePolicy(strict_template_check=False)
    )
    assert any("Template schemas differ" in w for w in report.warnings)

    # A matching hash passes the gate silently.
    matching = _write_run(
        tmp_path,
        "run_b",
        _graph_a(),
        _ledger("doc-b", "b.pdf", template_schema_hash(SampleCompany)),
    )
    _merged, report = merge_graphs([matching], template=SampleCompany)
    assert not any("Template schemas differ" in w for w in report.warnings)


def test_merged_export_stamped_with_template_hash():
    """A template-driven merge stamps the template's own schema hash."""
    from docling_graph.core.provenance.models import template_schema_hash
    from tests.fixtures.sample_templates.test_template import SampleCompany

    graph = nx.DiGraph()
    graph.add_node("C", id="C", **_node("SampleCompany", company_name="Acme", industry="Tech"))
    merged, _report = merge_graphs([graph], template=SampleCompany)
    assert merged.graph["template_schema_hash"] == template_schema_hash(SampleCompany)
    assert merged.graph["template_name"] == "SampleCompany"


# ------------------------------------------------------------ skolemization


def _root_graph(reference: str, child: str) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("Doc_root", id="Doc_root", **_node("Doc", reference=reference))
    g.add_node(f"Item_{child}", id=f"Item_{child}", **_node("Item", name=child))
    g.add_edge("Doc_root", f"Item_{child}", label="HAS_ITEM")
    g.graph["id_fields_map"] = {"Doc": ["reference"], "Item": ["name"]}
    return g


def test_root_stem_collision_is_skolemized(tmp_path):
    """Two provably distinct documents both named invoice.pdf: the stem-derived
    root identity must not fuse them."""
    run_a = _write_run(
        tmp_path, "run_a", _root_graph("invoice", "alpha"), _ledger("doc-aaaa", "/x/invoice.pdf")
    )
    run_b = _write_run(
        tmp_path, "run_b", _root_graph("invoice", "beta"), _ledger("doc-bbbb", "/y/invoice.pdf")
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    docs = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Doc"]
    assert len(docs) == 2
    assert len(report.root_skolemized) == 1
    record = report.root_skolemized[0]
    assert record["original_id"] == "Doc_root"
    assert record["skolemized_id"] == "Doc_root__doc_doc-bbbb"  # __doc_<document_id[:8]>
    assert record["identity_value"] == "invoice"
    # The skolemized root kept its own children.
    skolemized = record["skolemized_id"]
    assert merged.has_edge(skolemized, "Item_beta")
    assert merged.has_edge("Doc_root", "Item_alpha")


def test_content_derived_root_identity_is_never_skolemized(tmp_path):
    """Same collision shape but the identity is document data -> normal fold."""
    run_a = _write_run(
        tmp_path, "run_a", _root_graph("REF-123", "alpha"), _ledger("doc-aaaa", "/x/invoice.pdf")
    )
    run_b = _write_run(
        tmp_path, "run_b", _root_graph("REF-123", "beta"), _ledger("doc-bbbb", "/y/scan.pdf")
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    docs = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Doc"]
    assert len(docs) == 1
    assert report.root_skolemized == []


def test_root_collision_without_ledgers_warns_and_merges(tmp_path):
    run_a = _write_run(tmp_path, "run_a", _root_graph("invoice", "alpha"))
    run_b = _write_run(tmp_path, "run_b", _root_graph("invoice", "beta"))
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    docs = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Doc"]
    assert len(docs) == 1
    assert any("no provenance ledgers" in w for w in report.warnings)


def test_root_stem_collision_skolemized_without_id_fields(tmp_path):
    """v1 exports without --template declare no id fields: the stem match must
    fall back to scanning scalar string attributes instead of silently merging."""
    g_a, g_b = _root_graph("invoice", "alpha"), _root_graph("invoice", "beta")
    del g_a.graph["id_fields_map"], g_b.graph["id_fields_map"]
    run_a = _write_run(tmp_path, "run_a", g_a, _ledger("doc-aaaa", "/x/invoice.pdf"))
    run_b = _write_run(tmp_path, "run_b", g_b, _ledger("doc-bbbb", "/y/invoice.pdf"))
    merged, report = merge_graphs([run_a, run_b])
    assert report.identity_source == "node_ids"
    docs = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Doc"]
    assert len(docs) == 2
    assert len(report.root_skolemized) == 1
    assert report.root_skolemized[0]["identity_value"] == "invoice"


def test_root_collision_without_id_fields_or_stem_match_warns(tmp_path):
    """No id fields AND no stem match: the fusion cannot be ruled out, so it
    must be loud instead of a bare skip."""
    g_a, g_b = _root_graph("REF-123", "alpha"), _root_graph("REF-123", "beta")
    del g_a.graph["id_fields_map"], g_b.graph["id_fields_map"]
    run_a = _write_run(tmp_path, "run_a", g_a, _ledger("doc-aaaa", "/x/invoice.pdf"))
    run_b = _write_run(tmp_path, "run_b", g_b, _ledger("doc-bbbb", "/y/scan.pdf"))
    merged, report = merge_graphs([run_a, run_b])
    docs = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Doc"]
    assert len(docs) == 1
    assert report.root_skolemized == []
    warning = next(w for w in report.warnings if "declares no id fields" in w)
    assert str(run_a) in warning and str(run_b) in warning


def test_remerge_of_skolemized_export_keeps_roots_apart(tmp_path):
    """Skolemized roots must survive the auto-rekey of a re-merge: the skolem
    stamp is part of the fingerprint, so the root never recomputes back to its
    colliding base id and re-fuses (silent cross-document data fusion)."""
    run_a = _write_run(
        tmp_path, "run_a", _root_graph("invoice", "alpha"), _ledger("doc-aaaa", "/x/invoice.pdf")
    )
    run_b = _write_run(
        tmp_path, "run_b", _root_graph("invoice", "beta"), _ledger("doc-bbbb", "/y/invoice.pdf")
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    assert len(report.root_skolemized) == 1
    merged_run = _write_run(tmp_path, "merged_run", merged)

    run_c = _write_run(
        tmp_path, "run_c", _root_graph("REF-999", "gamma"), _ledger("doc-cccc", "/z/other.pdf")
    )
    # Default policy: the merged export is format-v2, so re-keying auto-enables.
    remerged, remerge_report = merge_graphs([merged_run, run_c])
    assert remerge_report.rekeyed is True
    docs = {n: d for n, d in remerged.nodes(data=True) if d["__class__"] == "Doc"}
    assert len(docs) == 3
    invoice_roots = {n: d for n, d in docs.items() if d["reference"] == "invoice"}
    assert len(invoice_roots) == 2
    # Each colliding root kept its own children through the re-merge.
    children_by_root = {
        str(remerged.nodes[child]["name"])
        for root in invoice_roots
        for child in remerged.successors(root)
    }
    assert children_by_root == {"alpha", "beta"}


# ------------------------------------------- cross-document conflict splits


def _invoice_graph(root_id: str, desc: str, qty: float, party: str | None = None) -> nx.DiGraph:
    """Root + weak-identity child (line_number is only locally unique)."""
    g = nx.DiGraph()
    g.add_node(root_id, id=root_id, **_node("Invoice", reference=root_id))
    g.add_node(
        "Line_1",
        id="Line_1",
        **_node("LineItem", line_number="1", description=desc, quantity=qty),
    )
    g.add_edge(root_id, "Line_1", label="CONTAINS_LINE")
    if party is not None:
        g.add_node("Party_p", id="Party_p", **_node("Party", name=party))
        g.add_edge(root_id, "Party_p", label="ISSUED_BY")
    g.graph["id_fields_map"] = {
        "Invoice": ["reference"],
        "LineItem": ["line_number"],
        "Party": ["name"],
    }
    return g


def test_conflicting_child_collision_across_documents_splits(tmp_path):
    """Line 1 of two unrelated invoices mints the same ID (weak local identity):
    folding would clobber amounts and share children across roots — split."""
    run_a = _write_run(
        tmp_path,
        "run_a",
        _invoice_graph("Inv_A", "Gardenwork", 28.0, party="Acme"),
        _ledger("doc-aaaa", "/x/a.jpg"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        _invoice_graph("Inv_B", "Keyboard", 1.0, party="Acme"),
        _ledger("doc-bbbb", "/y/b.docx"),
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))

    lines = {n: d for n, d in merged.nodes(data=True) if d["__class__"] == "LineItem"}
    assert set(lines) == {"Line_1", "Line_1__doc_doc-bbbb"}
    assert len(report.cross_document_splits) == 1
    record = report.cross_document_splits[0]
    assert record["original_id"] == "Line_1"
    assert record["split_id"] == "Line_1__doc_doc-bbbb"
    assert record["conflicting_fields"] == ["quantity"]  # description combines, never conflicts
    assert record["document_id"] == "doc-bbbb"
    # Each line keeps its own values and hangs under its own invoice only.
    assert merged.nodes["Line_1"]["quantity"] == 28.0
    assert merged.nodes["Line_1"]["description"] == "Gardenwork"
    assert merged.nodes["Line_1__doc_doc-bbbb"]["quantity"] == 1.0
    assert merged.nodes["Line_1__doc_doc-bbbb"]["skolem_document_id"] == "doc-bbbb"
    assert merged.has_edge("Inv_A", "Line_1")
    assert merged.has_edge("Inv_B", "Line_1__doc_doc-bbbb")
    assert not merged.has_edge("Inv_B", "Line_1")
    assert not merged.has_edge("Inv_A", "Line_1__doc_doc-bbbb")
    # The compatible collision (shared Party, no conflicting fields) still folds.
    parties = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Party"]
    assert parties == ["Party_p"]


def test_same_document_reextraction_conflicts_fold_not_split(tmp_path):
    """A JPG and a DOCX of the SAME invoice share the root id: their same-ID
    children are the same instance, so extraction noise folds under keep-first."""
    run_a = _write_run(
        tmp_path,
        "run_a",
        _invoice_graph("Inv_X", "Gardenwork", 28.0),
        _ledger("doc-aaaa", "/x/inv.jpg"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        _invoice_graph("Inv_X", "Gardenwork", 26.0),
        _ledger("doc-bbbb", "/y/inv.docx"),
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    assert report.cross_document_splits == []
    lines = [n for n, d in merged.nodes(data=True) if d["__class__"] == "LineItem"]
    assert lines == ["Line_1"]
    assert merged.nodes["Line_1"]["quantity"] == 28.0  # keep-first
    assert any(c["field"] == "quantity" for c in report.field_conflicts)


def test_conflicting_collision_without_ledgers_splits_with_src_suffix():
    """No ledgers: the structural + content evidence still stands; the suffix
    falls back to the input position."""
    merged, report = merge_graphs(
        [
            _invoice_graph("Inv_A", "Gardenwork", 28.0),
            _invoice_graph("Inv_B", "Keyboard", 1.0),
        ],
        policy=MergePolicy(rekey=False),
    )
    assert len(report.cross_document_splits) == 1
    assert report.cross_document_splits[0]["split_id"] == "Line_1__src_1"
    assert merged.nodes["Line_1__src_1"]["skolem_document_id"] == "input-1"


def test_remerge_of_split_export_keeps_instances_apart(tmp_path):
    """Split nodes carry the skolem stamp, so the auto-rekey of a re-merge
    never recomputes them back onto the colliding base id — and re-merging a
    constituent input converges instead of corrupting."""
    run_a = _write_run(
        tmp_path,
        "run_a",
        _invoice_graph("Inv_A", "Gardenwork", 28.0),
        _ledger("doc-aaaa", "/x/a.jpg"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        _invoice_graph("Inv_B", "Keyboard", 1.0),
        _ledger("doc-bbbb", "/y/b.docx"),
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    assert len(report.cross_document_splits) == 1
    merged_run = _write_run(tmp_path, "merged_run", merged)

    # Default policy: the merged export is format-v2, so re-keying auto-enables.
    remerged, re_report = merge_graphs([merged_run, run_b])
    assert re_report.rekeyed is True
    lines = {
        d["description"]: d for _n, d in remerged.nodes(data=True) if d["__class__"] == "LineItem"
    }
    assert set(lines) == {"Gardenwork", "Keyboard"}  # never re-fused into a chimera
    assert lines["Gardenwork"]["quantity"] == 28.0
    assert lines["Keyboard"]["quantity"] == 1.0


def test_same_class_conflict_splits_compatible_sibling_collision(tmp_path):
    """Once line 1 proves LineItem IDs under-determine instances across two
    documents, line 2 of the same pair splits too — even though its values
    agree — so unrelated invoices never share a line-item child. Classes
    without a conflict (the shared Party) are untouched by the contagion."""

    def invoice(root_id: str, qty1: float) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(root_id, id=root_id, **_node("Invoice", reference=root_id))
        g.add_node("Line_1", id="Line_1", **_node("LineItem", line_number="1", quantity=qty1))
        g.add_node("Line_2", id="Line_2", **_node("LineItem", line_number="2", quantity=1.0))
        g.add_node("Party_p", id="Party_p", **_node("Party", name="Acme"))
        g.add_edge(root_id, "Line_1", label="CONTAINS_LINE")
        g.add_edge(root_id, "Line_2", label="CONTAINS_LINE")
        g.add_edge(root_id, "Party_p", label="ISSUED_BY")
        g.graph["id_fields_map"] = {
            "Invoice": ["reference"],
            "LineItem": ["line_number"],
            "Party": ["name"],
        }
        return g

    run_a = _write_run(tmp_path, "run_a", invoice("Inv_A", 28.0), _ledger("doc-aaaa", "/x/a.jpg"))
    run_b = _write_run(tmp_path, "run_b", invoice("Inv_B", 1.0), _ledger("doc-bbbb", "/y/b.docx"))
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))

    splits = {r["original_id"]: r for r in report.cross_document_splits}
    assert set(splits) == {"Line_1", "Line_2"}
    assert splits["Line_1"]["reason"] == "field-conflict"
    assert splits["Line_1"]["conflicting_fields"] == ["quantity"]
    assert splits["Line_2"]["reason"] == "same-class-conflict"
    assert splits["Line_2"]["conflicting_fields"] == []
    assert splits["Line_2"]["triggered_by"] == "Line_1"
    # Each invoice keeps its own line 2; the value-identical pair never fused.
    assert merged.has_edge("Inv_A", "Line_2")
    assert merged.has_edge("Inv_B", "Line_2__doc_doc-bbbb")
    assert not merged.has_edge("Inv_B", "Line_2")
    parties = [n for n, d in merged.nodes(data=True) if d["__class__"] == "Party"]
    assert parties == ["Party_p"]


def test_formatting_noise_folds_shared_entity_instead_of_splitting(tmp_path):
    """The same real-world entity extracted from a PDF and a JPG differs only
    in phone spacing — OCR formatting noise, not a conflict. The Party folds
    (and enriches) instead of splitting into per-document copies."""

    def invoice(root_id: str, phone: str, country: str | None) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(root_id, id=root_id, **_node("Invoice", reference=root_id))
        g.add_node(
            "Party_p",
            id="Party_p",
            **_node("Party", name="Robert Schneider AG", phone=phone, country=country),
        )
        g.add_edge(root_id, "Party_p", label="ISSUED_BY")
        g.graph["id_fields_map"] = {"Invoice": ["reference"], "Party": ["name"]}
        return g

    run_a = _write_run(
        tmp_path,
        "run_a",
        invoice("Inv_A", "059/987 65 40", "Switzerland"),
        _ledger("doc-aaaa", "/x/a.pdf"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        invoice("Inv_B", "059/9876540", None),
        _ledger("doc-bbbb", "/y/b.jpg"),
    )
    merged, report = merge_graphs([run_a, run_b], policy=MergePolicy(rekey=False))
    assert report.cross_document_splits == []
    assert report.field_conflicts == []
    party = merged.nodes["Party_p"]
    assert party["phone"] == "059/987 65 40"  # survivor form kept verbatim
    assert party["country"] == "Switzerland"  # fill-empty enrichment
    assert merged.has_edge("Inv_A", "Party_p") and merged.has_edge("Inv_B", "Party_p")


def test_variants_mode_reifies_suppressed_values():
    """conflicts=variants: the canonical node is byte-identical to keep-first;
    each conflicting source's suppressed values become a <Class>Variant
    sub-node hanging off the canonical node."""
    merged, report = merge_graphs(
        [_graph_a(), _graph_b()], policy=MergePolicy(conflicts="variants")
    )
    person = merged.nodes["P_x"]
    assert person["role"] == "physicist"  # keep-first winner unchanged
    assert person["city"] == "Paris"  # fill-empty unchanged
    assert "__conflicts__" not in person
    variant = merged.nodes["P_x__var_in1"]
    assert variant["type"] == "variant"
    assert variant["label"] == "PersonVariant"
    assert variant["__class__"] == "Person"
    assert variant["role"] == "chemist"
    assert variant["variant_of"] == "P_x"
    assert variant["variant_document_id"] == "input-1"
    edge = merged.edges["P_x", "P_x__var_in1"]
    assert edge["label"] == "HAS_CONFLICT_VARIANT"
    assert edge["fields"] == ["role"]
    assert edge["document_id"] == "input-1"
    assert report.conflict_variants == 1
    # The report still records the conflict exactly like keep-first.
    assert [c["field"] for c in report.field_conflicts] == ["role"]


def test_keep_first_and_keep_all_create_no_variant_nodes():
    for policy in (MergePolicy(), MergePolicy(conflicts="keep-all")):
        merged, report = merge_graphs([_graph_a(), _graph_b()], policy=policy)
        assert report.conflict_variants == 0
        assert all(d.get("type") != "variant" for _n, d in merged.nodes(data=True))


def test_variants_mode_does_not_affect_cross_document_splits(tmp_path):
    """Split collisions never fold, so they never spawn variants: different
    instances stay whole nodes, not canonical-plus-variant."""
    run_a = _write_run(
        tmp_path,
        "run_a",
        _invoice_graph("Inv_A", "Gardenwork", 28.0),
        _ledger("doc-aaaa", "/x/a.jpg"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        _invoice_graph("Inv_B", "Keyboard", 1.0),
        _ledger("doc-bbbb", "/y/b.docx"),
    )
    _merged, report = merge_graphs(
        [run_a, run_b], policy=MergePolicy(conflicts="variants", rekey=False)
    )
    assert len(report.cross_document_splits) == 1
    assert report.conflict_variants == 0


def test_variants_are_stable_across_remerge_and_rekey(tmp_path):
    """Variant ids are derived from (base id, document id): auto-rekey moves
    them in lockstep with their re-keyed base, and re-merging a constituent
    input re-detects the conflict onto the existing variant instead of
    minting a duplicate."""
    run_a = _write_run(
        tmp_path,
        "run_a",
        _invoice_graph("Inv_X", "Gardenwork", 28.0),
        _ledger("doc-aaaa", "/x/inv.jpg"),
    )
    run_b = _write_run(
        tmp_path,
        "run_b",
        _invoice_graph("Inv_X", "Gardenwork", 26.0),
        _ledger("doc-bbbb", "/y/inv.docx"),
    )
    merged, report = merge_graphs(
        [run_a, run_b], policy=MergePolicy(conflicts="variants", rekey=False)
    )
    assert report.conflict_variants == 1
    assert merged.nodes["Line_1__var_doc-bbbb"]["quantity"] == 26.0
    merged_run = _write_run(tmp_path, "merged_run", merged)

    # Default rekey auto-enables on the format-v2 export.
    remerged, re_report = merge_graphs(
        [merged_run, run_b], policy=MergePolicy(conflicts="variants")
    )
    assert re_report.rekeyed is True
    variants = {n: d for n, d in remerged.nodes(data=True) if d.get("type") == "variant"}
    assert len(variants) == 1
    ((variant_id, variant),) = variants.items()
    assert variant["quantity"] == 26.0
    assert re_report.conflict_variants == 0  # existing variant reused, not duplicated
    bases = [
        n
        for n, d in remerged.nodes(data=True)
        if d["__class__"] == "LineItem" and d.get("type") != "variant"
    ]
    assert len(bases) == 1
    assert remerged.edges[bases[0], variant_id]["label"] == "HAS_CONFLICT_VARIANT"
    assert variant["variant_of"] == bases[0]


def test_alias_candidate_stubs_surface_field_conflicts():
    """Stubs list the content fields a confirmed merge would contradict, so a
    human can spot distinct instances sharing an identifier ('3139' vs
    'INV-3139' with different currencies) before flipping confirm. Identity
    fields and formatting-noise differences are excluded."""
    g = nx.DiGraph()
    g.add_node(
        "B_1",
        id="B_1",
        **_node("BillingDocument", document_number="3139", currency="CHF", notes="ACME"),
    )
    g.add_node(
        "B_2",
        id="B_2",
        **_node("BillingDocument", document_number="INV-3139", currency="EUR", notes="Acme"),
    )
    g.graph["id_fields_map"] = {"BillingDocument": ["document_number"]}
    _merged, report = merge_graphs([g], policy=MergePolicy(rekey=False))
    assert len(report.alias_candidates) == 1
    stub = report.alias_candidates[0]
    assert stub["keep_display"] == "3139"
    assert stub["merge_displays"] == ["INV-3139"]
    assert stub["field_conflicts"] == ["currency"]
    assert stub["confirm"] is False


# ------------------------------------------------------------------ identity


def test_alias_pass_skipped_without_id_fields_source():
    _merged, report = merge_graphs([_graph_a(), _graph_b()])
    assert report.identity_source == "node_ids"
    assert report.alias_candidates == []
    assert any("Alias reconciliation skipped" in w for w in report.warnings)


def test_rekey_requires_an_id_fields_source():
    with pytest.raises(ConfigurationError, match="id-fields source"):
        merge_graphs([_graph_a(), _graph_b()], policy=MergePolicy(rekey=True))


def test_identity_source_from_dense_ledger(tmp_path):
    from docling_graph.core.provenance.models import NodeProvenance

    ledger = _ledger("doc-a", "a.pdf")
    ledger.nodes["Doc|reference=x"] = NodeProvenance(
        identity_key="Doc|reference=x", node_type="Doc", ids={"reference": "x"}
    )
    graph = _root_graph("x", "alpha")
    del graph.graph["id_fields_map"]  # simulate a v1 export
    run = _write_run(tmp_path, "run_a", graph, ledger)
    _merged, report = merge_graphs([run])
    assert report.identity_source == "ledger"


def test_identity_source_template_beats_v2():
    from tests.fixtures.sample_templates.test_template import SampleCompany

    graph = nx.DiGraph()
    graph.add_node("C", id="C", **_node("SampleCompany", company_name="Acme", industry="Tech"))
    graph.graph["id_fields_map"] = {"SampleCompany": ["WRONG"]}
    _merged, report = merge_graphs([graph], template=SampleCompany, policy=MergePolicy(rekey=False))
    assert report.identity_source == "template"


def test_template_accepts_dotted_string_and_auto_rekeys():
    graph = nx.DiGraph()
    graph.add_node("C", id="C", **_node("SampleCompany", company_name="Acme", industry="Tech"))
    merged, report = merge_graphs(
        [graph], template="tests.fixtures.sample_templates.test_template.SampleCompany"
    )
    assert report.identity_source == "template"
    assert report.rekeyed is True and report.rekeyed_changed == 1
    node_id = next(iter(merged.nodes))
    assert node_id.startswith("SampleCompany_") and node_id != "C"


def test_invalid_template_type_rejected():
    with pytest.raises(ConfigurationError, match="template"):
        merge_graphs([_graph_a()], template=42)  # type: ignore[arg-type]


# ------------------------------------------------------------------ metadata


def test_merge_metadata_shape():
    merged, report = merge_graphs([_graph_a(), _graph_b()])
    meta = merged.graph["merge"]
    assert set(meta) == {
        "sources",
        "identity_source",
        "nodes_folded",
        "field_conflicts",
        "conflict_variants",
        "edge_label_conflicts",
        "cross_document_splits",
        "alias_candidates",
        "alias_merged",
        "rekeyed",
        "rekeyed_changed",
    }
    assert merged.graph["format"] == "docling-graph/v2"
    assert len(meta["sources"]) == 2
    assert meta["nodes_folded"] == 1
    assert report.node_count == merged.number_of_nodes()
    assert report.edge_count == merged.number_of_edges()
    assert report.node_types == {"Person": 1, "Org": 2}


def test_empty_inputs_rejected():
    with pytest.raises(ConfigurationError, match="at least one input"):
        merge_graphs([])


def test_inputs_are_not_mutated_by_the_merge():
    graph = _graph_a()
    before_nodes = {n: dict(d) for n, d in graph.nodes(data=True)}
    merge_graphs([graph, _graph_b()])
    assert {n: dict(d) for n, d in graph.nodes(data=True)} == before_nodes
