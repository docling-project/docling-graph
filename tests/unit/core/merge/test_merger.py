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
        "edge_label_conflicts",
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
