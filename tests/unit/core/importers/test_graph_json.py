"""Unit tests for the graph.json loader (core.importers.graph_json)."""

import json
from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.importers.graph_json import (
    load_graph_input,
    load_graph_json,
    load_sibling_ledger,
    resolve_graph_path,
)
from docling_graph.core.provenance.identity import PROVENANCE_NODE_ATTR
from docling_graph.core.provenance.models import DocumentOrigin, ProvenanceLedger
from docling_graph.exceptions import ConfigurationError
from tests.fixtures.sample_templates.test_template import SampleCompany, SamplePerson


def _sample_graph() -> nx.DiGraph:
    company = SampleCompany(
        company_name="Acme",
        industry="Tech",
        founded_year=1999,
        employees=[
            SamplePerson(first_name="Ada", last_name="Byron", email="ada@acme.com"),
        ],
    )
    graph, _meta = GraphConverter().pydantic_list_to_graph([company])
    return graph


def _export(graph: nx.DiGraph, path: Path) -> Path:
    JSONExporter().export(graph, path)
    return path


def test_round_trip_preserves_nodes_edges_and_provenance(tmp_path):
    graph = _sample_graph()
    some_node = next(iter(graph.nodes))
    graph.nodes[some_node][PROVENANCE_NODE_ATTR] = {
        "document_id": "doc-a",
        "match": "verbatim",
        "chunks": [0, 3],
        "pages": [1],
    }
    path = _export(graph, tmp_path / "graph.json")

    loaded = load_graph_json(path)
    assert set(loaded.nodes) == set(graph.nodes)
    assert set(loaded.edges) == set(graph.edges)
    for node_id, attrs in graph.nodes(data=True):
        assert loaded.nodes[node_id] == attrs
        assert loaded.nodes[node_id]["id"] == node_id  # id attr contract preserved
    for source, target, attrs in graph.edges(data=True):
        assert loaded.edges[source, target] == attrs


def test_round_trip_preserves_v2_graph_metadata(tmp_path):
    graph = _sample_graph()
    path = _export(graph, tmp_path / "graph.json")
    loaded = load_graph_json(path)
    assert loaded.graph["format"] == "docling-graph/v2"
    assert loaded.graph["template_name"] == "SampleCompany"
    assert loaded.graph["id_fields_map"]["SampleCompany"] == ["company_name"]
    assert loaded.graph["template_schema_hash"]


def test_v1_export_without_graph_key_loads_with_empty_metadata(tmp_path):
    graph = _sample_graph()
    path = _export(graph, tmp_path / "graph.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["graph"]  # simulate a pre-v2 export
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_graph_json(path)
    assert loaded.graph == {}
    assert set(loaded.nodes) == set(graph.nodes)


def test_sniff_rejects_arbitrary_json(tmp_path):
    path = tmp_path / "not_a_graph.json"
    path.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="nodes"):
        load_graph_json(path)


def test_sniff_rejects_docling_document_json(tmp_path):
    path = tmp_path / "document.json"
    path.write_text(
        json.dumps({"schema_name": "DoclingDocument", "texts": [], "body": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_graph_json(path)


def test_empty_graph_export_rejected(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no nodes"):
        load_graph_json(path)


def test_dangling_edge_endpoint_rejected(tmp_path):
    path = tmp_path / "dangling.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "A", "name": "a"}],
                "edges": [{"source": "A", "target": "MISSING", "label": "REL"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="endpoint"):
        load_graph_json(path)


def test_resolve_run_dir_finds_export_stage_layout(tmp_path):
    graph = _sample_graph()
    run_dir = tmp_path / "run"
    _export(graph, run_dir / "docling_graph" / "graph.json")
    assert resolve_graph_path(run_dir) == run_dir / "docling_graph" / "graph.json"


def test_resolve_run_dir_falls_back_to_top_level_graph_json(tmp_path):
    graph = _sample_graph()
    run_dir = tmp_path / "run"
    _export(graph, run_dir / "graph.json")
    assert resolve_graph_path(run_dir) == run_dir / "graph.json"


def test_resolve_rejects_directory_without_export(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    with pytest.raises(ConfigurationError, match=r"docling_graph/graph\.json"):
        resolve_graph_path(tmp_path / "empty_dir")


@pytest.mark.parametrize("suffix", [".csv", ".cypher"])
def test_resolve_rejects_lossy_formats_with_pointer_to_graph_json(tmp_path, suffix):
    path = tmp_path / f"nodes{suffix}"
    path.write_text("id,label\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"graph\.json"):
        resolve_graph_path(path)


def test_resolve_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        resolve_graph_path(tmp_path / "nope.json")


def test_load_graph_input_reads_sibling_ledger(tmp_path):
    graph = _sample_graph()
    run_dir = tmp_path / "run"
    graph_path = _export(graph, run_dir / "docling_graph" / "graph.json")
    ledger = ProvenanceLedger(
        document=DocumentOrigin(
            document_id="doc-abc",
            source="invoice.pdf",
            template_name="SampleCompany",
            template_schema_hash="deadbeef",
        )
    )
    (graph_path.parent / "provenance.json").write_text(
        ledger.model_dump_json(indent=2), encoding="utf-8"
    )

    loaded_graph, loaded_ledger, loaded_path = load_graph_input(run_dir)
    assert loaded_path == graph_path
    assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
    assert loaded_ledger is not None
    assert loaded_ledger.document is not None
    assert loaded_ledger.document.document_id == "doc-abc"


def test_missing_sibling_ledger_returns_none(tmp_path):
    graph = _sample_graph()
    graph_path = _export(graph, tmp_path / "graph.json")
    assert load_sibling_ledger(graph_path) is None


def test_corrupt_sibling_ledger_degrades_to_none(tmp_path):
    graph = _sample_graph()
    graph_path = _export(graph, tmp_path / "graph.json")
    (graph_path.parent / "provenance.json").write_text("{not json", encoding="utf-8")
    assert load_sibling_ledger(graph_path) is None
