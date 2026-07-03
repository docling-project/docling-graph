"""Pipeline-level provenance tests: stage binding, persistence, exporters, config knob."""

import json
from unittest.mock import Mock

import networkx as nx

from docling_graph.config import PipelineConfig
from docling_graph.core.exporters.csv_exporter import CSVExporter
from docling_graph.core.exporters.cypher_exporter import CypherExporter
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.provenance import (
    PROVENANCE_NODE_ATTR,
    ChunkRecord,
    DocumentOrigin,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
)
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.stages import ExportStage, GraphConversionStage
from tests.fixtures.sample_templates.test_template import SampleCompany, SamplePerson


def _ledger() -> ProvenanceLedger:
    root_entry = NodeProvenance(
        identity_key="|company_name=acme",
        catalog_path="",
        node_type="SampleCompany",
        ids={"company_name": "Acme"},
        anchors=[SourceAnchor(chunk_id=0)],
        notes=["scope:document"],
    )
    emp_entry = NodeProvenance(
        identity_key="employees[]|email=janeacmecom",
        catalog_path="employees[]",
        node_type="SamplePerson",
        ids={"email": "jane@acme.com"},
        anchors=[SourceAnchor(chunk_id=0)],
    )
    return ProvenanceLedger(
        document=DocumentOrigin(document_id="doc-1", source="x.pdf"),
        chunks={0: ChunkRecord(chunk_id=0, batch_index=0, page_numbers=(4,))},
        nodes={e.identity_key: e for e in (root_entry, emp_entry)},
    )


def _context(provenance_mode: str) -> PipelineContext:
    config = PipelineConfig(
        source="test.pdf",
        template="tests.fixtures.sample_templates.test_template.SampleCompany",
        provenance=provenance_mode,
    )
    context = PipelineContext(config=config)
    context.template = SampleCompany
    context.extracted_models = [
        SampleCompany(
            company_name="Acme",
            industry="Robotics",
            founded_year=1999,
            employees=[SamplePerson(first_name="Jane", last_name="Doe", email="jane@acme.com")],
        )
    ]
    context.provenance = _ledger()
    return context


class TestGraphConversionStageBinding:
    def test_nodes_annotated_when_provenance_on(self):
        context = GraphConversionStage().execute(_context("standard"))
        graph = context.knowledge_graph
        annotated = [
            data[PROVENANCE_NODE_ATTR]
            for _, data in graph.nodes(data=True)
            if PROVENANCE_NODE_ATTR in data
        ]
        assert len(annotated) == graph.number_of_nodes() == 2
        views_by_kind = {("scope" in v): v for v in annotated}
        assert views_by_kind[True] == {"document_id": "doc-1", "scope": "document"}
        assert views_by_kind[False]["pages"] == [4]

    def test_no_annotation_when_provenance_off(self):
        context = _context("off")
        context = GraphConversionStage().execute(context)
        for _, data in context.knowledge_graph.nodes(data=True):
            assert PROVENANCE_NODE_ATTR not in data


class TestExportStagePersistence:
    def test_provenance_json_written_next_to_graph(self, tmp_path):
        context = GraphConversionStage().execute(_context("standard"))
        output_manager = Mock()
        output_manager.get_docling_graph_dir.return_value = tmp_path
        context.output_manager = output_manager

        ExportStage().execute(context)

        provenance_path = tmp_path / "provenance.json"
        assert provenance_path.is_file()
        restored = ProvenanceLedger.model_validate_json(provenance_path.read_text("utf-8"))
        assert restored.document is not None
        assert restored.document.document_id == "doc-1"
        assert restored.chunks[0].page_numbers == (4,)
        # graph.json carries the node annotations
        graph_data = json.loads((tmp_path / "graph.json").read_text("utf-8"))
        annotated = [n for n in graph_data["nodes"] if PROVENANCE_NODE_ATTR in n]
        assert len(annotated) == 2

    def test_no_provenance_file_when_off(self, tmp_path):
        context = _context("off")
        context = GraphConversionStage().execute(context)
        output_manager = Mock()
        output_manager.get_docling_graph_dir.return_value = tmp_path
        context.output_manager = output_manager

        ExportStage().execute(context)
        assert not (tmp_path / "provenance.json").exists()


def _tiny_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        "Thing_1",
        id="Thing_1",
        label="Thing",
        type="entity",
        name="X",
        **{
            PROVENANCE_NODE_ATTR: {
                "document_id": "d",
                "match": "observed",
                "chunks": [0],
                "pages": [1],
            }
        },
    )
    graph.add_node("Thing_2", id="Thing_2", label="Thing", type="entity", name="Y")
    graph.add_edge("Thing_1", "Thing_2", label="related")
    return graph


class TestExporterSerialization:
    def test_csv_provenance_column_is_parseable_json(self, tmp_path):
        CSVExporter().export(_tiny_graph(), tmp_path)
        import pandas as pd

        nodes_csv = next(tmp_path.glob("*.csv"))
        nodes_df = pd.read_csv(nodes_csv)
        raw = nodes_df.loc[nodes_df["id"] == "Thing_1", PROVENANCE_NODE_ATTR].iloc[0]
        assert json.loads(raw)["chunks"] == [0]

    def test_cypher_provenance_property_embedded_as_json_string(self, tmp_path):
        path = tmp_path / "graph.cypher"
        CypherExporter().export(_tiny_graph(), path)
        script = path.read_text("utf-8")
        assert PROVENANCE_NODE_ATTR in script
        assert '\\"chunks\\": [0]' in script

    def test_json_exporter_keeps_nested_dict(self, tmp_path):
        path = tmp_path / "graph.json"
        JSONExporter().export(_tiny_graph(), path)
        data = json.loads(path.read_text("utf-8"))
        node = next(n for n in data["nodes"] if n["id"] == "Thing_1")
        assert node[PROVENANCE_NODE_ATTR]["pages"] == [1]


class TestConfigKnob:
    def test_default_and_round_trip(self):
        config = PipelineConfig()
        assert config.provenance == "standard"
        as_dict = config.to_dict()
        assert as_dict["provenance"] == "standard"
        rebuilt = PipelineConfig(**as_dict)
        assert rebuilt.provenance == "standard"

    def test_yaml_defaults_include_provenance(self):
        assert PipelineConfig.generate_yaml_dict()["defaults"]["provenance"] == "standard"

    def test_detailed_accepted(self):
        assert PipelineConfig(provenance="detailed").provenance == "detailed"
