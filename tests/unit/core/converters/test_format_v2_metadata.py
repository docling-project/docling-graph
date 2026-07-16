"""Unit tests for format-v2 self-describing export metadata and
wrapped-provenance weighting (converters.graph_converter)."""

import json

import networkx as nx
from pydantic import BaseModel

from docling_graph.core.converters.graph_converter import GraphConverter, _provenance_weight
from docling_graph.core.provenance import (
    PROVENANCE_NODE_ATTR,
    content_hash,
    template_schema_hash,
)


class Company(BaseModel):
    name: str

    model_config = {"graph_id_fields": ["name"]}


class Person(BaseModel):
    name: str
    works_for: Company | None = None

    model_config = {"graph_id_fields": ["name"]}


def _convert() -> nx.DiGraph:
    person = Person(name="Alice", works_for=Company(name="ACME"))
    graph, _ = GraphConverter().pydantic_list_to_graph([person])
    return graph


class TestFormatV2Metadata:
    """graph.graph carries the identity contract of the export (format v2)."""

    def test_format_marker(self):
        assert _convert().graph["format"] == "docling-graph/v2"

    def test_template_name_is_root_class(self):
        assert _convert().graph["template_name"] == "Person"

    def test_id_fields_map_covers_reachable_classes(self):
        assert _convert().graph["id_fields_map"] == {
            "Person": ["name"],
            "Company": ["name"],
        }

    def test_schema_hash_matches_provenance_derivation(self):
        """The embedded hash must equal DocumentOrigin.template_schema_hash's
        derivation (content_hash over the sorted JSON schema), so merge-time
        compatibility gates can compare export and ledger directly."""
        graph = _convert()
        expected = content_hash(
            json.dumps(Person.model_json_schema(), sort_keys=True).encode("utf-8")
        )
        assert graph.graph["template_schema_hash"] == expected
        assert graph.graph["template_schema_hash"] == template_schema_hash(Person)


class TestProvenanceWeight:
    """_provenance_weight over plain and wrapped multi-document views."""

    def test_plain_view_counts_chunks_and_omitted(self):
        view = {"document_id": "a", "chunks": [1, 2], "chunks_omitted": 3}
        assert _provenance_weight({PROVENANCE_NODE_ATTR: view}) == 5

    def test_missing_or_malformed_provenance_scores_zero(self):
        assert _provenance_weight({}) == 0
        assert _provenance_weight({PROVENANCE_NODE_ATTR: "junk"}) == 0

    def test_wrapped_view_sums_all_sources(self):
        wrapped = {
            "multi_document": True,
            "sources": [
                {"document_id": "a", "chunks": [1, 2]},
                {"document_id": "b", "chunks": [0], "chunks_omitted": 3},
            ],
        }
        assert _provenance_weight({PROVENANCE_NODE_ATTR: wrapped}) == 6
