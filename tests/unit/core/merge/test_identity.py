"""Unit tests for merge identity (template walk, fingerprint parity, re-keying)."""

from datetime import datetime
from typing import Any, List, Optional, Union

import networkx as nx
import pytest
from pydantic import BaseModel, Field

import docling_graph.core.merge.identity as identity_module
from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry
from docling_graph.core.exporters.json_exporter import JSONExporter
from docling_graph.core.importers.graph_json import load_graph_json
from docling_graph.core.merge.identity import (
    id_fields_by_template,
    recompute_node_id,
    rekey_graph,
)
from docling_graph.core.merge.policy import MergePolicy
from docling_graph.exceptions import ConfigurationError
from tests.fixtures.sample_templates.test_template import (
    SampleCompany,
    SampleInvoice,
    SamplePerson,
)


class TaggedItem(BaseModel):
    """List-valued identity fixture (mirrors the registry tests)."""

    tags: List[str]
    note: str = ""

    model_config = {"graph_id_fields": ["tags"]}


class Measurement(BaseModel):
    """Id-less fixture: exercises the component fingerprint branch."""

    name: str
    numeric_value: float
    unit: str
    empty_note: str = ""


class TimestampedEvent(BaseModel):
    """Datetime-valued identity fixture (exports carry the ISO 'T' form)."""

    timestamp: datetime
    note: str = ""

    model_config = {"graph_id_fields": ["timestamp"]}


class LogEntry(BaseModel):
    """Id-less datetime fixture: component branch parity for temporal values."""

    message: str
    at: datetime


def _node_attrs(instance: BaseModel) -> dict[str, Any]:
    """Node attributes as GraphConverter writes them (scalar fields verbatim)."""
    attrs: dict[str, Any] = {
        "id": "placeholder",
        "label": type(instance).__name__,
        "type": "entity",
        "__class__": type(instance).__name__,
    }
    for name, value in instance:
        attrs[name] = value
    return attrs


# ---------------------------------------------------------------- parity lock


def test_fingerprint_parity_entity():
    invoice = SampleInvoice(
        invoice_number="INV-001",
        date="2024-01-01",
        total_amount=10.5,
        vendor_name="Acme",
        items=["a", "b"],
    )
    assert recompute_node_id(
        _node_attrs(invoice), ["invoice_number"]
    ) == NodeIDRegistry().get_node_id(invoice)


def test_fingerprint_parity_entity_canonicalization():
    lower = SamplePerson(first_name="A", last_name="B", email="Ada@Acme.COM")
    upper = SamplePerson(first_name="A", last_name="B", email="ada@acme.com")
    node_id = recompute_node_id(_node_attrs(lower), ["email"])
    assert node_id == NodeIDRegistry().get_node_id(upper)


def test_fingerprint_parity_list_valued_id():
    item = TaggedItem(tags=["Beta", "alpha", "beta"], note="x")
    assert recompute_node_id(_node_attrs(item), ["tags"]) == NodeIDRegistry().get_node_id(item)


def test_fingerprint_parity_component_branch():
    measurement = Measurement(name="width", numeric_value=42.5, unit="mm")
    attrs = _node_attrs(measurement)
    assert recompute_node_id(attrs, []) == NodeIDRegistry().get_node_id(measurement)
    assert recompute_node_id(attrs, None) == NodeIDRegistry().get_node_id(measurement)


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 1, 12, 0, 0, 123456),
    ],
)
def test_fingerprint_parity_datetime_id_field(timestamp):
    """The registry canonicalizes str(datetime); exports carry isoformat().
    Both forms must recompute to the same node id."""
    event = TimestampedEvent(timestamp=timestamp)
    attrs = _node_attrs(event)
    attrs["timestamp"] = timestamp.isoformat()  # exported attr form
    assert recompute_node_id(attrs, ["timestamp"]) == NodeIDRegistry().get_node_id(event)


def test_fingerprint_parity_datetime_component_branch():
    entry = LogEntry(message="boot", at=datetime(2024, 1, 1, 12, 0, 0))
    attrs = _node_attrs(entry)
    attrs["at"] = entry.at.isoformat()  # exported attr form
    assert recompute_node_id(attrs, []) == NodeIDRegistry().get_node_id(entry)


def test_skolem_document_id_is_fingerprinted():
    """The skolem stamp is content-bearing identity: without it a re-merge
    would recompute the skolemized root back to its colliding base id."""
    attrs = {
        "id": "Doc_root",
        "label": "Doc",
        "type": "entity",
        "__class__": "Doc",
        "reference": "invoice",
    }
    base_id = recompute_node_id(attrs, ["reference"])
    skolemized = {**attrs, "skolem_document_id": "doc-bbbb"}
    assert recompute_node_id(skolemized, ["reference"]) != base_id
    # Deterministic: same skolem stamp always recomputes to the same id.
    assert recompute_node_id(skolemized, ["reference"]) == recompute_node_id(
        dict(skolemized), ["reference"]
    )


def test_fingerprint_parity_survives_json_round_trip(tmp_path):
    """Loaded (flattened) attrs recompute to the exported node keys exactly."""
    company = SampleCompany(
        company_name="Électroménager S.A.",
        industry="Retail",
        founded_year=1980,
        employees=[SamplePerson(first_name="A", last_name="B", email="a@x.com")],
    )
    graph, _ = GraphConverter().pydantic_list_to_graph([company])
    JSONExporter().export(graph, tmp_path / "graph.json")
    loaded = load_graph_json(tmp_path / "graph.json")
    id_fields_map = id_fields_by_template(SampleCompany)
    for node_id, attrs in loaded.nodes(data=True):
        fields = id_fields_map.get(str(attrs["__class__"]), [])
        assert recompute_node_id(attrs, fields) == node_id


def test_recompute_requires_class_attribute():
    with pytest.raises(ConfigurationError, match="__class__"):
        recompute_node_id({"id": "X", "name": "n"}, ["name"])


# ------------------------------------------------------------- template walk


def test_id_fields_by_template_walks_nested_optional_list_union():
    class Leaf(BaseModel):
        name: str
        model_config = {"graph_id_fields": ["name"]}

    class Mid(BaseModel):
        title: str
        leaf: Leaf | None = None
        alt: Union[Leaf, str, None] = None
        model_config = {"graph_id_fields": ["title"]}

    class Root(BaseModel):
        mids: List[Mid] = Field(default_factory=list)
        maybe: List[Leaf] | None = None

    mapping = id_fields_by_template(Root)
    assert mapping == {"Root": [], "Mid": ["title"], "Leaf": ["name"]}


def test_id_fields_by_template_is_cycle_safe():
    class Node(BaseModel):
        nid: str
        children: List["Node"] = Field(default_factory=list)
        model_config = {"graph_id_fields": ["nid"]}

    Node.model_rebuild()
    assert id_fields_by_template(Node) == {"Node": ["nid"]}


def test_id_fields_by_template_fixture_templates():
    mapping = id_fields_by_template(SampleCompany)
    assert mapping["SampleCompany"] == ["company_name"]
    assert mapping["SamplePerson"] == ["email"]


# ------------------------------------------------------------------ re-keying


def test_rekey_restores_drifted_ids():
    person = SamplePerson(first_name="A", last_name="B", email="a@x.com")
    true_id = NodeIDRegistry().get_node_id(person)
    graph = nx.DiGraph()
    graph.add_node("SamplePerson_drifted", id="SamplePerson_drifted", **_strip_id(person))
    rekeyed, changed, field_conflicts, edge_conflicts = rekey_graph(
        graph, {"SamplePerson": ["email"]}, MergePolicy(), "test-input"
    )
    assert changed == 1
    assert set(rekeyed.nodes) == {true_id}
    assert rekeyed.nodes[true_id]["id"] == true_id
    assert field_conflicts == [] and edge_conflicts == []


def test_rekey_fan_in_folds_instead_of_clobbering():
    """Two surface variants of one identity fold through fold_node_attrs."""
    graph = nx.DiGraph()
    graph.add_node(
        "P1",
        id="P1",
        label="SamplePerson",
        type="entity",
        __class__="SamplePerson",
        email="Ada@Acme.com",
        first_name="Ada",
        last_name="",
    )
    graph.add_node(
        "P2",
        id="P2",
        label="SamplePerson",
        type="entity",
        __class__="SamplePerson",
        email="ada@acme.com",
        first_name="Ada",
        last_name="Byron",
    )
    graph.add_node(
        "C",
        id="C",
        label="SampleCompany",
        type="entity",
        __class__="SampleCompany",
        company_name="Acme",
    )
    graph.add_edge("C", "P1", label="employees")
    graph.add_edge("C", "P2", label="employees")

    rekeyed, changed, _conflicts, edge_conflicts = rekey_graph(
        graph,
        {"SamplePerson": ["email"], "SampleCompany": ["company_name"]},
        MergePolicy(),
        "test-input",
    )
    persons = [n for n, d in rekeyed.nodes(data=True) if d["__class__"] == "SamplePerson"]
    assert len(persons) == 1
    survivor = rekeyed.nodes[persons[0]]
    assert survivor["first_name"] == "Ada"
    assert survivor["last_name"] == "Byron"  # filled from the folded twin
    assert changed == 3
    assert edge_conflicts == []
    company = next(n for n, d in rekeyed.nodes(data=True) if d["__class__"] == "SampleCompany")
    assert rekeyed.has_edge(company, persons[0])


def test_rekey_fan_in_unions_merged_aliases():
    """Alias audit records survive fan-in (merged_aliases is a meta attr the
    fold skips, so the fan-in must carry it explicitly)."""
    graph = nx.DiGraph()
    graph.add_node(
        "P1",
        id="P1",
        label="SamplePerson",
        type="entity",
        __class__="SamplePerson",
        email="Ada@Acme.com",
        merged_aliases=[{"id": "SamplePerson_old1", "email": "ada@old.com"}],
    )
    graph.add_node(
        "P2",
        id="P2",
        label="SamplePerson",
        type="entity",
        __class__="SamplePerson",
        email="ada@acme.com",
        merged_aliases=[{"id": "SamplePerson_old2", "email": "ada@older.com"}],
    )
    rekeyed, _changed, _conflicts, _edge_conflicts = rekey_graph(
        graph, {"SamplePerson": ["email"]}, MergePolicy(), "test-input"
    )
    assert rekeyed.number_of_nodes() == 1
    survivor = rekeyed.nodes[next(iter(rekeyed.nodes))]
    assert {a["id"] for a in survivor["merged_aliases"]} == {
        "SamplePerson_old1",
        "SamplePerson_old2",
    }


def test_rekey_cross_class_collision_raises(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("A1", id="A1", __class__="Alpha", name="x")
    graph.add_node("B1", id="B1", __class__="Beta", name="x")
    monkeypatch.setattr(identity_module, "recompute_node_id", lambda attrs, fields: "Collide_0000")
    with pytest.raises(ConfigurationError, match="collision"):
        rekey_graph(graph, {}, MergePolicy(), "test-input")


def _strip_id(instance: BaseModel) -> dict[str, Any]:
    attrs = {
        "label": type(instance).__name__,
        "type": "entity",
        "__class__": type(instance).__name__,
    }
    for name, value in instance:
        attrs[name] = value
    return attrs
