from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.converters.config import GraphConfig
from docling_graph.core.converters.graph_converter import GraphConverter


# Test Pydantic Models
class SimpleModel(BaseModel):
    name: str
    age: int

    model_config = {"graph_id_fields": ["name"]}


class Company(BaseModel):
    name: str
    location: str

    model_config = {"graph_id_fields": ["name"]}


class Person(BaseModel):
    name: str
    works_for: Company | None = None
    friends: List["Person"] = []

    model_config = {"graph_id_fields": ["name"]}


Person.model_rebuild()


@pytest.fixture
def default_config():
    return GraphConfig()


@pytest.fixture
def converter(default_config):
    return GraphConverter(config=default_config)


def test_converter_init(converter):
    """Test converter initialization."""
    assert converter.registry is not None
    assert converter.config is not None


def test_convert_simple_model(converter):
    """Test converting a single model with nested relationship to create edges."""
    company = Company(name="TechCorp", location="SF")
    person = Person(name="Alice", works_for=company)

    # This creates 2 nodes (Person, Company) and 1 edge (Person -> Company)
    graph, _ = converter.pydantic_list_to_graph([person])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


def test_convert_nested_model_creates_edge(converter):
    """Test that a nested model creates two nodes and one edge."""
    company = Company(name="Acme Inc.", location="NY")
    person = Person(name="Alice", works_for=company)

    graph, _ = converter.pydantic_list_to_graph([person])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    # Check nodes exist
    node_labels = [data["label"] for _, data in graph.nodes(data=True)]
    assert "Person" in node_labels
    assert "Company" in node_labels

    # Check edge exists
    edges = list(graph.edges(data=True))
    assert len(edges) == 1


def test_convert_list_of_models_creates_edges(converter):
    """Test a list of nested models."""
    alice = Person(name="Alice")
    bob = Person(name="Bob")
    charlie = Person(name="Charlie", friends=[alice, bob])

    graph, _ = converter.pydantic_list_to_graph([charlie])

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_model_deduplication(converter):
    """Test that identical nested models are not duplicated."""
    company = Company(name="Acme Inc.", location="NY")
    alice = Person(name="Alice", works_for=company)
    bob = Person(name="Bob", works_for=company)

    graph, _ = converter.pydantic_list_to_graph([alice, bob])

    # Should be 3 nodes: Alice, Bob, and one Acme Inc.
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    company_nodes = [
        node_id for node_id, data in graph.nodes(data=True) if data["label"] == "Company"
    ]
    assert len(company_nodes) == 1


@patch("docling_graph.core.converters.graph_converter.GraphCleaner")
@patch("docling_graph.core.converters.graph_converter.validate_graph_structure")
def test_conversion_with_validation(mock_validate, mock_cleaner_class, converter):
    """Test that validation is called."""
    model = SimpleModel(name="Test", age=25)

    _, _ = converter.pydantic_list_to_graph([model])

    mock_validate.assert_called_once()


def test_conversion_without_cleanup():
    """Test conversion with auto_cleanup disabled and validation disabled."""
    config = GraphConfig()
    # Disable validation to allow graphs without edges
    converter = GraphConverter(config=config, auto_cleanup=False, validate_graph=False)

    company = Company(name="TestCorp", location="LA")
    person = Person(name="Test", works_for=company)

    graph, _ = converter.pydantic_list_to_graph([person])

    # Should have 2 nodes and 1 edge
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


# --- Integrity: exported nodes with empty identity are surfaced loudly ---


def test_empty_identity_node_is_flagged_on_graph(converter, caplog):
    """A node whose declared identity fields are all empty lands in graph metadata + warning."""
    import logging

    class RootDoc(BaseModel):
        reference: str = ""
        holder: Company | None = None
        model_config = {"graph_id_fields": ["reference"]}

    model = RootDoc(reference="", holder=Company(name="TechCorp", location="SF"))
    with caplog.at_level(logging.WARNING):
        graph, _ = converter.pydantic_list_to_graph([model])
    flagged = graph.graph.get("empty_identity_nodes", [])
    assert len(flagged) == 1
    assert any("empty identity" in rec.message for rec in caplog.records)


def test_filled_identity_nodes_are_not_flagged(converter):
    company = Company(name="TechCorp", location="SF")
    person = Person(name="Alice", works_for=company)
    graph, _ = converter.pydantic_list_to_graph([person])
    assert "empty_identity_nodes" not in graph.graph
