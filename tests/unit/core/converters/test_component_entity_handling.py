"""
Regression tests for component vs entity handling in graph conversion.

These tests validate the critical bug fix where components (is_entity=False)
were being incorrectly set to None instead of being embedded as dictionaries.
"""

from typing import Any, List, Optional

import networkx as nx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry


# Test helper function
def edge(label: str, **kwargs: Any) -> Any:
    """Helper to create graph edges."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)


# Test models: Components (is_entity=False)
class Address(BaseModel):
    """Address component - should be embedded as dict."""

    model_config = ConfigDict(is_entity=False)

    street: str = Field(...)
    city: str = Field(...)
    postal_code: str | None = Field(None)


class MonetaryAmount(BaseModel):
    """Monetary amount component - should be embedded as dict."""

    model_config = ConfigDict(is_entity=False)

    value: float = Field(...)
    currency: str = Field(...)


class ContactInfo(BaseModel):
    """Contact info component - should be embedded as dict."""

    model_config = ConfigDict(is_entity=False)

    email: str | None = Field(None)
    phone: str | None = Field(None)


# Test models: Entities (graph_id_fields)
class Organization(BaseModel):
    """Organization entity - should be separate node."""

    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)
    tax_id: str | None = Field(None)

    # Edge to component - should embed address as dict
    address: Address = edge(label="LOCATED_AT")

    # Optional component
    contact: ContactInfo | None = Field(None)


class Person(BaseModel):
    """Person entity - should be separate node."""

    model_config = ConfigDict(graph_id_fields=["first_name", "last_name"])

    first_name: str = Field(...)
    last_name: str = Field(...)

    # Edge to component - should embed address as dict
    address: Address = edge(label="LIVES_AT")


class Invoice(BaseModel):
    """Invoice entity - root document."""

    model_config = ConfigDict(graph_id_fields=["invoice_number"])

    invoice_number: str = Field(...)
    date: str = Field(...)

    # Edge to entity - should create edge to separate node
    issued_by: Organization = edge(label="ISSUED_BY")

    # Edge to entity - should create edge to separate node
    sent_to: Person = edge(label="SENT_TO")

    # Component - should be embedded as dict
    total: MonetaryAmount = Field(...)


# ============================================================================
# Test Cases
# ============================================================================


class TestComponentEmbedding:
    """Test that components are correctly embedded as dictionaries."""

    def test_component_embedded_in_entity_node(self):
        """Test that component data is embedded in parent entity node."""
        # Create test data
        org = Organization(
            name="Acme Corp",
            tax_id="FR123456789",
            address=Address(street="123 Main St", city="Paris", postal_code="75001"),
            contact=ContactInfo(email="contact@acme.com", phone="+33 1 23 45 67 89"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry, validate_graph=False)
        graph, _ = converter.pydantic_list_to_graph([org])

        # Verify graph structure - only Organization node (components are embedded)
        assert graph.number_of_nodes() == 1  # Only Organization node

        # Get organization node
        org_nodes = [n for n in graph.nodes() if n.startswith("Organization_")]
        assert len(org_nodes) == 1
        org_node_id = org_nodes[0]
        org_data = graph.nodes[org_node_id]

        # CRITICAL: Address should be embedded as dict, NOT None
        assert org_data["address"] is not None, "Component address should not be None!"
        assert isinstance(org_data["address"], dict), (
            "Component address should be embedded as dict!"
        )
        assert org_data["address"]["street"] == "123 Main St"
        assert org_data["address"]["city"] == "Paris"
        assert org_data["address"]["postal_code"] == "75001"

        # Contact should also be embedded as dict
        assert org_data["contact"] is not None, "Component contact should not be None!"
        assert isinstance(org_data["contact"], dict), (
            "Component contact should be embedded as dict!"
        )
        assert org_data["contact"]["email"] == "contact@acme.com"
        assert org_data["contact"]["phone"] == "+33 1 23 45 67 89"

    def test_component_not_created_as_separate_node(self):
        """Test that components don't create separate nodes (unless used in edge)."""
        # Create invoice with embedded monetary amount
        invoice = Invoice(
            invoice_number="INV-001",
            date="2024-01-15",
            issued_by=Organization(
                name="Acme Corp", address=Address(street="123 Main St", city="Paris")
            ),
            sent_to=Person(
                first_name="John",
                last_name="Doe",
                address=Address(street="456 Oak Ave", city="London"),
            ),
            total=MonetaryAmount(value=1000.00, currency="EUR"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry)
        graph, _ = converter.pydantic_list_to_graph([invoice])

        # Get invoice node
        invoice_nodes = [n for n in graph.nodes() if n.startswith("Invoice_")]
        assert len(invoice_nodes) == 1
        invoice_data = graph.nodes[invoice_nodes[0]]

        # CRITICAL: Total (MonetaryAmount) should be embedded as dict
        assert invoice_data["total"] is not None, "Component total should not be None!"
        assert isinstance(invoice_data["total"], dict), (
            "Component total should be embedded as dict!"
        )
        assert invoice_data["total"]["value"] == 1000.00
        assert invoice_data["total"]["currency"] == "EUR"

    def test_multiple_entities_share_same_component(self):
        """Test that identical components are embedded in each entity."""
        # Create two people at the same address
        shared_address = Address(street="123 Main St", city="Paris", postal_code="75001")

        person1 = Person(first_name="John", last_name="Doe", address=shared_address)

        person2 = Person(first_name="Jane", last_name="Smith", address=shared_address)

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry, validate_graph=False)
        graph, _ = converter.pydantic_list_to_graph([person1, person2])

        # Should have 2 Person nodes (components are embedded, not separate nodes)
        person_nodes = [n for n in graph.nodes() if n.startswith("Person_")]

        assert len(person_nodes) == 2, "Should have 2 Person nodes"

        # Both persons should have address embedded as dict
        for person_node in person_nodes:
            person_data = graph.nodes[person_node]

            # Address should be embedded as dict
            assert person_data["address"] is not None
            assert isinstance(person_data["address"], dict)
            assert person_data["address"]["street"] == "123 Main St"


class TestEntitySeparation:
    """Test that entities create separate nodes with edges."""

    def test_entity_creates_separate_node(self):
        """Test that entity fields create separate nodes, not embedded dicts."""
        invoice = Invoice(
            invoice_number="INV-001",
            date="2024-01-15",
            issued_by=Organization(
                name="Acme Corp", address=Address(street="123 Main St", city="Paris")
            ),
            sent_to=Person(
                first_name="John",
                last_name="Doe",
                address=Address(street="456 Oak Ave", city="London"),
            ),
            total=MonetaryAmount(value=1000.00, currency="EUR"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry)
        graph, _ = converter.pydantic_list_to_graph([invoice])

        # Should have: Invoice + Organization + Person + 2 Addresses
        assert graph.number_of_nodes() >= 3, (
            "Should have at least Invoice, Organization, and Person nodes"
        )

        # Get invoice node
        invoice_nodes = [n for n in graph.nodes() if n.startswith("Invoice_")]
        assert len(invoice_nodes) == 1
        invoice_node = invoice_nodes[0]
        invoice_data = graph.nodes[invoice_node]

        # CRITICAL: Entity fields should be None (linked via edges)
        assert invoice_data["issued_by"] is None, "Entity field should be None (linked via edge)"
        assert invoice_data["sent_to"] is None, "Entity field should be None (linked via edge)"

        # Should have edges to entity nodes
        edges = list(graph.out_edges(invoice_node, data=True))
        assert len(edges) == 2, "Should have 2 edges (ISSUED_BY, SENT_TO)"

        edge_labels = {e[2].get("label") for e in edges}
        assert "ISSUED_BY" in edge_labels
        assert "SENT_TO" in edge_labels

    def test_entity_deduplication_by_id_fields(self):
        """Test that entities with same ID fields are deduplicated."""
        # Create two invoices from same organization
        org = Organization(name="Acme Corp", address=Address(street="123 Main St", city="Paris"))

        invoice1 = Invoice(
            invoice_number="INV-001",
            date="2024-01-15",
            issued_by=org,
            sent_to=Person(
                first_name="John",
                last_name="Doe",
                address=Address(street="456 Oak Ave", city="London"),
            ),
            total=MonetaryAmount(value=1000.00, currency="EUR"),
        )

        invoice2 = Invoice(
            invoice_number="INV-002",
            date="2024-01-16",
            issued_by=org,  # Same organization
            sent_to=Person(
                first_name="Jane",
                last_name="Smith",
                address=Address(street="789 Elm St", city="Berlin"),
            ),
            total=MonetaryAmount(value=2000.00, currency="EUR"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry)
        graph, _ = converter.pydantic_list_to_graph([invoice1, invoice2])

        # Should have: 2 Invoices + 1 Organization (shared) + 2 Persons + addresses
        invoice_nodes = [n for n in graph.nodes() if n.startswith("Invoice_")]
        org_nodes = [n for n in graph.nodes() if n.startswith("Organization_")]
        person_nodes = [n for n in graph.nodes() if n.startswith("Person_")]

        assert len(invoice_nodes) == 2, "Should have 2 Invoice nodes"
        assert len(org_nodes) == 1, "Should have 1 shared Organization node"
        assert len(person_nodes) == 2, "Should have 2 Person nodes"


class TestEdgeCreation:
    """Test that edges are created correctly for entities but not components."""

    def test_no_edges_created_for_embedded_components(self):
        """Test that embedded components don't create edges."""
        invoice = Invoice(
            invoice_number="INV-001",
            date="2024-01-15",
            issued_by=Organization(
                name="Acme Corp", address=Address(street="123 Main St", city="Paris")
            ),
            sent_to=Person(
                first_name="John",
                last_name="Doe",
                address=Address(street="456 Oak Ave", city="London"),
            ),
            total=MonetaryAmount(value=1000.00, currency="EUR"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry)
        graph, _ = converter.pydantic_list_to_graph([invoice])

        # Get invoice node
        invoice_nodes = [n for n in graph.nodes() if n.startswith("Invoice_")]
        invoice_node = invoice_nodes[0]

        # Get all edges from invoice
        edges = list(graph.out_edges(invoice_node, data=True))
        edge_labels = [e[2].get("label") for e in edges]

        # Should NOT have edge for embedded component (total)
        # The total field is a regular Field(), not edge()
        assert "HAS_TOTAL" not in edge_labels, "Should not create edge for embedded component"

        # Should have edges for entities
        assert "ISSUED_BY" in edge_labels
        assert "SENT_TO" in edge_labels

    def test_edges_created_for_component_with_edge_helper(self):
        """Test that components are embedded even when using edge() helper."""
        org = Organization(name="Acme Corp", address=Address(street="123 Main St", city="Paris"))

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry, validate_graph=False)
        graph, _ = converter.pydantic_list_to_graph([org])

        # Get organization node
        org_nodes = [n for n in graph.nodes() if n.startswith("Organization_")]
        org_node = org_nodes[0]
        org_data = graph.nodes[org_node]

        # Components are embedded as dicts, not separate nodes with edges
        assert org_data["address"] is not None
        assert isinstance(org_data["address"], dict)
        assert org_data["address"]["street"] == "123 Main St"
        assert org_data["address"]["city"] == "Paris"


class TestRegressionScenarios:
    """Test specific regression scenarios from the bug report."""

    def test_invoice_with_all_components(self):
        """
        Regression test for the original bug:
        All component data (addresses, amounts, contacts) was being set to null.
        """
        invoice = Invoice(
            invoice_number="INV-2024-001",
            date="2024-01-15",
            issued_by=Organization(
                name="Acme Corporation Ltd",
                tax_id="FR123456789",
                address=Address(
                    street="123 Avenue des Champs-Élysées", city="Paris", postal_code="75008"
                ),
                contact=ContactInfo(email="contact@acme.com", phone="+33 1 23 45 67 89"),
            ),
            sent_to=Person(
                first_name="Jean",
                last_name="Dupont",
                address=Address(street="456 Rue de la Paix", city="Lyon", postal_code="69001"),
            ),
            total=MonetaryAmount(value=5000.00, currency="EUR"),
        )

        # Convert to graph
        registry = NodeIDRegistry()
        converter = GraphConverter(registry=registry)
        graph, _ = converter.pydantic_list_to_graph([invoice])

        # Get all nodes
        invoice_nodes = [n for n in graph.nodes() if n.startswith("Invoice_")]
        org_nodes = [n for n in graph.nodes() if n.startswith("Organization_")]
        person_nodes = [n for n in graph.nodes() if n.startswith("Person_")]

        assert len(invoice_nodes) == 1
        assert len(org_nodes) == 1
        assert len(person_nodes) == 1

        # CRITICAL REGRESSION TEST: Check invoice node has embedded total
        invoice_data = graph.nodes[invoice_nodes[0]]
        assert invoice_data["total"] is not None, "BUG: Invoice total should not be None!"
        assert isinstance(invoice_data["total"], dict), "BUG: Invoice total should be dict!"
        assert invoice_data["total"]["value"] == 5000.00
        assert invoice_data["total"]["currency"] == "EUR"

        # CRITICAL REGRESSION TEST: Check organization node has embedded address and contact
        org_data = graph.nodes[org_nodes[0]]
        assert org_data["address"] is not None, "BUG: Organization address should not be None!"
        assert isinstance(org_data["address"], dict), "BUG: Organization address should be dict!"
        assert org_data["address"]["street"] == "123 Avenue des Champs-Élysées"
        assert org_data["address"]["city"] == "Paris"

        assert org_data["contact"] is not None, "BUG: Organization contact should not be None!"
        assert isinstance(org_data["contact"], dict), "BUG: Organization contact should be dict!"
        assert org_data["contact"]["email"] == "contact@acme.com"

        # CRITICAL REGRESSION TEST: Check person node has embedded address
        person_data = graph.nodes[person_nodes[0]]
        assert person_data["address"] is not None, "BUG: Person address should not be None!"
        assert isinstance(person_data["address"], dict), "BUG: Person address should be dict!"
        assert person_data["address"]["street"] == "456 Rue de la Paix"
        assert person_data["address"]["city"] == "Lyon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
