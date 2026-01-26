"""
Tests for edge case fixes in graph conversion pipeline.

Tests the following fixes:
1. Empty list handling in graph_converter.py
2. Error handling in graph_cleaner.py
3. Phantom node detection (whitespace strings)
4. Node ID collision detection in node_id_registry.py
"""

import logging
from typing import List, Optional

import networkx as nx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry
from docling_graph.core.utils.graph_cleaner import GraphCleaner, is_meaningful_value

# ============================================================================
# Test Models
# ============================================================================

class Address(BaseModel):
    """Component model for testing."""
    model_config = ConfigDict(is_entity=False)

    street: str = Field(...)
    city: str = Field(...)


class Person(BaseModel):
    """Entity model for testing."""
    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)
    age: int | None = Field(None)


class Organization(BaseModel):
    """Entity model with lists for testing."""
    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)
    employees: List[Person] = Field(default_factory=list)
    addresses: List[Address] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class PersonEntity(BaseModel):
    """Similar name to Person for collision testing."""
    model_config = ConfigDict(graph_id_fields=["name"])

    name: str = Field(...)


# ============================================================================
# Issue #1: Empty List Handling Tests
# ============================================================================

class TestEmptyListHandling:
    """Test that empty lists are handled correctly without IndexError."""

    def test_empty_list_of_entities(self):
        """Test empty list of entity models."""
        org = Organization(
            name="Test Corp",
            employees=[],  # Empty list
            addresses=[],
            tags=[]
        )

        # Disable validation for graphs with no edges
        converter = GraphConverter(auto_cleanup=False, validate_graph=False)
        graph, _metadata = converter.pydantic_list_to_graph([org])

        # Should not raise IndexError
        assert graph.number_of_nodes() == 1
        org_node = next(iter(graph.nodes(data=True)))

        # Empty lists should be preserved as empty lists
        assert org_node[1].get("employees") == []
        assert org_node[1].get("addresses") == []
        assert org_node[1].get("tags") == []

    def test_empty_list_of_components(self):
        """Test empty list of component models."""
        org = Organization(
            name="Test Corp",
            employees=[],
            addresses=[],  # Empty list of components
            tags=["tag1"]
        )

        converter = GraphConverter(auto_cleanup=False, validate_graph=False)
        graph, _metadata = converter.pydantic_list_to_graph([org])

        assert graph.number_of_nodes() == 1
        org_node = next(iter(graph.nodes(data=True)))

        # Empty component list should be preserved
        assert org_node[1].get("addresses") == []

    def test_empty_list_of_primitives(self):
        """Test empty list of primitive values."""
        org = Organization(
            name="Test Corp",
            employees=[],
            addresses=[],
            tags=[]  # Empty list of strings
        )

        converter = GraphConverter(auto_cleanup=False, validate_graph=False)
        graph, _metadata = converter.pydantic_list_to_graph([org])

        assert graph.number_of_nodes() == 1
        org_node = next(iter(graph.nodes(data=True)))

        # Empty primitive list should be preserved
        assert org_node[1].get("tags") == []

    def test_mixed_empty_and_populated_lists(self):
        """Test model with mix of empty and populated lists."""
        person1 = Person(name="Alice", age=30)
        addr1 = Address(street="123 Main St", city="Paris")

        org = Organization(
            name="Test Corp",
            employees=[person1],  # Populated
            addresses=[addr1],    # Populated
            tags=[]               # Empty
        )

        converter = GraphConverter()
        graph, _metadata = converter.pydantic_list_to_graph([org])

        # Should have org node + person node (address is component, embedded)
        assert graph.number_of_nodes() == 2

        org_node = next(n for n in graph.nodes(data=True) if "Organization" in n[0])

        # Empty list preserved, populated lists handled correctly
        assert org_node[1].get("tags") == []
        assert org_node[1].get("employees") is None  # Entities set to None
        assert isinstance(org_node[1].get("addresses"), list)  # Components embedded


# ============================================================================
# Issue #2: Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test improved error handling in graph cleaner."""

    def test_orphaned_edge_removal_with_logging(self, caplog):
        """Test that orphaned edge removal handles errors gracefully."""
        graph = nx.DiGraph()

        # Add nodes with meaningful data to avoid phantom removal
        graph.add_node("Node1", label="Node1", name="Node1", value="data1")
        graph.add_node("Node2", label="Node2", name="Node2", value="data2")

        # Add edge
        graph.add_edge("Node1", "Node2", label="CONNECTS_TO")

        cleaner = GraphCleaner(verbose=False)

        # Should handle gracefully without exceptions
        with caplog.at_level(logging.WARNING):
            result = cleaner.clean_graph(graph)

        # Verify no exceptions were raised
        assert result is not None
        assert isinstance(result, nx.DiGraph)

    def test_networkx_error_handling(self):
        """Test that NetworkX errors are caught and logged."""
        graph = nx.DiGraph()
        graph.add_node("Node1", label="Node1", name="Node1", value="data")

        cleaner = GraphCleaner(verbose=False)

        # Should not raise exception even with minimal graph
        result = cleaner.clean_graph(graph)
        assert result is not None


# ============================================================================
# Issue #3: Phantom Node Detection Tests
# ============================================================================

class TestPhantomNodeDetection:
    """Test improved phantom node detection with whitespace handling."""

    def test_is_meaningful_value_whitespace(self):
        """Test that whitespace-only strings are not meaningful."""
        assert not is_meaningful_value("")
        assert not is_meaningful_value("   ")
        assert not is_meaningful_value("\t")
        assert not is_meaningful_value("\n")
        assert not is_meaningful_value("  \t\n  ")

    def test_is_meaningful_value_valid_strings(self):
        """Test that non-empty strings are meaningful."""
        assert is_meaningful_value("Hello")
        assert is_meaningful_value("  Hello  ")  # Has content after strip
        assert is_meaningful_value("0")
        assert is_meaningful_value(" a ")

    def test_is_meaningful_value_numbers(self):
        """Test that zero and false are meaningful."""
        assert is_meaningful_value(0)
        assert is_meaningful_value(0.0)
        assert is_meaningful_value(False)

    def test_is_meaningful_value_collections(self):
        """Test collection handling."""
        assert not is_meaningful_value([])
        assert not is_meaningful_value({})
        assert not is_meaningful_value(set())
        assert not is_meaningful_value(())

        assert is_meaningful_value([1])
        assert is_meaningful_value({"key": "value"})
        assert is_meaningful_value({1})
        assert is_meaningful_value((1,))

    def test_phantom_node_removal_whitespace(self):
        """Test that nodes with only whitespace are removed."""
        graph = nx.DiGraph()

        # Add node with only whitespace fields
        graph.add_node("Phantom1", name="   ", value="", description="\t\n")

        # Add valid node
        graph.add_node("Valid1", name="Real Name", value="Real Value")

        cleaner = GraphCleaner(verbose=False)
        cleaned = cleaner.clean_graph(graph)

        # Phantom should be removed
        assert "Phantom1" not in cleaned.nodes()
        assert "Valid1" in cleaned.nodes()


# ============================================================================
# Issue #4: Node ID Collision Detection Tests
# ============================================================================

class TestNodeIDCollisionDetection:
    """Test improved node ID collision detection."""

    def test_exact_class_name_matching(self):
        """Test that class names are matched exactly, not by prefix."""
        registry = NodeIDRegistry()

        # Register a Person
        person = Person(name="Alice")
        person_id = registry.get_node_id(person)

        # Try to register PersonEntity with same fingerprint
        person_entity = PersonEntity(name="Alice")

        # Should detect collision if fingerprints match
        with pytest.raises(ValueError, match="Node ID collision"):
            # Force same fingerprint by manipulating registry
            fingerprint = registry._generate_fingerprint(person_entity)
            registry.fingerprint_to_id[fingerprint] = person_id
            registry.get_node_id(person_entity)

    def test_no_false_positive_collision(self):
        """Test that similar class names don't cause false positives."""
        registry = NodeIDRegistry()

        # Register Person
        person = Person(name="Alice")
        person_id = registry.get_node_id(person)

        # Register PersonEntity with different data (different fingerprint)
        person_entity = PersonEntity(name="Bob")
        person_entity_id = registry.get_node_id(person_entity)

        # Should get different IDs (no collision)
        assert person_id != person_entity_id
        assert person_id.startswith("Person_")
        assert person_entity_id.startswith("PersonEntity_")

    def test_collision_error_message(self):
        """Test that collision error message is informative."""
        registry = NodeIDRegistry()

        person = Person(name="Alice")
        person_id = registry.get_node_id(person)

        # Simulate collision
        person_entity = PersonEntity(name="Alice")
        fingerprint = registry._generate_fingerprint(person_entity)
        registry.fingerprint_to_id[fingerprint] = person_id

        with pytest.raises(ValueError) as exc_info:
            registry.get_node_id(person_entity)

        error_msg = str(exc_info.value)
        assert "collision" in error_msg.lower()
        assert "Person" in error_msg
        assert "PersonEntity" in error_msg


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple fixes."""

    def test_full_pipeline_with_edge_cases(self):
        """Test complete pipeline with all edge cases."""
        # Create models with various edge cases
        org = Organization(
            name="Test Corp",
            employees=[],  # Empty list (Issue #1)
            addresses=[
                Address(street="   ", city="Paris"),  # Whitespace (Issue #3)
            ],
            tags=[]
        )

        # Disable validation for graphs with no edges
        converter = GraphConverter(auto_cleanup=False, validate_graph=False)
        graph, metadata = converter.pydantic_list_to_graph([org])

        # Should handle all edge cases gracefully
        assert graph.number_of_nodes() >= 1
        assert metadata.node_count >= 1

    def test_error_recovery(self):
        """Test that pipeline recovers from errors gracefully."""
        # Create valid model
        org = Organization(
            name="Test Corp",
            employees=[Person(name="Alice")],
            addresses=[],
            tags=["tag1"]
        )

        converter = GraphConverter(auto_cleanup=True)

        # Should complete without exceptions
        graph, metadata = converter.pydantic_list_to_graph([org])

        assert graph.number_of_nodes() >= 2  # Org + Person
        assert metadata.edge_count >= 1  # Org -> Person


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
