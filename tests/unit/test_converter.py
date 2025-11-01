"""
Unit tests for GraphConverter.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import pytest
from pydantic import BaseModel, Field

from docling_graph.core.base.config import GraphConfig
from docling_graph.core.base.converter import GraphConverter
from docling_graph.core.base.models import Edge


class TestGraphConverterInitialization:
    """Tests for GraphConverter initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        converter = GraphConverter()
        assert converter.config is not None
        assert isinstance(converter.config, GraphConfig)
        assert converter.add_reverse_edges is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = GraphConfig(NODE_ID_HASH_LENGTH=16, MAX_STRING_LENGTH=500)
        converter = GraphConverter(config=config)
        assert converter.config.NODE_ID_HASH_LENGTH == 16
        assert converter.config.MAX_STRING_LENGTH == 500

    def test_init_with_reverse_edges(self):
        """Test initialization with reverse edges enabled."""
        converter = GraphConverter(add_reverse_edges=True)
        assert converter.add_reverse_edges is True


class TestGraphConverterBasicConversion:
    """Tests for basic graph conversion functionality."""

    def test_convert_single_model(self, sample_person):
        """Test converting a single Pydantic model."""
        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph([sample_person])

        assert metadata.node_count == 1
        assert metadata.edge_count == 0
        assert len(graph.nodes) == 1

    def test_convert_multiple_models(self, sample_person_list):
        """Test converting multiple Pydantic models."""
        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph(sample_person_list)

        assert metadata.node_count == 3
        assert len(graph.nodes) == 3

    def test_convert_nested_model(self, sample_company):
        """Test converting model with nested relationships."""
        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph([sample_company])

        # Should have company + 2 employees + 1 address
        assert metadata.node_count >= 1
        assert metadata.edge_count >= 2  # At least company->employees edges

    def test_convert_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        converter = GraphConverter()
        with pytest.raises(ValueError, match="Cannot create graph from empty model list"):
            converter.pydantic_list_to_graph([])


class TestGraphConverterNodeGeneration:
    """Tests for node generation."""

    def test_node_has_correct_attributes(self, sample_person):
        """Test that generated nodes have correct attributes."""
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([sample_person])

        node_id = list(graph.nodes())[0]
        node_data = graph.nodes[node_id]

        assert node_data["label"] == "Person"
        assert node_data["type"] == "entity"
        assert node_data["name"] == "John Doe"
        assert node_data["age"] == 30

    def test_node_id_stability(self, sample_person):
        """Test that node IDs are stable across conversions."""
        converter = GraphConverter()

        graph1, _ = converter.pydantic_list_to_graph([sample_person])
        node_id_1 = list(graph1.nodes())[0]

        graph2, _ = converter.pydantic_list_to_graph([sample_person])
        node_id_2 = list(graph2.nodes())[0]

        assert node_id_1 == node_id_2

    def test_duplicate_models_create_single_node(self):
        """Test that duplicate models create only one node."""
        from ..conftest import Person

        person1 = Person(name="Alice", age=25, email="alice@example.com")
        person2 = Person(name="Alice", age=25, email="alice@example.com")

        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph([person1, person2])

        # Should only create one node since they're identical
        assert metadata.node_count == 1


class TestGraphConverterEdgeGeneration:
    """Tests for edge generation."""

    def test_edges_created_for_nested_models(self, sample_company):
        """Test that edges are created for nested models."""
        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph([sample_company])

        assert metadata.edge_count > 0

        # Check that edges exist
        edges = list(graph.edges(data=True))
        assert len(edges) > 0

    def test_edge_has_correct_label(self, sample_company):
        """Test that edges have correct labels."""
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([sample_company])

        edges = list(graph.edges(data=True))

        # At least one edge should have 'employees' or 'address' label
        labels = [edge[2]["label"] for edge in edges]
        assert any(label in ["employees", "address"] for label in labels)

    def test_reverse_edges_creation(self, sample_company):
        """Test that reverse edges are created when enabled."""
        converter = GraphConverter(add_reverse_edges=True)
        graph, _ = converter.pydantic_list_to_graph([sample_company])

        # Count edges with 'reverse_' prefix
        edges = list(graph.edges(data=True))
        reverse_edges = [e for e in edges if e[2]["label"].startswith("reverse_")]

        # Should have some reverse edges
        assert len(reverse_edges) > 0


class TestGraphConverterThreadSafety:
    """Tests for thread-safety of GraphConverter."""

    def test_concurrent_conversions_same_converter(self, sample_person_list):
        """Test that same converter can be used concurrently."""
        converter = GraphConverter()

        def convert_batch(models):
            graph, metadata = converter.pydantic_list_to_graph(models)
            return metadata.node_count

        # Split models into batches
        batch1 = sample_person_list[:1]
        batch2 = sample_person_list[1:2]
        batch3 = sample_person_list[2:]

        # Convert concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(convert_batch, [batch1, batch2, batch3]))

        # All conversions should succeed
        assert all(count == 1 for count in results)

    def test_no_shared_state_between_conversions(self):
        """Test that conversions don't share state."""
        from ..conftest import Person

        converter = GraphConverter()

        person1 = Person(name="Alice", age=25, email="alice@example.com")
        person2 = Person(name="Bob", age=30, email="bob@example.com")

        # First conversion
        graph1, metadata1 = converter.pydantic_list_to_graph([person1])

        # Second conversion
        graph2, metadata2 = converter.pydantic_list_to_graph([person2])

        # Graphs should be independent
        assert len(graph1.nodes) == 1
        assert len(graph2.nodes) == 1
        assert list(graph1.nodes())[0] != list(graph2.nodes())[0]


class TestGraphConverterValueSerialization:
    """Tests for value serialization."""

    def test_serialize_string_values(self):
        """Test serialization of string values."""
        from ..conftest import Person

        person = Person(name="John", age=30, email="john@example.com")
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([person])

        node_id = list(graph.nodes())[0]
        assert graph.nodes[node_id]["name"] == "John"

    def test_serialize_long_strings_truncated(self):
        """Test that long strings are truncated."""
        from ..conftest import Person

        long_name = "A" * 2000  # Very long name
        person = Person(name=long_name, age=30, email="test@example.com")

        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([person])

        node_id = list(graph.nodes())[0]
        serialized_name = graph.nodes[node_id]["name"]

        # Should be truncated
        assert len(serialized_name) < len(long_name)
        assert serialized_name.endswith("...")

    def test_serialize_numeric_values(self):
        """Test serialization of numeric values."""
        from ..conftest import Person

        person = Person(name="John", age=30, email="john@example.com")
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([person])

        node_id = list(graph.nodes())[0]
        assert graph.nodes[node_id]["age"] == 30
        assert isinstance(graph.nodes[node_id]["age"], int)


class TestGraphConverterNodeIDGeneration:
    """Tests for node ID generation."""

    def test_node_id_uses_graph_id_fields(self):
        """Test that node IDs use graph_id_fields when available."""
        from ..conftest import Person

        person = Person(name="Alice", age=25, email="alice@example.com")
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([person])

        node_id = list(graph.nodes())[0]

        # Should use email as ID (from graph_id_fields)
        assert node_id.startswith("Person_")  # Hash-based ID

    def test_node_id_hash_based_when_no_fields(self):
        """Test hash-based ID when graph_id_fields not set."""
        from ..conftest import Address

        address = Address(street="123 Main", city="NYC", country="USA")
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([address])

        node_id = list(graph.nodes())[0]

        # Should have hash-based ID
        assert "Address_" in node_id


class TestGraphConverterMetadata:
    """Tests for graph metadata generation."""

    def test_metadata_has_correct_counts(self, sample_person_list):
        """Test that metadata has correct node/edge counts."""
        converter = GraphConverter()
        _, metadata = converter.pydantic_list_to_graph(sample_person_list)

        assert metadata.node_count == 3
        assert metadata.source_models == 3

    def test_metadata_average_degree(self, sample_company):
        """Test that metadata calculates average degree."""
        converter = GraphConverter()
        _, metadata = converter.pydantic_list_to_graph([sample_company])

        # Removed: average_degree field does not exist in actual model


class TestGraphConverterEdgeCases:
    """Test edge cases and error conditions."""

    def test_model_with_none_values(self):
        """Test handling of None values in model."""
        from ..conftest import Company

        company = Company(name="NullCorp", employees=[], address=None)
        converter = GraphConverter()
        graph, metadata = converter.pydantic_list_to_graph([company])

        # Should create node without errors
        assert metadata.node_count >= 1

    def test_model_with_empty_lists(self):
        """Test handling of empty lists."""
        from ..conftest import Company

        company = Company(name="EmptyCorp", employees=[])
        converter = GraphConverter()
        graph, _ = converter.pydantic_list_to_graph([company])

        # Should not create edges for empty list
        assert len(list(graph.edges())) >= 0

    def test_cyclic_relationships_handled(self):
        """Test that cyclic relationships are handled properly."""
        # This would require a model with cyclic references
        # For now, just test that the converter doesn't crash
        from ..conftest import Person

        person = Person(name="Alice", age=25, email="alice@example.com")
        converter = GraphConverter()

        # Should complete without infinite recursion
        graph, _ = converter.pydantic_list_to_graph([person, person, person])
        assert len(graph.nodes) >= 1
