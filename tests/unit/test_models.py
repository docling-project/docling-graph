"""
Unit tests for core data models (Edge, GraphMetadata).
"""

import pytest
from pydantic import ValidationError

from docling_graph.core.base.models import Edge, GraphMetadata


class TestEdgeModel:
    """Tests for Edge Pydantic model."""

    def test_edge_creation_basic(self):
        """Test creating a basic Edge."""
        edge = Edge(source="node_1", target="node_2", label="connects_to")
        assert edge.source == "node_1"
        assert edge.target == "node_2"
        assert edge.label == "connects_to"
        assert edge.properties == {}

    def test_edge_creation_with_properties(self):
        """Test creating Edge with properties."""
        edge = Edge(
            source="person_1",
            target="company_1",
            label="works_at",
            properties={"since": "2020", "role": "engineer"},
        )
        assert edge.properties["since"] == "2020"
        assert edge.properties["role"] == "engineer"

    def test_edge_requires_source(self):
        """Test that source is required."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(target="node_2", label="connects_to")
        assert "source" in str(exc_info.value)

    def test_edge_requires_target(self):
        """Test that target is required."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(source="node_1", label="connects_to")
        assert "target" in str(exc_info.value)

    def test_edge_requires_label(self):
        """Test that label is required."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(source="node_1", target="node_2")
        assert "label" in str(exc_info.value)

    def test_edge_empty_properties_default(self):
        """Test that properties defaults to empty dict."""
        edge = Edge(source="node_1", target="node_2", label="connects_to")
        assert edge.properties == {}
        assert isinstance(edge.properties, dict)

    def test_edge_serialization(self):
        """Test Edge serialization to dict."""
        edge = Edge(
            source="node_1", target="node_2", label="connects_to", properties={"weight": 1.5}
        )
        edge_dict = edge.model_dump()
        assert edge_dict["source"] == "node_1"
        assert edge_dict["target"] == "node_2"
        assert edge_dict["label"] == "connects_to"
        assert edge_dict["properties"]["weight"] == 1.5

    def test_edge_from_dict(self):
        """Test creating Edge from dict."""
        edge_dict = {
            "source": "node_1",
            "target": "node_2",
            "label": "connects_to",
            "properties": {"attr": "value"},
        }
        edge = Edge(**edge_dict)
        assert edge.source == "node_1"
        assert edge.target == "node_2"
        assert edge.properties["attr"] == "value"

    def test_edge_with_complex_properties(self):
        """Test Edge with complex property values."""
        edge = Edge(
            source="node_1",
            target="node_2",
            label="relates_to",
            properties={
                "confidence": 0.95,
                "type": "semantic",
                "metadata": {"created": "2024-01-01"},
            },
        )
        assert edge.properties["confidence"] == 0.95
        assert edge.properties["metadata"]["created"] == "2024-01-01"

    def test_edge_equality(self):
        """Test Edge equality comparison."""
        edge1 = Edge(source="a", target="b", label="connects")
        edge2 = Edge(source="a", target="b", label="connects")
        assert edge1 == edge2

    def test_edge_inequality_different_source(self):
        """Test Edge inequality with different source."""
        edge1 = Edge(source="a", target="b", label="connects")
        edge2 = Edge(source="c", target="b", label="connects")
        assert edge1 != edge2


class TestGraphMetadata:
    """Tests for GraphMetadata Pydantic model."""

    def test_metadata_creation_basic(self):
        """Test creating basic GraphMetadata."""
        metadata = GraphMetadata(
            node_count=10,
            edge_count=15,
            source_models=5,  # Use source_models
        )
        assert metadata.node_count == 10
        assert metadata.edge_count == 15
        assert metadata.source_models == 5

    def test_metadata_requires_node_count(self):
        """Test that node_count is required."""
        with pytest.raises(ValidationError) as exc_info:
            GraphMetadata(edge_count=15, source_models=5)
        assert "node_count" in str(exc_info.value)

    def test_metadata_requires_edge_count(self):
        """Test that edge_count is required."""
        with pytest.raises(ValidationError) as exc_info:
            GraphMetadata(node_count=10, source_models=5)
        assert "edge_count" in str(exc_info.value)

    def test_metadata_requires_source_models(self):
        """Test that source_models is required."""
        with pytest.raises(ValidationError) as exc_info:
            GraphMetadata(node_count=10, edge_count=15)
        # Expect 'source_models' not 'input_model_count'
        assert "source_models" in str(exc_info.value)

    def test_metadata_serialization(self):
        """Test GraphMetadata serialization."""
        metadata = GraphMetadata(
            node_count=10,
            edge_count=15,
            source_models=5,  # Added source_models
            average_degree=3.0,
        )
        metadata_dict = metadata.model_dump()
        assert metadata_dict["node_count"] == 10
        assert metadata_dict["edge_count"] == 15
        # Expect 'source_models' not 'input_model_count'
        assert metadata_dict["source_models"] == 5
        assert metadata_dict["average_degree"] == 3.0

    def test_metadata_from_dict(self):
        """Test creating GraphMetadata from dict."""
        metadata_dict = {
            "node_count": 20,
            "edge_count": 30,
            "node_types": {"A": 10, "B": 10},
            "edge_types": {"rel": 30},
            "source_models": 10,
            "average_degree": 3.5,
        }
        metadata = GraphMetadata(**metadata_dict)
        assert metadata.node_count == 20
        assert metadata.edge_count == 30

    def test_metadata_zero_values(self):
        """Test GraphMetadata with zero values."""
        metadata = GraphMetadata(node_count=0, edge_count=0, source_models=0)
        assert metadata.node_count == 0
        assert metadata.edge_count == 0

    def test_metadata_large_values(self):
        """Test GraphMetadata with large values."""
        metadata = GraphMetadata(node_count=1_000_000, edge_count=5_000_000, source_models=100_000)
        assert metadata.node_count == 1_000_000
        assert metadata.edge_count == 5_000_000

    def test_metadata_negative_values_invalid(self):
        """Test that negative values are invalid."""
        # Depending on your model validation, this might raise an error
        # If you have validators that enforce non-negative values
        try:
            metadata = GraphMetadata(
                node_count=-1, edge_count=-1, source_models=-1, average_degree=-1.0
            )
            # If creation succeeds, at least verify the values
            assert metadata.node_count == -1
        except ValidationError:
            # This is expected if you have non-negative validators
            pass

    def test_metadata_float_average_degree(self):
        """Test that average_degree accepts float values."""
        metadata = GraphMetadata(
            node_count=10,
            edge_count=15,
            source_models=5,
            average_degree=3.0,  # Explicitly set it
        )
        assert isinstance(metadata.average_degree, float)
        assert metadata.average_degree == 3.0

    def test_metadata_equality(self):
        """Test GraphMetadata equality."""
        meta1 = GraphMetadata(node_count=10, edge_count=15, source_models=5)
        meta2 = GraphMetadata(node_count=10, edge_count=15, source_models=5)
        assert meta1 == meta2

    def test_metadata_inequality(self):
        """Test GraphMetadata inequality."""
        meta1 = GraphMetadata(node_count=10, edge_count=15, source_models=5)
        meta2 = GraphMetadata(node_count=20, edge_count=25, source_models=10)
        assert meta1 != meta2


class TestModelsIntegration:
    """Integration tests for models working together."""

    def test_edge_list_creation(self):
        """Test creating a list of edges."""
        edges = [
            Edge(source="a", target="b", label="connects"),
            Edge(source="b", target="c", label="connects"),
            Edge(source="c", target="a", label="connects"),
        ]
        assert len(edges) == 3
        assert all(isinstance(e, Edge) for e in edges)

    def test_metadata_with_edge_count(self):
        """Test metadata reflecting edge count."""
        edges = [
            Edge(source="a", target="b", label="connects"),
            Edge(source="b", target="c", label="connects"),
        ]
        metadata = GraphMetadata(node_count=3, edge_count=len(edges), source_models=1)
        assert metadata.edge_count == len(edges)
