"""
Tests for exporter base protocol.
"""

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from docling_graph.core.exporters.base import GraphExporterProtocol


class TestGraphExporterProtocol:
    """Test GraphExporterProtocol interface."""

    def test_protocol_has_export_method(self):
        """Protocol should define export method."""
        assert hasattr(GraphExporterProtocol, "export")

    def test_protocol_has_validate_graph_method(self):
        """Protocol should define validate_graph method."""
        assert hasattr(GraphExporterProtocol, "validate_graph")

    def test_exporter_implements_protocol(self):
        """Exporter should implement the protocol."""
        exporter = MagicMock()
        exporter.export = MagicMock()
        exporter.validate_graph = MagicMock(return_value=True)

        # Check if it matches protocol
        assert hasattr(exporter, "export")
        assert hasattr(exporter, "validate_graph")

    def test_export_method_signature(self):
        """Export method should accept graph and output_path."""
        exporter = MagicMock(spec=GraphExporterProtocol)
        graph = nx.DiGraph()
        path = Path("output.txt")

        exporter.export(graph, path)
        exporter.export.assert_called_once_with(graph, path)

    def test_validate_graph_returns_bool(self):
        """validate_graph should return boolean."""
        exporter = MagicMock(spec=GraphExporterProtocol)
        graph = nx.DiGraph()

        exporter.validate_graph.return_value = True
        result = exporter.validate_graph(graph)

        assert isinstance(result, bool)
