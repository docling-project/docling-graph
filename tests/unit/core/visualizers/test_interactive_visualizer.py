"""
Tests for InteractiveVisualizer class.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest

from docling_graph.core.visualizers.interactive_visualizer import InteractiveVisualizer


@pytest.fixture
def visualizer():
    """Create InteractiveVisualizer instance."""
    return InteractiveVisualizer()


@pytest.fixture
def sample_nodes_csv(tmp_path):
    """Create sample nodes CSV file."""
    csv_path = tmp_path / "nodes.csv"
    csv_path.write_text("id,label,type\nnode_1,Invoice,entity\nnode_2,Amount,entity\n")
    return csv_path


@pytest.fixture
def sample_edges_csv(tmp_path):
    """Create sample edges CSV file."""
    csv_path = tmp_path / "edges.csv"
    csv_path.write_text("source,target,label\nnode_1,node_2,contains\n")
    return csv_path


@pytest.fixture
def sample_json_file(tmp_path):
    """Create sample JSON graph file."""
    import json

    json_path = tmp_path / "graph.json"
    data = {
        "nodes": [{"id": "node_1", "label": "Invoice"}, {"id": "node_2", "label": "Amount"}],
        "edges": [{"source": "node_1", "target": "node_2", "label": "contains"}],
    }
    json_path.write_text(json.dumps(data))
    return json_path


class TestLoadCSV:
    """Test CSV loading."""

    def test_load_csv_success(self, tmp_path):
        """Should load CSV files successfully."""
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")

        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target,label\nnode_1,node_2,rel\n")

        visualizer = InteractiveVisualizer()
        nodes_df, edges_df = visualizer.load_csv(tmp_path)

        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(edges_df, pd.DataFrame)
        assert len(nodes_df) > 0
        assert len(edges_df) > 0

    def test_load_csv_missing_nodes_raises_error(self, tmp_path):
        """Should raise error if nodes.csv missing."""
        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target\nnode_1,node_2\n")

        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(tmp_path)

    def test_load_csv_missing_edges_raises_error(self, tmp_path):
        """Should raise error if edges.csv missing."""
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")

        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_csv(tmp_path)


class TestLoadJSON:
    """Test JSON loading."""

    def test_load_json_success(self, sample_json_file):
        """Should load JSON file successfully."""
        visualizer = InteractiveVisualizer()
        nodes_df, edges_df = visualizer.load_json(sample_json_file)

        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(edges_df, pd.DataFrame)
        assert len(nodes_df) == 2
        assert len(edges_df) == 1

    def test_load_json_file_not_found(self):
        """Should raise error for missing file."""
        visualizer = InteractiveVisualizer()

        with pytest.raises(FileNotFoundError):
            visualizer.load_json(Path("nonexistent.json"))


class TestPrepareDataForCytoscape:
    """Test data preparation for Cytoscape."""

    def test_prepare_data_basic(self, visualizer):
        """Should prepare basic data for Cytoscape."""
        nodes_df = pd.DataFrame({"id": ["node_1", "node_2"], "label": ["Person", "Company"]})
        edges_df = pd.DataFrame(
            {"source": ["node_1"], "target": ["node_2"], "label": ["works_for"]}
        )

        result = visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

        assert "nodes" in result
        assert "edges" in result
        assert "meta" in result
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_prepare_data_validates_edges(self, visualizer):
        """Should require source and target in edges."""
        nodes_df = pd.DataFrame({"id": ["n1", "n2"]})
        edges_df = pd.DataFrame(
            {
                "source": ["n1"],
                "label": ["rel"],  # Missing target
            }
        )

        with pytest.raises(ValueError):
            visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

    def test_prepare_data_handles_missing_values(self, visualizer):
        """Should handle NaN and None values."""
        nodes_df = pd.DataFrame({"id": ["node_1"], "label": [None], "value": [float("nan")]})
        edges_df = pd.DataFrame({"source": ["node_1"], "target": ["node_2"], "label": ["rel"]})

        result = visualizer.prepare_data_for_cytoscape(nodes_df, edges_df)

        assert result is not None
        assert len(result["nodes"]) > 0


class TestSerializeValue:
    """Test value serialization."""

    def test_serialize_none(self, visualizer):
        """Should serialize None."""
        result = visualizer._serialize_value(None)
        assert result is None

    def test_serialize_list(self, visualizer):
        """Should serialize lists."""
        result = visualizer._serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_serialize_dict(self, visualizer):
        """Should serialize dicts."""
        d = {"key": "value"}
        result = visualizer._serialize_value(d)
        assert result == d

    def test_serialize_string(self, visualizer):
        """Should serialize strings."""
        result = visualizer._serialize_value("test")
        assert result == "test"


class TestIsValidValue:
    """Test value validation."""

    def test_is_valid_value_none(self, visualizer):
        """Should return False for None."""
        assert visualizer._is_valid_value(None) is False

    def test_is_valid_value_empty_list(self, visualizer):
        """Should return False for empty list."""
        assert visualizer._is_valid_value([]) is False

    def test_is_valid_value_filled_list(self, visualizer):
        """Should return True for non-empty list."""
        assert visualizer._is_valid_value([1, 2, 3]) is True

    def test_is_valid_value_empty_string(self, visualizer):
        """Should return False for empty string."""
        assert visualizer._is_valid_value("") is False

    def test_is_valid_value_filled_string(self, visualizer):
        """Should return True for non-empty string."""
        assert visualizer._is_valid_value("test") is True


class TestDisplayCytoscapeGraph:
    """Test Cytoscape graph display."""

    def test_display_csv_format(self, tmp_path):
        """Should display CSV format graph."""
        # Create test files
        nodes_file = tmp_path / "nodes.csv"
        nodes_file.write_text("id,label\nnode_1,Test\n")
        edges_file = tmp_path / "edges.csv"
        edges_file.write_text("source,target,label\nnode_1,node_2,rel\n")

        visualizer = InteractiveVisualizer()

        with patch("webbrowser.open"):
            output = visualizer.display_cytoscape_graph(
                tmp_path, input_format="csv", open_browser=False
            )

        assert output.exists()
        assert output.suffix == ".html"

    def test_display_json_format(self, sample_json_file, tmp_path):
        """Should display JSON format graph."""
        visualizer = InteractiveVisualizer()
        output_file = tmp_path / "output.html"

        with patch("webbrowser.open"):
            output = visualizer.display_cytoscape_graph(
                sample_json_file, input_format="json", output_path=output_file, open_browser=False
            )

        assert output.exists()

    def test_display_invalid_format_raises_error(self, tmp_path):
        """Should raise error for invalid format."""
        visualizer = InteractiveVisualizer()

        with pytest.raises(ValueError):
            visualizer.display_cytoscape_graph(tmp_path, input_format="invalid")


class TestSaveCytoscapeGraph:
    """Test NetworkX graph visualization."""

    def test_save_cytoscape_graph_from_networkx(self, tmp_path):
        """Test converting NetworkX graph to Cytoscape visualization."""
        # Create a NetworkX graph with nodes and edges with attributes
        graph = nx.DiGraph()
        graph.add_node("node_1", label="Person", type="entity")
        graph.add_node("node_2", label="Company", type="organization")
        graph.add_edge("node_1", "node_2", label="works_for", weight=1.0)

        visualizer = InteractiveVisualizer()
        output_file = tmp_path / "networkx_graph.html"

        with patch("webbrowser.open"):
            output = visualizer.save_cytoscape_graph(
                graph, output_path=output_file, open_browser=False
            )

        # Verify the output file was created
        assert output.exists()
        assert output.suffix == ".html"

        # Verify the HTML contains graph data
        content = output.read_text()
        assert "node_1" in content
        assert "node_2" in content
        assert "works_for" in content

    def test_save_cytoscape_graph_with_browser_open(self, tmp_path):
        """Test NetworkX graph visualization with browser opening."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="Test")
        graph.add_edge("n1", "n2", label="rel")

        visualizer = InteractiveVisualizer()
        output_file = tmp_path / "graph.html"

        with patch("webbrowser.open") as mock_browser:
            output = visualizer.save_cytoscape_graph(
                graph, output_path=output_file, open_browser=True
            )

        # Verify browser was called
        assert mock_browser.called
        assert output.exists()


class TestExportAndOpen:
    """Test browser opening logic."""

    def test_export_and_open_with_browser_true(self, tmp_path, visualizer):
        """Test that webbrowser.open() is called when open_browser=True."""
        elements = {
            "nodes": [{"data": {"id": "n1", "label": "Test"}}],
            "edges": [],
            "meta": {"node_types": {}, "node_count": 1, "edge_count": 0},
        }
        output_file = tmp_path / "test_graph.html"

        with patch("webbrowser.open") as mock_browser:
            result = visualizer._export_and_open(elements, output_file, open_browser=True)

        # Verify webbrowser.open was called with correct file path
        assert mock_browser.called
        call_args = mock_browser.call_args[0][0]
        assert "file://" in call_args
        assert str(output_file) in call_args
        assert result == output_file

    def test_export_and_open_with_browser_false(self, tmp_path, visualizer):
        """Test that webbrowser.open() is not called when open_browser=False."""
        elements = {
            "nodes": [{"data": {"id": "n1", "label": "Test"}}],
            "edges": [],
            "meta": {"node_types": {}, "node_count": 1, "edge_count": 0},
        }
        output_file = tmp_path / "test_graph.html"

        with patch("webbrowser.open") as mock_browser:
            result = visualizer._export_and_open(elements, output_file, open_browser=False)

        # Verify webbrowser.open was not called
        assert not mock_browser.called
        assert result == output_file
        assert output_file.exists()


class TestWriteCytoscapeHtml:
    """Test HTML writing with fallback scenarios."""

    def test_write_html_with_missing_libraries(self, tmp_path, visualizer):
        """Test fallback to CDN when library files are missing."""
        elements = {
            "nodes": [{"data": {"id": "n1", "label": "Test"}}],
            "edges": [],
            "meta": {"node_types": {}, "node_count": 1, "edge_count": 0},
        }
        output_file = tmp_path / "test.html"

        # Mock the library methods to raise FileNotFoundError
        with patch.object(
            visualizer, "_get_cytoscape_library", side_effect=FileNotFoundError("Library not found")
        ):
            visualizer._write_cytoscape_html(elements, output_file)

        # Verify file was created with CDN links
        assert output_file.exists()
        content = output_file.read_text()
        assert "unpkg.com/cytoscape" in content
        assert "unpkg.com/dagre" in content
        assert "unpkg.com/cytoscape-dagre" in content

    def test_write_html_with_missing_template(self, tmp_path, visualizer):
        """Test fallback to default template when external template is missing."""
        elements = {
            "nodes": [{"data": {"id": "n1", "label": "Test"}}],
            "edges": [],
            "meta": {"node_types": {}, "node_count": 1, "edge_count": 0},
        }
        output_file = tmp_path / "test.html"

        # Mock the template path to not exist
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(visualizer, "_get_default_template") as mock_template:
                mock_template.return_value = """<!DOCTYPE html>
                <html><body><div id="cy"></div>
                <script>/* ELEMENTS_DATA_PLACEHOLDER */</script>
                </body></html>"""
                visualizer._write_cytoscape_html(elements, output_file)

        # Verify default template was called
        assert mock_template.called
        assert output_file.exists()

    def test_write_html_with_valid_template(self, tmp_path, visualizer):
        """Test HTML writing with valid template and libraries."""
        elements = {
            "nodes": [{"data": {"id": "n1", "label": "Test"}}],
            "edges": [],
            "meta": {"node_types": {}, "node_count": 1, "edge_count": 0},
        }
        output_file = tmp_path / "test.html"

        # Create a mock template

        # Create a real template file
        Path(
            __file__
        ).parent.parent.parent.parent / "docling_graph/core/visualizers/assets/interactive_template.html"

        with patch.object(visualizer, "_get_cytoscape_library", return_value="// cytoscape lib"):
            with patch.object(visualizer, "_get_dagre_library", return_value="// dagre lib"):
                with patch.object(
                    visualizer, "_get_cytoscape_dagre_library", return_value="// dagre ext"
                ):
                    visualizer._write_cytoscape_html(elements, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "n1" in content
        assert "Test" in content


class TestGetDefaultTemplate:
    """Test default template generation."""

    def test_get_default_template_with_missing_cytoscape_lib(self, visualizer):
        """Test default template generation with CDN fallback."""
        # Mock Cytoscape library file to be missing
        with patch.object(
            visualizer, "_get_cytoscape_library", side_effect=FileNotFoundError("Library not found")
        ):
            template = visualizer._get_default_template()

        # Verify CDN link is used
        assert "unpkg.com/cytoscape" in template
        assert "<!DOCTYPE html>" in template
        assert "ELEMENTS_DATA_PLACEHOLDER" in template
        assert '<div id="cy"></div>' in template

    def test_get_default_template_with_inline_lib(self, visualizer):
        """Test default template with inline library embedding."""
        # Mock Cytoscape library file to exist
        mock_lib_content = "// Mock Cytoscape.js library content"
        with patch.object(visualizer, "_get_cytoscape_library", return_value=mock_lib_content):
            template = visualizer._get_default_template()

        # Verify library is embedded inline
        assert mock_lib_content in template
        assert "unpkg.com" not in template
        assert "<!DOCTYPE html>" in template
        assert "ELEMENTS_DATA_PLACEHOLDER" in template

    def test_get_default_template_structure(self, visualizer):
        """Test that default template has required structure."""
        with patch.object(
            visualizer, "_get_cytoscape_library", side_effect=FileNotFoundError("Not found")
        ):
            template = visualizer._get_default_template()

        # Verify template structure
        assert "<!DOCTYPE html>" in template
        assert "<html" in template
        assert "<head>" in template
        assert "<body>" in template
        assert '<div id="cy"></div>' in template
        assert "ELEMENTS_DATA_PLACEHOLDER" in template
        assert "cytoscape(" in template


class TestGetLibraryContent:
    """Test library content retrieval."""

    def test_get_library_content_file_not_found(self, visualizer):
        """Test that FileNotFoundError is raised for missing library."""
        with pytest.raises(FileNotFoundError) as exc_info:
            visualizer._get_library_content("nonexistent.js")

        assert "Library file not found" in str(exc_info.value)

    def test_get_cytoscape_library_calls_get_library_content(self, visualizer):
        """Test that _get_cytoscape_library calls _get_library_content."""
        with patch.object(visualizer, "_get_library_content", return_value="lib content") as mock:
            result = visualizer._get_cytoscape_library()

        mock.assert_called_once_with("cytoscape.min.js")
        assert result == "lib content"

    def test_get_dagre_library_calls_get_library_content(self, visualizer):
        """Test that _get_dagre_library calls _get_library_content."""
        with patch.object(visualizer, "_get_library_content", return_value="dagre content") as mock:
            result = visualizer._get_dagre_library()

        mock.assert_called_once_with("dagre.min.js")
        assert result == "dagre content"

    def test_get_cytoscape_dagre_library_calls_get_library_content(self, visualizer):
        """Test that _get_cytoscape_dagre_library calls _get_library_content."""
        with patch.object(visualizer, "_get_library_content", return_value="dagre ext") as mock:
            result = visualizer._get_cytoscape_dagre_library()

        mock.assert_called_once_with("cytoscape-dagre.js")
        assert result == "dagre ext"
