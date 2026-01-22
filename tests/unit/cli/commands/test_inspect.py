"""
Tests for inspect command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.commands.inspect import inspect_command


class TestInspectCommand:
    """Test inspect command functionality."""

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    def test_inspect_command_csv_format(self, mock_visualizer_class, tmp_path):
        """Should handle CSV format inspection."""
        # Create CSV files
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "nodes.csv").write_text("id,label\n1,node1")
        (csv_dir / "edges.csv").write_text("source,target,label\n1,2,edge")

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        inspect_command(path=csv_dir, input_format="csv", open_browser=False)

        mock_visualizer.display_cytoscape_graph.assert_called_once()
        call_args = mock_visualizer.display_cytoscape_graph.call_args
        assert call_args.kwargs["input_format"] == "csv"

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    def test_inspect_command_json_format(self, mock_visualizer_class, tmp_path):
        """Should handle JSON format inspection."""
        # Create JSON file
        json_file = tmp_path / "graph.json"
        json_file.write_text('{"nodes": [], "edges": []}')

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        inspect_command(path=json_file, input_format="json", open_browser=False)

        mock_visualizer.display_cytoscape_graph.assert_called_once()
        call_args = mock_visualizer.display_cytoscape_graph.call_args
        assert call_args.kwargs["input_format"] == "json"

    def test_inspect_command_invalid_format(self, tmp_path):
        """Should exit for invalid format."""
        csv_dir = tmp_path / "data"
        csv_dir.mkdir()

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=csv_dir, input_format="invalid")
        assert exc_info.value.exit_code == 1

    def test_inspect_command_csv_missing_nodes(self, tmp_path):
        """Should exit if nodes.csv missing for CSV format."""
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "edges.csv").write_text("source,target\n1,2")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=csv_dir, input_format="csv")
        assert exc_info.value.exit_code == 1

    def test_inspect_command_csv_missing_edges(self, tmp_path):
        """Should exit if edges.csv missing for CSV format."""
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "nodes.csv").write_text("id,label\n1,node1")

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=csv_dir, input_format="csv")
        assert exc_info.value.exit_code == 1

    def test_inspect_command_json_wrong_type(self, tmp_path):
        """Should exit if JSON path is directory."""
        json_dir = tmp_path / "not_a_file"
        json_dir.mkdir()

        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=json_dir, input_format="json")
        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    def test_inspect_command_with_output_path(self, mock_visualizer_class, tmp_path):
        """Should save output to specified path."""
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "nodes.csv").write_text("id,label\n1,node1")
        (csv_dir / "edges.csv").write_text("source,target,label\n1,2,edge")

        output_file = tmp_path / "output.html"

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        inspect_command(path=csv_dir, input_format="csv", output=output_file, open_browser=False)

        call_args = mock_visualizer.display_cytoscape_graph.call_args
        assert call_args.kwargs["output_path"] == output_file

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    def test_inspect_command_open_browser_flag(self, mock_visualizer_class, tmp_path):
        """Should respect open_browser flag."""
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "nodes.csv").write_text("id,label\n1,node1")
        (csv_dir / "edges.csv").write_text("source,target,label\n1,2,edge")

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        inspect_command(path=csv_dir, open_browser=True)

        call_args = mock_visualizer.display_cytoscape_graph.call_args
        assert call_args.kwargs["open_browser"] is True

    @patch("docling_graph.cli.commands.inspect.InteractiveVisualizer")
    def test_inspect_command_error_handling(self, mock_visualizer_class, tmp_path, capsys):
        """Should handle visualizer error and exit with code 1."""
        csv_dir = tmp_path / "graph_data"
        csv_dir.mkdir()
        (csv_dir / "nodes.csv").write_text("id,label\n1,node1")
        (csv_dir / "edges.csv").write_text("source,target,label\n1,2,edge")

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.display_cytoscape_graph.side_effect = RuntimeError("Viz error")

        # Command catches error and raises typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            inspect_command(path=csv_dir, open_browser=False)

        assert exc_info.value.exit_code == 1
