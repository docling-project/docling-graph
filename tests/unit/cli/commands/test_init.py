"""
Tests for init command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml

from docling_graph.cli.commands.init import init_command
from docling_graph.cli.constants import CONFIG_FILE_NAME


class TestInitCommand:
    """Test init command functionality."""

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_success(
        self,
        mock_cwd,
        mock_build_config,
        mock_validate_deps,
        mock_print_next,
        mock_print_next_deps,
        tmp_path,
    ):
        """Should successfully initialize config."""
        mock_cwd.return_value = tmp_path
        mock_build_config.return_value = {
            "defaults": {"backend": "llm", "inference": "local"},
            "docling": {"pipeline": "ocr"},
        }
        mock_validate_deps.return_value = True

        init_command()

        # Verify config was saved
        assert (tmp_path / CONFIG_FILE_NAME).exists()
        mock_print_next.assert_called_once()
        mock_print_next_deps.assert_called_once()

    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_config_exists_no_overwrite(self, mock_cwd, tmp_path):
        """Should cancel if config exists and user declines overwrite."""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text("defaults: {}\n")
        mock_cwd.return_value = tmp_path

        with patch("typer.confirm", return_value=False):
            init_command()

        # Verify no additional save happened (original content preserved)
        assert config_file.exists()
        assert config_file.read_text() == "defaults: {}\n"

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_config_exists_overwrite(
        self,
        mock_cwd,
        mock_build_config,
        mock_validate_deps,
        mock_print_next,
        mock_print_next_deps,
        tmp_path,
    ):
        """Should overwrite existing config if user confirms."""
        config_file = tmp_path / CONFIG_FILE_NAME
        config_file.write_text("old: config\n")
        mock_cwd.return_value = tmp_path

        mock_build_config.return_value = {
            "defaults": {"backend": "llm", "inference": "local"},
            "docling": {"pipeline": "ocr"},
        }
        mock_validate_deps.return_value = True

        with patch("typer.confirm", return_value=True):
            init_command()

        # Verify new config was saved
        with open(config_file) as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["defaults"]["backend"] == "llm"

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_fallback_on_eof(
        self,
        mock_cwd,
        mock_build_config,
        mock_validate_deps,
        mock_print_next,
        mock_print_next_deps,
        tmp_path,
    ):
        """Should use fallback config on EOFError."""
        mock_cwd.return_value = tmp_path
        mock_build_config.side_effect = EOFError()
        mock_validate_deps.return_value = False

        # This test verifies the fallback logic is exercised
        # The template path won't exist in tests, so it uses minimal default config
        init_command()

        # Verify config was saved (with fallback)
        config_file = tmp_path / CONFIG_FILE_NAME
        assert config_file.exists()

        with open(config_file) as f:
            config = yaml.safe_load(f)
        # Should have default values from fallback
        assert config["defaults"]["backend"] in ["llm", "vlm"]

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_fallback_on_keyboard_interrupt(
        self,
        mock_cwd,
        mock_build_config,
        mock_validate_deps,
        mock_print_next,
        mock_print_next_deps,
        tmp_path,
    ):
        """Should use fallback config on KeyboardInterrupt."""
        mock_cwd.return_value = tmp_path
        mock_build_config.side_effect = KeyboardInterrupt()
        mock_validate_deps.return_value = True

        # This test verifies the fallback logic is exercised
        init_command()

        # Verify config was saved (with fallback)
        config_file = tmp_path / CONFIG_FILE_NAME
        assert config_file.exists()

        with open(config_file) as f:
            config = yaml.safe_load(f)
        # Should have default values from fallback
        assert "defaults" in config

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_save_error_exits(
        self, mock_cwd, mock_build_config, mock_print_next, mock_print_next_deps, tmp_path
    ):
        """Should exit on save error."""
        mock_cwd.return_value = tmp_path
        mock_build_config.return_value = {"defaults": {"inference": "local"}}

        with patch(
            "docling_graph.cli.commands.init.save_config", side_effect=OSError("Cannot write")
        ):
            with pytest.raises(typer.Exit) as exc_info:
                init_command()
            assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.commands.init.print_next_steps_with_deps")
    @patch("docling_graph.cli.commands.init.print_next_steps")
    @patch("docling_graph.cli.commands.init.validate_and_warn_dependencies")
    @patch("docling_graph.cli.commands.init.build_config_interactive")
    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_creates_valid_yaml(
        self,
        mock_cwd,
        mock_build_config,
        mock_validate_deps,
        mock_print_next,
        mock_print_next_deps,
        tmp_path,
    ):
        """Should create valid YAML config."""
        mock_cwd.return_value = tmp_path
        test_config = {
            "defaults": {
                "processing_mode": "many-to-one",
                "backend": "llm",
                "inference": "local",
                "export_format": "csv",
            },
            "docling": {"pipeline": "ocr"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        mock_build_config.return_value = test_config
        mock_validate_deps.return_value = True

        init_command()

        # Verify YAML is valid
        with open(tmp_path / CONFIG_FILE_NAME) as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_config

    @patch("docling_graph.cli.commands.init.Path.cwd")
    def test_init_command_build_error_exits(self, mock_cwd, tmp_path):
        """Should exit on build config error."""
        mock_cwd.return_value = tmp_path

        with patch(
            "docling_graph.cli.commands.init.build_config_interactive",
            side_effect=RuntimeError("Build failed"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                init_command()
            assert exc_info.value.exit_code == 1
