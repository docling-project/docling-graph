"""
Tests for configuration loading and utility functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import typer
import yaml

from docling_graph.cli.config_utils import (
    get_config_value,
    load_config,
    save_config,
)
from docling_graph.cli.constants import CONFIG_FILE_NAME


class TestLoadConfig:
    """Test configuration loading."""

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_success(self, mock_cwd, tmp_path):
        """Should successfully load valid YAML config."""
        config_path = tmp_path / CONFIG_FILE_NAME
        config_path.write_text("defaults:\n  backend: llm\n")
        mock_cwd.return_value = tmp_path

        config = load_config()

        assert isinstance(config, dict)
        assert config["defaults"]["backend"] == "llm"

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_file_not_found(self, mock_cwd, tmp_path):
        """Should exit with error if config file not found."""
        mock_cwd.return_value = tmp_path
        with pytest.raises(typer.Exit) as exc_info:
            load_config()
        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_empty_file_returns_empty_dict(self, mock_cwd, tmp_path):
        """Should return empty dict for empty YAML file."""
        config_path = tmp_path / CONFIG_FILE_NAME
        config_path.write_text("")
        mock_cwd.return_value = tmp_path

        config = load_config()

        assert config == {}

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_invalid_format_raises_exit(self, mock_cwd, tmp_path):
        """Should exit with error if YAML is not a mapping."""
        config_path = tmp_path / CONFIG_FILE_NAME
        config_path.write_text("- item1\n- item2\n")
        mock_cwd.return_value = tmp_path

        with pytest.raises(typer.Exit) as exc_info:
            load_config()
        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_yaml_error_raises_exit(self, mock_cwd, tmp_path):
        """Should exit with error on YAML parsing error."""
        config_path = tmp_path / CONFIG_FILE_NAME
        config_path.write_text("invalid: [yaml:")
        mock_cwd.return_value = tmp_path

        with pytest.raises(typer.Exit) as exc_info:
            load_config()
        assert exc_info.value.exit_code == 1

    @patch("docling_graph.cli.config_utils.Path.cwd")
    def test_load_config_complex_structure(self, mock_cwd, tmp_path):
        """Should load complex nested config."""
        config_path = tmp_path / CONFIG_FILE_NAME
        config_data = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        mock_cwd.return_value = tmp_path

        config = load_config()

        assert config["defaults"]["backend"] == "llm"
        assert config["models"]["llm"]["local"]["provider"] == "ollama"


class TestSaveConfig:
    """Test configuration saving."""

    def test_save_config_creates_valid_yaml(self, tmp_path):
        """Should save config as valid YAML."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"vlm": {"local": {"default_model": "test-model"}}},
        }
        output_path = tmp_path / "config.yaml"
        save_config(config, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_save_config_overwrites_existing(self, tmp_path):
        """Should overwrite existing config file."""
        output_path = tmp_path / "config.yaml"
        output_path.write_text("old: config\n")

        new_config = {"new": "config"}
        save_config(new_config, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == new_config

    def test_save_config_preserves_structure(self, tmp_path):
        """Should preserve nested structure in saved config."""
        config = {
            "defaults": {"a": "b", "c": "d"},
            "models": {"nested": {"deep": {"value": 123}}},
        }
        output_path = tmp_path / "config.yaml"
        save_config(config, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["models"]["nested"]["deep"]["value"] == 123

    def test_save_config_creates_directory_if_needed(self, tmp_path):
        """Should handle nested directory creation."""
        nested_path = tmp_path / "nested" / "dir" / "config.yaml"
        nested_path.parent.mkdir(parents=True, exist_ok=True)

        config = {"test": "config"}
        save_config(config, nested_path)

        assert nested_path.exists()


class TestGetConfigValue:
    """Test nested configuration value retrieval."""

    def test_get_config_value_single_key(self):
        """Should retrieve single-level config value."""
        config = {"key": "value"}
        assert get_config_value(config, "key") == "value"

    def test_get_config_value_nested_keys(self):
        """Should retrieve deeply nested config value."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        result = get_config_value(config, "models", "llm", "local", "provider")
        assert result == "ollama"

    def test_get_config_value_missing_key_returns_default(self):
        """Should return default value for missing keys."""
        config = {"key1": "value1"}
        result = get_config_value(config, "missing_key", default="default_value")
        assert result == "default_value"

    def test_get_config_value_missing_nested_key_returns_default(self):
        """Should return default for missing nested key."""
        config = {"level1": {"level2": "value"}}
        result = get_config_value(config, "level1", "missing", "level3", default="fallback")
        assert result == "fallback"

    def test_get_config_value_non_dict_traversal_returns_default(self):
        """Should return default when traversing non-dict value."""
        config = {"key": "string_value"}
        result = get_config_value(config, "key", "nested", default="default")
        assert result == "default"

    def test_get_config_value_none_returns_default(self):
        """Should return default when value is None."""
        config = {"key": None}
        result = get_config_value(config, "key", default="default")
        assert result == "default"

    def test_get_config_value_empty_keys_returns_config(self):
        """Should return entire config when no keys specified."""
        config = {"key": "value"}
        result = get_config_value(config)
        assert result == config

    def test_get_config_value_with_zero_value(self):
        """Should return 0 or False values, not default."""
        config = {"count": 0, "enabled": False}
        assert get_config_value(config, "count") == 0
        assert get_config_value(config, "enabled") is False

    def test_get_config_value_with_empty_string(self):
        """Should return empty string, not default."""
        config = {"text": ""}
        assert get_config_value(config, "text") == ""
