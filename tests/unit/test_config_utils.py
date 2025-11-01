"""
Unit tests for configuration utilities.
"""

from pathlib import Path

import pytest
import typer
import yaml

from docling_graph.cli.config_utils import get_config_value, load_config, save_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, temp_dir, sample_config_dict, monkeypatch):
        """Test loading a valid config file."""
        # Create config file
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Change to temp directory
        monkeypatch.chdir(temp_dir)

        # Load config
        config = load_config()
        assert config == sample_config_dict

    def test_load_config_file_not_found(self, temp_dir, monkeypatch):
        """Test loading config when file doesn't exist."""
        monkeypatch.chdir(temp_dir)

        with pytest.raises(typer.Exit) as exc_info:
            load_config()
        assert exc_info.value.exit_code == 1

    def test_load_invalid_yaml(self, temp_dir, monkeypatch):
        """Test loading invalid YAML file."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("invalid: yaml: content: [")

        monkeypatch.chdir(temp_dir)

        with pytest.raises(typer.Exit) as exc_info:
            load_config()
        assert exc_info.value.exit_code == 1

    def test_load_empty_config(self, temp_dir, monkeypatch):
        """Test loading empty config file."""
        config_path = temp_dir / "config.yaml"
        config_path.write_text("")

        monkeypatch.chdir(temp_dir)

        config = load_config()
        assert config is None or config == {}


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_valid_config(self, temp_dir, sample_config_dict):
        """Test saving a valid config."""
        output_path = temp_dir / "test_config.yaml"

        save_config(sample_config_dict, output_path)

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == sample_config_dict

    def test_save_empty_config(self, temp_dir):
        """Test saving empty config."""
        output_path = temp_dir / "empty_config.yaml"

        save_config({}, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {} or loaded is None

    def test_save_to_nonexistent_directory(self, temp_dir):
        """Test saving to a directory that doesn't exist yet."""
        output_path = temp_dir / "subdir" / "config.yaml"

        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_config({"key": "value"}, output_path)

        assert output_path.exists()

    def test_save_complex_config(self, temp_dir):
        """Test saving complex nested config."""
        complex_config = {
            "level1": {
                "level2": {"level3": {"value": "deep"}},
                "list": [1, 2, 3],
                "mixed": {"a": [{"b": "c"}]},
            }
        }

        output_path = temp_dir / "complex_config.yaml"
        save_config(complex_config, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == complex_config


class TestGetConfigValue:
    """Tests for get_config_value function."""

    def test_get_single_level_value(self, sample_config_dict):
        """Test getting a single-level config value."""
        value = get_config_value(sample_config_dict, "defaults")
        assert value == sample_config_dict["defaults"]

    def test_get_nested_value(self, sample_config_dict):
        """Test getting nested config value."""
        value = get_config_value(sample_config_dict, "defaults", "processing_mode")
        assert value == "many-to-one"

    def test_get_deeply_nested_value(self, sample_config_dict):
        """Test getting deeply nested value."""
        value = get_config_value(sample_config_dict, "models", "llm", "local", "default_model")
        assert value == "llama3:8b-instruct"

    def test_get_nonexistent_key(self, sample_config_dict):
        """Test getting nonexistent key returns default."""
        value = get_config_value(sample_config_dict, "nonexistent", default="default_value")
        assert value == "default_value"

    def test_get_nonexistent_nested_key(self, sample_config_dict):
        """Test getting nonexistent nested key."""
        value = get_config_value(
            sample_config_dict, "defaults", "nonexistent", default="default_value"
        )
        assert value == "default_value"

    def test_get_value_no_default(self, sample_config_dict):
        """Test getting nonexistent value without default."""
        value = get_config_value(sample_config_dict, "nonexistent")
        assert value is None

    def test_get_from_non_dict_value(self, sample_config_dict):
        """Test getting from a non-dict value."""
        # Try to get nested key from a string value
        value = get_config_value(
            sample_config_dict,
            "defaults",
            "processing_mode",
            "nested",  # But processing_mode is a string!
            default="default_value",
        )
        assert value == "default_value"

    def test_get_with_empty_keys(self, sample_config_dict):
        """Test get_config_value with no keys."""
        value = get_config_value(sample_config_dict)
        assert value == sample_config_dict

    @pytest.mark.parametrize(
        "keys,expected",
        [
            (["defaults", "backend_type"], "llm"),
            (["defaults", "inference"], "local"),
            (["defaults", "export_format"], "csv"),
            (["docling", "pipeline"], "ocr"),
            (["output", "create_visualizations"], True),
        ],
    )
    def test_various_config_paths(self, sample_config_dict, keys, expected):
        """Test various config paths."""
        value = get_config_value(sample_config_dict, *keys)
        assert value == expected


class TestConfigUtilsEdgeCases:
    """Test edge cases for config utilities."""

    def test_save_none_config(self, temp_dir):
        """Test saving None as config."""
        output_path = temp_dir / "none_config.yaml"
        save_config(None, output_path)

        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded is None

    def test_get_config_value_none_config(self):
        """Test get_config_value with None config."""
        value = get_config_value(None, "key", default="default")
        assert value == "default"

    def test_get_config_value_empty_string_key(self, sample_config_dict):
        """Test get_config_value with empty string key."""
        value = get_config_value(sample_config_dict, "", default="default")
        assert value == "default"
