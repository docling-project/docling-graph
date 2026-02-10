"""
Tests for interactive configuration builder.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from docling_graph.cli.config_builder import (
    ConfigurationBuilder,
    PromptConfig,
    build_config_interactive,
    print_next_steps,
)


class TestPromptConfig:
    """Test PromptConfig dataclass."""

    def test_prompt_config_initialization(self):
        """Should initialize with all required fields."""
        config = PromptConfig(
            label="Test",
            description="Test description",
            options=["opt1", "opt2"],
            default="opt1",
            option_help={"opt1": "Help 1", "opt2": "Help 2"},
            step_num=1,
        )
        assert config.label == "Test"
        assert config.description == "Test description"
        assert config.options == ["opt1", "opt2"]
        assert config.default == "opt1"
        assert config.option_help == {"opt1": "Help 1", "opt2": "Help 2"}
        assert config.step_num == 1


class TestConfigurationBuilder:
    """Test ConfigurationBuilder class."""

    def test_initialization(self):
        """Should initialize with step counter at 1."""
        builder = ConfigurationBuilder()
        assert builder.step_counter == 1

    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_output")
    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_models")
    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_docling")
    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_defaults")
    def test_build_config_calls_all_builders(
        self, mock_defaults, mock_docling, mock_models, mock_output
    ):
        """Should call all builder methods in sequence."""
        mock_defaults.return_value = {"backend": "llm", "inference": "local"}
        mock_docling.return_value = {"pipeline": "ocr"}
        mock_models.return_value = {"llm": {"local": {"provider": "ollama"}}}
        mock_output.return_value = {"directory": "./output"}

        builder = ConfigurationBuilder()
        result = builder.build_config()

        mock_defaults.assert_called_once()
        mock_docling.assert_called_once()
        mock_models.assert_called_once_with("llm", "local")
        mock_output.assert_called_once()

        assert "defaults" in result
        assert "docling" in result
        assert "models" in result
        assert "output" in result

    @patch("typer.prompt")
    def test_prompt_option_returns_selected_value(self, mock_prompt):
        """Should return user's selected option."""
        mock_prompt.return_value = "ocr"

        builder = ConfigurationBuilder()
        config = PromptConfig(
            label="Pipeline",
            description="Choose pipeline",
            options=["ocr", "vision"],
            default="ocr",
            option_help={"ocr": "OCR pipeline", "vision": "Vision pipeline"},
            step_num=1,
        )
        result = builder._prompt_option(config)

        assert result == "ocr"

    @patch("typer.prompt")
    def test_build_defaults_returns_dict(self, mock_prompt):
        """Should return dictionary with default settings."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "staged",  # extraction_contract
            "llm",  # backend
            "local",  # inference
            "csv",  # export_format
        ]

        builder = ConfigurationBuilder()
        result = builder._build_defaults()

        assert isinstance(result, dict)
        assert result["processing_mode"] == "one-to-one"
        assert result["extraction_contract"] == "staged"
        assert result["backend"] == "llm"
        assert result["inference"] == "local"
        assert result["export_format"] == "csv"

    @patch("typer.prompt")
    def test_build_defaults_vlm_forces_local(self, mock_prompt):
        """Should force local inference for VLM backend."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "direct",  # extraction_contract
            "vlm",  # backend
            "csv",  # export_format (no inference prompt for VLM)
        ]

        builder = ConfigurationBuilder()
        result = builder._build_defaults()

        assert result["backend"] == "vlm"
        assert result["inference"] == "local"

    @patch("typer.confirm")
    @patch("typer.prompt")
    def test_build_docling_returns_config(self, mock_prompt, mock_confirm):
        """Should return docling configuration."""
        mock_prompt.return_value = "ocr"
        mock_confirm.side_effect = [True, True, False]  # json, markdown, per-page

        builder = ConfigurationBuilder()
        result = builder._build_docling()

        assert isinstance(result, dict)
        assert "pipeline" in result
        assert "export" in result

    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_vlm_config")
    def test_build_models_vlm_backend(self, mock_vlm):
        """Should build VLM config for VLM backend."""
        mock_vlm.return_value = {
            "vlm": {"local": {"provider": "docling", "model": "llava"}},
            "llm": {"local": {}, "remote": {}},
        }

        builder = ConfigurationBuilder()
        result = builder._build_models("vlm", "local")

        mock_vlm.assert_called_once()
        assert "vlm" in result

    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_local_llm_config")
    def test_build_models_llm_local(self, mock_local_llm):
        """Should build local LLM config for LLM backend with local inference."""
        mock_local_llm.return_value = {
            "llm": {"local": {"provider": "ollama", "model": "llama3"}},
            "vlm": {"local": {}},
        }

        builder = ConfigurationBuilder()
        result = builder._build_models("llm", "local")

        mock_local_llm.assert_called_once()
        assert "llm" in result

    @patch("docling_graph.cli.config_builder.ConfigurationBuilder._build_remote_llm_config")
    def test_build_models_llm_remote(self, mock_remote_llm):
        """Should build remote LLM config for LLM backend with remote inference."""
        mock_remote_llm.return_value = {
            "llm": {
                "remote": {"provider": "openai", "model": "gpt-4"},
                "local": {},
            },
            "vlm": {"local": {}},
        }

        builder = ConfigurationBuilder()
        result = builder._build_models("llm", "remote")

        mock_remote_llm.assert_called_once()
        assert "llm" in result

    @patch("typer.prompt")
    def test_build_vlm_config_returns_full_structure(self, mock_prompt):
        """Should return full model structure for VLM."""
        mock_prompt.return_value = "llava"

        builder = ConfigurationBuilder()
        result = builder._build_vlm_config()

        assert "vlm" in result
        assert "llm" in result
        assert "local" in result["vlm"]

    @patch("typer.prompt")
    def test_build_local_llm_config_returns_full_structure(self, mock_prompt):
        """Should return full model structure for local LLM."""
        mock_prompt.side_effect = ["ollama", "llama3"]

        builder = ConfigurationBuilder()
        result = builder._build_local_llm_config()

        assert "llm" in result
        assert "vlm" in result
        assert "local" in result["llm"]
        assert result["llm"]["local"]["provider"] == "ollama"
        assert result["llm"]["local"]["model"] == "llama3"

    @patch("typer.prompt")
    def test_build_remote_llm_config_returns_full_structure(self, mock_prompt):
        """Should return full model structure for remote LLM."""
        mock_prompt.side_effect = ["openai", "gpt-4"]

        builder = ConfigurationBuilder()
        result = builder._build_remote_llm_config()

        assert "llm" in result
        assert "vlm" in result
        assert "remote" in result["llm"]
        assert result["llm"]["remote"]["provider"] == "openai"
        assert result["llm"]["remote"]["model"] == "gpt-4"

    @patch("typer.prompt")
    def test_build_output_returns_directory_config(self, mock_prompt):
        """Should return output directory configuration."""
        mock_prompt.return_value = "./my_output"

        builder = ConfigurationBuilder()
        result = builder._build_output()

        assert "directory" in result
        assert result["directory"] == "./my_output"


class TestBuildConfigInteractive:
    """Test build_config_interactive function."""

    @patch("docling_graph.cli.config_builder.ConfigurationBuilder.build_config")
    def test_build_config_interactive_returns_dict(self, mock_build):
        """Should return configuration dictionary."""
        mock_build.return_value = {
            "defaults": {"backend": "llm"},
            "docling": {"pipeline": "ocr"},
            "models": {"llm": {"local": {}}},
            "output": {"directory": "./output"},
        }

        result = build_config_interactive()

        assert isinstance(result, dict)
        assert "defaults" in result
        assert "docling" in result
        assert "models" in result
        assert "output" in result


class TestPrintNextSteps:
    """Test print_next_steps function."""

    def test_print_next_steps_with_return_text(self):
        """Should return formatted text when return_text=True."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        result = print_next_steps(config, return_text=True)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Next steps" in result

    def test_print_next_steps_returns_none_when_printing(self):
        """Should return None when return_text=False."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        result = print_next_steps(config, return_text=False)

        assert result is None

    def test_print_next_steps_includes_instructions(self):
        """Should include setup instructions in output."""
        config = {
            "defaults": {"backend": "vlm", "inference": "local"},
            "models": {"vlm": {"provider": "ollama"}},
        }

        result = print_next_steps(config, return_text=True)

        # Should include general instructions (not config-specific)
        assert "Pydantic model" in result or "convert" in result
