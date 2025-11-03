"""
Tests for interactive configuration builder.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.cli.config_builder import (
    _prompt_defaults,
    _prompt_docling,
    _prompt_models,
    _prompt_output,
    build_config_interactive,
)


class TestPromptDefaults:
    """Test default settings prompt."""

    @patch("typer.prompt")
    def test_prompt_defaults_returns_dict(self, mock_prompt):
        """Should return dictionary with default settings."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "llm",  # backend
            "local",  # inference
            "csv",  # export_format
        ]

        result = _prompt_defaults()

        assert isinstance(result, dict)
        assert "processing_mode" in result
        assert "backend" in result
        assert "inference" in result
        assert "export_format" in result

    @patch("typer.prompt")
    def test_prompt_defaults_vlm_sets_local_inference(self, mock_prompt):
        """Should force local inference for VLM backend."""
        mock_prompt.side_effect = [
            "one-to-one",  # processing_mode
            "vlm",  # backend
            "csv",  # export_format
        ]

        result = _prompt_defaults()

        assert result["backend"] == "vlm"
        assert result["inference"] == "local"


class TestPromptDocling:
    """Test Docling configuration prompt."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_docling_returns_config(self, mock_confirm, mock_prompt):
        """Should return Docling configuration dictionary."""
        mock_prompt.return_value = "ocr"
        mock_confirm.side_effect = [True, True, False]

        result = _prompt_docling()

        assert isinstance(result, dict)
        assert "pipeline" in result
        assert "export" in result
        assert result["pipeline"] == "ocr"

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_docling_export_settings(self, mock_confirm, mock_prompt):
        """Should include export settings in result."""
        mock_prompt.return_value = "vision"
        mock_confirm.side_effect = [True, False, True]

        result = _prompt_docling()

        assert result["export"]["docling_json"] is True
        assert result["export"]["markdown"] is False
        assert result["export"]["per_page_markdown"] is True


class TestPromptModels:
    """Test model configuration prompt."""

    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_models_llm_local(self, mock_confirm, mock_prompt):
        """Should configure local LLM model."""
        mock_prompt.side_effect = [
            "vllm",  # local_provider
            "llama-3.1-8b",  # llm_model
        ]

        result = _prompt_models("llm", "local")

        assert "vlm" in result
        assert "llm" in result
        assert result["llm"]["local"]["provider"] == "vllm"

    @patch("typer.prompt")
    def test_prompt_models_vlm_backend(self, mock_prompt):
        """Should configure VLM model."""
        mock_prompt.return_value = "numind/NuExtract-2.0-2B"

        result = _prompt_models("vlm", "local")

        assert "vlm" in result
        assert result["vlm"]["local"]["provider"] == "docling"


class TestPromptOutput:
    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_prompt_output_returns_config(self, mock_confirm, mock_prompt):
        mock_prompt.return_value = "outputs"
        mock_confirm.side_effect = [True, True]

        result = _prompt_output()

        assert isinstance(result, dict)
        # Key changed from 'default_directory' to 'directory'
        assert "directory" in result


class TestBuildConfigInteractive:
    @patch("typer.prompt")
    @patch("typer.confirm")
    def test_build_config_has_all_required_sections(self, mock_confirm, mock_prompt):
        """Should have all required configuration sections."""
        mock_prompt.side_effect = [
            "one-to-one",
            "llm",
            "local",
            "csv",  # defaults
            "vision",  # docling
            "ollama",
            "llama3:8b",  # models - local
            "outputs",  # output
        ]
        mock_confirm.side_effect = [True, True, False, True, True]

        result = build_config_interactive()

        assert result["defaults"]["processing_mode"] == "one-to-one"
        assert result["defaults"]["backend"] == "llm"
        assert result["docling"]["pipeline"] == "vision"
        # Fix here: change 'default_directory' to 'directory'
        assert result["output"]["directory"] == "outputs"
