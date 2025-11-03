"""
Tests for command-specific validators (init, convert, inspect).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.cli.validators import (
    print_dependency_setup_guide,
    print_next_steps_with_deps,
    validate_provider,
)


class TestValidateProvider:
    """Test provider validation."""

    def test_validate_provider_local_ollama(self):
        """Should validate ollama as local provider."""
        result = validate_provider("ollama", "local")
        assert result == "ollama"

    def test_validate_provider_local_vllm(self):
        """Should validate vllm as local provider."""
        result = validate_provider("vllm", "local")
        assert result == "vllm"

    def test_validate_provider_remote_mistral(self):
        """Should validate mistral as remote provider."""
        result = validate_provider("mistral", "remote")
        assert result == "mistral"

    def test_validate_provider_remote_openai(self):
        """Should validate openai as remote provider."""
        result = validate_provider("openai", "remote")
        assert result == "openai"

    def test_validate_provider_invalid_raises_error(self):
        """Should raise error for invalid provider."""
        with pytest.raises(ValueError):
            validate_provider("ollama", "remote")  # ollama is not remote

    def test_validate_provider_wrong_inference_raises_error(self):
        """Should raise error when provider doesn't match inference type."""
        with pytest.raises(ValueError):
            validate_provider("mistral", "local")  # mistral is not local


class TestPrintDependencySetupGuide:
    """Test dependency setup guide printing."""

    @patch("docling_graph.cli.validators.rich_print")
    def test_print_dependency_setup_guide_local(self, mock_print):
        """Should print setup guide for local inference."""
        print_dependency_setup_guide("local")
        mock_print.assert_called()

    @patch("docling_graph.cli.validators.rich_print")
    def test_print_dependency_setup_guide_remote(self, mock_print):
        """Should print setup guide for remote inference."""
        print_dependency_setup_guide("remote")
        mock_print.assert_called()


class TestPrintNextStepsWithDeps:
    """Test next steps printing with dependencies."""

    @patch("docling_graph.cli.validators.print_dependency_setup_guide")
    def test_print_next_steps_with_deps_local(self, mock_guide):
        """Should print next steps for local config."""
        config = {
            "defaults": {"inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        print_next_steps_with_deps(config)
        mock_guide.assert_called_with("local")

    @patch("docling_graph.cli.validators.print_dependency_setup_guide")
    def test_print_next_steps_with_deps_remote(self, mock_guide):
        """Should print next steps for remote config."""
        config = {
            "defaults": {"inference": "remote"},
            "models": {"llm": {"remote": {"provider": "mistral"}}},
        }
        print_next_steps_with_deps(config)
        mock_guide.assert_called_with("remote")

    @patch("docling_graph.cli.validators.print_dependency_setup_guide")
    def test_print_next_steps_defaults_to_remote(self, mock_guide):
        """Should default to remote when config missing."""
        config = {}
        print_next_steps_with_deps(config)
        mock_guide.assert_called_with("remote")
