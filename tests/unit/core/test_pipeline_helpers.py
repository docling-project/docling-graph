"""
Tests for pipeline helper functions.
"""

import importlib
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import BaseModel

from docling_graph.pipeline import (
    _get_model_config,
    _initialize_llm_client,
    _load_template_class,
)


# Test Pydantic models (renamed to avoid pytest collection)
class SamplePydanticModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class TestLoadTemplateClass:
    """Test template loading functionality."""

    def test_load_template_class_success(self):
        """Should load valid Pydantic template."""
        template_str = "tests.fixtures.test_template.SamplePydanticModel"

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.SamplePydanticModel = SamplePydanticModel
            mock_import.return_value = mock_module

            result = _load_template_class(template_str)
            assert result is SamplePydanticModel

    def test_load_template_class_with_dots_in_path(self):
        """Should handle multiple dots in class path."""
        template_str = "examples.templates.invoice.Invoice"

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            # Mock an invoice class
            invoice_model = type("Invoice", (BaseModel,), {"__annotations__": {}})
            mock_module.Invoice = invoice_model
            mock_import.return_value = mock_module

            result = _load_template_class(template_str)
            assert result is invoice_model

    def test_load_template_class_invalid_format(self):
        """Should raise exception for invalid format."""
        with pytest.raises(ValueError):
            _load_template_class("invalid_no_dot_path")

    def test_load_template_class_module_not_found(self):
        """Should raise exception if module not found."""
        with patch("importlib.import_module", side_effect=ModuleNotFoundError()):
            with pytest.raises(ModuleNotFoundError):
                _load_template_class("nonexistent.module.Model")

    def test_load_template_class_not_basemodel(self):
        """Should raise TypeError if class is not Pydantic model."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.NotAModel = str  # Not a BaseModel
            mock_import.return_value = mock_module

            with pytest.raises(TypeError):
                _load_template_class("some.module.NotAModel")

    def test_load_template_class_attribute_not_found(self):
        """Should raise exception if class not found in module."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=[])  # Empty module
            mock_import.return_value = mock_module

            with pytest.raises(AttributeError):
                _load_template_class("some.module.MissingClass")


class TestGetModelConfig:
    """Test model configuration retrieval."""

    def test_get_model_config_local_llm(self):
        """Should return config for local LLM."""
        config = {"llm": {"local": {"provider": "ollama", "default_model": "llama-3.1-8b"}}}

        result = _get_model_config(config, "llm", "local")

        assert result["model"] == "llama-3.1-8b"
        assert result["provider"] == "ollama"

    def test_get_model_config_remote_llm(self):
        """Should return config for remote LLM."""
        config = {"llm": {"remote": {"provider": "mistral", "default_model": "mistral-small"}}}

        result = _get_model_config(config, "llm", "remote")

        assert result["model"] == "mistral-small"
        assert result["provider"] == "mistral"

    def test_get_model_config_vlm(self):
        """Should return config for VLM."""
        config = {"vlm": {"local": {"provider": "docling", "default_model": "nuextract"}}}

        result = _get_model_config(config, "vlm", "local")

        assert result["model"] == "nuextract"
        assert result["provider"] == "docling"

    def test_get_model_config_with_override(self):
        """Should apply model override."""
        config = {"llm": {"local": {"provider": "ollama", "default_model": "llama-3.1-8b"}}}

        result = _get_model_config(config, "llm", "local", model_override="custom-model")

        assert result["model"] == "custom-model"

    def test_get_model_config_with_provider_override(self):
        """Should apply provider override."""
        config = {"llm": {"remote": {"provider": "mistral", "default_model": "mistral-small"}}}

        result = _get_model_config(config, "llm", "remote", provider_override="openai")

        assert result["provider"] == "openai"

    def test_get_model_config_missing_raises_error(self):
        """Should raise ValueError if config missing."""
        config = {}

        with pytest.raises(ValueError):
            _get_model_config(config, "llm", "local")

    def test_get_model_config_default_provider(self):
        """Should use default provider if not specified."""
        config = {"llm": {"local": {"default_model": "llama"}}}

        result = _get_model_config(config, "llm", "local")

        assert result["provider"] == "ollama"  # Default for local


class TestInitializeLLMClient:
    """Test LLM client initialization."""

    def test_initialize_llm_client_ollama(self):
        """Should initialize Ollama client."""
        with patch("docling_graph.pipeline.get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client_class.return_value = MagicMock()
            mock_get_client.return_value = mock_client_class

            _initialize_llm_client("ollama", "llama-3.1-8b")

            mock_get_client.assert_called_with("ollama")
            mock_client_class.assert_called_with(model="llama-3.1-8b")

    def test_initialize_llm_client_mistral(self):
        """Should initialize Mistral client."""
        with patch("docling_graph.pipeline.get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client_class.return_value = MagicMock()
            mock_get_client.return_value = mock_client_class

            _initialize_llm_client("mistral", "mistral-small")

            mock_get_client.assert_called_with("mistral")
            mock_client_class.assert_called_with(model="mistral-small")

    def test_initialize_llm_client_invalid_provider(self):
        """Should raise error for invalid provider."""
        with patch("docling_graph.pipeline.get_client", side_effect=ValueError("Unknown provider")):
            with pytest.raises(ValueError):
                _initialize_llm_client("invalid_provider", "model")

    def test_initialize_llm_client_vllm(self):
        """Should initialize vLLM client."""
        with patch("docling_graph.pipeline.get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client_class.return_value = MagicMock()
            mock_get_client.return_value = mock_client_class

            _initialize_llm_client("openai", "gpt-4")
            mock_get_client.assert_called_with("openai")

    def test_initialize_llm_client_openai(self):
        """Should initialize OpenAI client."""
        with patch("docling_graph.pipeline.get_client") as mock_get_client:
            mock_client_class = MagicMock()
            mock_client_class.return_value = MagicMock()
            mock_get_client.return_value = mock_client_class

            _initialize_llm_client("openai", "gpt-4")
            mock_get_client.assert_called_with("openai")
