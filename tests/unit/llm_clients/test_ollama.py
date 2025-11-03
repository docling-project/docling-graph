"""
Tests for Ollama LLM client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.ollama import OllamaClient

from .conftest import OLLAMA_TEST_MODELS


class TestOllamaClientInitialization:
    """Test Ollama client initialization."""

    def test_initialization_with_valid_model(self, mock_ollama_module):
        """Should initialize with valid model."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        client = OllamaClient(model="llama3.1:8b")
        assert client.model == "llama3.1:8b"
        mock_ollama_module.show.assert_called_once_with("llama3.1:8b")

    def test_initialization_ollama_not_installed_raises_error(self):
        """Should raise error if Ollama not installed."""
        with patch("docling_graph.llm_clients.ollama.ollama", None):
            with pytest.raises(ImportError, match="ollama"):
                OllamaClient(model="llama3.1:8b")

    def test_initialization_model_not_available_raises_error(self, mock_ollama_module):
        """Should raise error if model not available."""
        mock_ollama_module.show.side_effect = Exception("Model not found")
        with pytest.raises(RuntimeError):
            OllamaClient(model="nonexistent-model")


class TestOllamaClientContextLimit:
    """Test Ollama context limits."""

    @pytest.mark.parametrize("model,expected_limit", OLLAMA_TEST_MODELS)
    def test_context_limit_models(self, model, expected_limit, mock_ollama_module):
        """Should have correct context limits for Ollama models."""
        mock_ollama_module.show.return_value = {"name": model}
        client = OllamaClient(model=model)
        assert client.context_limit == expected_limit

    def test_context_limit_unknown_model_defaults(self, mock_ollama_module):
        """Unknown model should default to 8000."""
        mock_ollama_module.show.return_value = {"name": "custom-model"}
        client = OllamaClient(model="custom-model")
        assert client.context_limit == 8000


class TestOllamaClientGetJsonResponse:
    """Test Ollama JSON extraction."""

    def test_get_json_response_with_string_prompt(self, mock_ollama_module):
        """Should handle string prompt."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": '{"result": "test"}'}}

        client = OllamaClient(model="llama3.1:8b")
        result = client.get_json_response("extract this", "{}")

        assert result["result"] == "test"
        mock_ollama_module.chat.assert_called_once()

    def test_get_json_response_with_dict_prompt(self, mock_ollama_module):
        """Should handle dict prompt with system and user."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": '{"data": "extracted"}'}}

        client = OllamaClient(model="llama3.1:8b")
        prompt = {"system": "You are a JSON extractor", "user": "Extract data"}
        result = client.get_json_response(prompt, "{}")

        assert result["data"] == "extracted"

    def test_get_json_response_invalid_json_returns_empty_dict(self, mock_ollama_module):
        """Should return empty dict for invalid JSON."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": "invalid json"}}

        client = OllamaClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_empty_content_returns_empty_dict(self, mock_ollama_module):
        """Should return empty dict for empty content."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": ""}}

        client = OllamaClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_api_error_returns_empty_dict(self, mock_ollama_module):
        """Should return empty dict on API error."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.side_effect = Exception("Connection Error")

        client = OllamaClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_uses_json_format(self, mock_ollama_module):
        """Should request JSON format from Ollama."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": "{}"}}

        client = OllamaClient(model="llama3.1:8b")
        client.get_json_response("test", "{}")

        # Verify format was set to "json"
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs.get("format") == "json"

    def test_get_json_response_sets_low_temperature(self, mock_ollama_module):
        """Should use low temperature for consistent extraction."""
        mock_ollama_module.show.return_value = {"name": "llama3.1:8b"}
        mock_ollama_module.chat.return_value = {"message": {"content": "{}"}}

        client = OllamaClient(model="llama3.1:8b")
        client.get_json_response("test", "{}")

        # Verify low temperature was used
        call_kwargs = mock_ollama_module.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.1
