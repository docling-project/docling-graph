"""
Tests for OpenAI LLM client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.openai import OpenAIClient

from .conftest import OPENAI_TEST_MODELS


class TestOpenAIClientInitialization:
    """Test OpenAI client initialization."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_initialization_with_api_key(self, mock_openai_client):
        """Should initialize with API key from environment."""
        client = OpenAIClient(model="gpt-4-turbo")
        assert client.model == "gpt-4-turbo"
        assert client.client is not None

    def test_initialization_missing_api_key_raises_error(self, mock_openai_client):
        """Should raise error if API key not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIClient(model="gpt-4-turbo")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_initialization_different_models(self, mock_openai_client):
        """Should accept different model names."""
        for model, _ in OPENAI_TEST_MODELS:
            client = OpenAIClient(model=model)
            assert client.model == model


class TestOpenAIClientContextLimit:
    """Test OpenAI context limits."""

    @pytest.mark.parametrize("model,expected_limit", OPENAI_TEST_MODELS)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_context_limit_models(self, model, expected_limit, mock_openai_client):
        """Should have correct context limits for OpenAI models."""
        client = OpenAIClient(model=model)
        assert client.context_limit == expected_limit

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_context_limit_unknown_model_defaults(self, mock_openai_client):
        """Unknown model should default to 128k."""
        client = OpenAIClient(model="gpt-custom-model")
        assert client.context_limit == 128000


class TestOpenAIClientGetJsonResponse:
    """Test OpenAI JSON extraction."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_with_string_prompt(self, mock_openai_client):
        """Should handle string prompt."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        result = client.get_json_response("extract this", "{}")

        assert result["result"] == "test"
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_with_dict_prompt(self, mock_openai_client):
        """Should handle dict prompt with system and user."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"data": "extracted"}'
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        prompt = {"system": "You are a JSON extractor", "user": "Extract data"}
        result = client.get_json_response(prompt, "{}")

        assert result["data"] == "extracted"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_empty_content_returns_empty_dict(self, mock_openai_client):
        """Should return empty dict if API returns empty content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_invalid_json_returns_empty_dict(self, mock_openai_client):
        """Should return empty dict if response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid json"
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_api_error_returns_empty_dict(self, mock_openai_client):
        """Should return empty dict on API error."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        client = OpenAIClient(model="gpt-4-turbo")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_uses_json_format(self, mock_openai_client):
        """Should request JSON format from OpenAI."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        client.get_json_response("test", "{}")

        # Verify JSON format was requested
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_json_response_sets_low_temperature(self, mock_openai_client):
        """Should use low temperature for consistent extraction."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4-turbo")
        client.get_json_response("test", "{}")

        # Verify low temperature was used
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.1
