"""
Tests for Mistral LLM client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.mistral import MistralClient

from .conftest import MISTRAL_TEST_MODELS


class TestMistralClientInitialization:
    """Test Mistral client initialization."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_initialization_with_api_key(self, mock_mistral_api):
        """Should initialize with API key from environment."""
        client = MistralClient(model="mistral-small-latest")
        assert client.model == "mistral-small-latest"
        assert client.client is not None

    def test_initialization_missing_api_key_raises_error(self, mock_mistral_api):
        """Should raise error if API key not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                MistralClient(model="mistral-small-latest")


class TestMistralClientContextLimit:
    """Test Mistral context limits."""

    @pytest.mark.parametrize("model,expected_limit", MISTRAL_TEST_MODELS)
    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_context_limit_models(self, model, expected_limit, mock_mistral_api):
        """Should have correct context limits for Mistral models."""
        client = MistralClient(model=model)
        assert client.context_limit == expected_limit


class TestMistralClientGetJsonResponse:
    """Test Mistral JSON extraction."""

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_with_string_prompt(self, mock_mistral_api):
        """Should handle string prompt."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        result = client.get_json_response("extract this", "{}")

        assert result["result"] == "test"
        mock_mistral_api.chat.complete.assert_called_once()

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_with_dict_prompt(self, mock_mistral_api):
        """Should handle dict prompt with system and user."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"data": "extracted"}'
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        prompt = {"system": "You are a JSON extractor", "user": "Extract data"}
        result = client.get_json_response(prompt, "{}")

        assert result["data"] == "extracted"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_empty_content_returns_empty_dict(self, mock_mistral_api):
        """Should return empty dict if API returns empty content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_invalid_json_returns_empty_dict(self, mock_mistral_api):
        """Should return empty dict if response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid json"
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_api_error_returns_empty_dict(self, mock_mistral_api):
        """Should return empty dict on API error."""
        mock_mistral_api.chat.complete.side_effect = Exception("API Error")

        client = MistralClient(model="mistral-small-latest")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_uses_json_format(self, mock_mistral_api):
        """Should request JSON format from Mistral."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        client.get_json_response("test", "{}")

        # Verify JSON format was requested
        call_kwargs = mock_mistral_api.chat.complete.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"})
    def test_get_json_response_sets_low_temperature(self, mock_mistral_api):
        """Should use low temperature for consistent extraction."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_mistral_api.chat.complete.return_value = mock_response

        client = MistralClient(model="mistral-small-latest")
        client.get_json_response("test", "{}")

        # Verify low temperature was used
        call_kwargs = mock_mistral_api.chat.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.1
