"""
Tests for Google Gemini LLM client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.gemini import GeminiClient

from .conftest import GEMINI_TEST_MODELS


class TestGeminiClientInitialization:
    """Test Gemini client initialization."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_initialization_with_api_key(self, mock_gemini):
        """Should initialize with API key."""
        client = GeminiClient(model="gemini-1.5-pro")
        assert client.model == "gemini-1.5-pro"
        assert client.client is not None

    def test_initialization_missing_api_key_raises_error(self, mock_gemini):
        """Should raise error if API key not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiClient(model="gemini-1.5-pro")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_initialization_different_models(self, mock_gemini):
        """Should accept different model names from conftest."""
        for model, _ in GEMINI_TEST_MODELS:
            client = GeminiClient(model=model)
            assert client.model == model


class TestGeminiClientContextLimit:
    """Test Gemini context limits."""

    @pytest.mark.parametrize("model,expected_limit", GEMINI_TEST_MODELS)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_context_limit_models(self, model, expected_limit, mock_gemini):
        """Should have correct context limits for Gemini models from conftest."""
        client = GeminiClient(model=model)
        assert client.context_limit == expected_limit

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_context_limit_unknown_model_defaults(self, mock_gemini):
        """Unknown model should default to 1M tokens."""
        client = GeminiClient(model="gemini-custom-model")
        assert client.context_limit == 1000000


class TestGeminiClientGetJsonResponse:
    """Test Gemini JSON extraction."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_with_string_prompt(self, mock_gemini):
        """Should handle string prompt."""
        mock_response = MagicMock()
        mock_response.text = '{"result": "test"}'
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        result = client.get_json_response("extract this", "{}")

        assert result["result"] == "test"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_with_dict_prompt(self, mock_gemini):
        """Should handle dict prompt with system and user."""
        mock_response = MagicMock()
        mock_response.text = '{"data": "extracted"}'
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        prompt = {"system": "You are a JSON extractor", "user": "Extract data"}
        result = client.get_json_response(prompt, "{}")

        assert result["data"] == "extracted"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_empty_content_returns_empty_dict(self, mock_gemini):
        """Should return empty dict if API returns empty content."""
        mock_response = MagicMock()
        mock_response.text = None
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_invalid_json_returns_empty_dict(self, mock_gemini):
        """Should return empty dict if response is not valid JSON."""
        mock_response = MagicMock()
        mock_response.text = "not valid json"
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_api_error_returns_empty_dict(self, mock_gemini):
        """Should return empty dict on API error."""
        mock_gemini.Client.return_value.models.generate_content.side_effect = Exception("API Error")

        client = GeminiClient(model="gemini-1.5-pro")
        result = client.get_json_response("test", "{}")

        assert result == {}

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_uses_json_format(self, mock_gemini):
        """Should request JSON format from Gemini."""
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        client.get_json_response("test", "{}")

        # Verify JSON format was requested
        call_kwargs = mock_gemini.Client.return_value.models.generate_content.call_args[1]
        assert call_kwargs["config"].response_mime_type == "application/json"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_get_json_response_sets_low_temperature(self, mock_gemini):
        """Should use low temperature for consistent extraction."""
        mock_response = MagicMock()
        mock_response.text = "{}"
        mock_gemini.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(model="gemini-1.5-pro")
        client.get_json_response("test", "{}")

        # Verify low temperature was used
        call_kwargs = mock_gemini.Client.return_value.models.generate_content.call_args[1]
        assert call_kwargs["config"].temperature == 0.1
