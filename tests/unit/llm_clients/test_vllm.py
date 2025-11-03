"""
Tests for vLLM client.
"""

from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.vllm import VllmClient

from .conftest import VLLM_TEST_MODELS


class TestVllmClientInitialization:
    """Test vLLM client initialization."""

    def test_initialization_with_default_params(self, mock_vllm_openai_client):
        """Should initialize with default parameters."""
        client = VllmClient(model="meta-llama/Llama-3.1-8B")

        assert client.model == "meta-llama/Llama-3.1-8B"
        assert client.base_url == "http://localhost:8000/v1"
        assert client.api_key == "EMPTY"

    def test_initialization_with_custom_base_url(self, mock_vllm_openai_client):
        """Should initialize with custom base URL."""
        client = VllmClient(model="llama3.1:8b", base_url="http://192.168.1.100:8000/v1")

        assert client.base_url == "http://192.168.1.100:8000/v1"

    def test_initialization_with_custom_api_key(self, mock_vllm_openai_client):
        """Should initialize with custom API key."""
        client = VllmClient(model="llama3.1:8b", api_key="custom-key-123")

        assert client.api_key == "custom-key-123"

    def test_initialization_creates_openai_client(self, mock_vllm_openai_client):
        """Should create OpenAI client for vLLM."""
        client = VllmClient(model="llama3.1:8b")
        assert client.client is not None


class TestVllmClientContextLimit:
    """Test vLLM context limits."""

    @pytest.mark.parametrize("model,expected_limit", VLLM_TEST_MODELS)
    def test_context_limit_models(self, model, expected_limit, mock_vllm_openai_client):
        """Should have correct context limits for vLLM models."""
        client = VllmClient(model=model)
        assert client.context_limit == expected_limit

    def test_context_limit_unknown_model(self, mock_vllm_openai_client):
        """Unknown model should default to 32k."""
        client = VllmClient(model="custom-model:7b")
        assert client.context_limit == 32000


class TestVllmClientGetJsonResponse:
    """Test vLLM JSON extraction."""

    def test_get_json_response_with_string_prompt(self, mock_vllm_openai_client):
        """Should handle string prompt."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        result = client.get_json_response("extract this", "{}")

        assert result["result"] == "test"
        mock_vllm_openai_client.chat.completions.create.assert_called_once()

    def test_get_json_response_with_dict_prompt(self, mock_vllm_openai_client):
        """Should handle dict prompt with system and user."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"data": "extracted"}'
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        prompt = {"system": "You are a JSON extractor", "user": "Extract data"}
        result = client.get_json_response(prompt, "{}")

        assert result["data"] == "extracted"

    def test_get_json_response_empty_content_returns_empty_dict(self, mock_vllm_openai_client):
        """Should return empty dict if vLLM returns empty content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_invalid_json_returns_empty_dict(self, mock_vllm_openai_client):
        """Should return empty dict for invalid JSON."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "invalid json"
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_api_error_returns_empty_dict(self, mock_vllm_openai_client):
        """Should return empty dict on API error."""
        mock_vllm_openai_client.chat.completions.create.side_effect = Exception("Connection Error")

        client = VllmClient(model="llama3.1:8b")
        result = client.get_json_response("test", "{}")

        assert result == {}

    def test_get_json_response_requests_json_format(self, mock_vllm_openai_client):
        """Should request JSON format from vLLM."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"test": "data"}'
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        client.get_json_response("test", "{}")

        # Verify JSON format was requested
        call_kwargs = mock_vllm_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"

    def test_get_json_response_sets_temperature_low(self, mock_vllm_openai_client):
        """Should use low temperature for consistent extraction."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "{}"
        mock_vllm_openai_client.chat.completions.create.return_value = mock_response

        client = VllmClient(model="llama3.1:8b")
        client.get_json_response("test", "{}")

        # Verify low temperature was used
        call_kwargs = mock_vllm_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.1
