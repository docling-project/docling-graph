"""Tests for MistralClient with refactored architecture."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.exceptions import ClientError, ConfigurationError
from docling_graph.llm_clients.mistral import MistralClient


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")


@pytest.fixture
def mock_mistral_client():
    """Create a mock Mistral client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"test": "response"}'
    mock_client.chat.complete.return_value = mock_response
    return mock_client


class TestMistralClient:
    """Test suite for MistralClient."""

    def test_provider_id(self, mock_env_vars):
        """Test provider ID."""
        with patch("docling_graph.llm_clients.mistral.Mistral"):
            client = MistralClient(model="mistral-large-latest")
            assert client._provider_id() == "mistral"

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_client_initialization(self, mock_mistral_class, mock_env_vars):
        """Test client initialization."""
        mock_mistral_class.return_value = MagicMock()

        client = MistralClient(model="mistral-large-latest")

        assert client.model == "mistral-large-latest"
        assert client.api_key == "test-mistral-key"
        mock_mistral_class.assert_called_once_with(api_key="test-mistral-key")

    def test_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ConfigurationError."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            MistralClient(model="mistral-large-latest")

        assert "MISTRAL_API_KEY" in str(exc_info.value)

    @patch("docling_graph.llm_clients.mistral.Mistral")
    @patch("docling_graph.llm_clients.response_handler.ResponseHandler.parse_json_response")
    def test_get_json_response_success(self, mock_parse, mock_mistral_class, mock_env_vars):
        """Test successful JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "Bob", "role": "engineer"}'
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        mock_parse.return_value = {"name": "Bob", "role": "engineer"}

        client = MistralClient(model="mistral-large-latest")
        result = client.get_json_response(
            prompt={"system": "Extract data", "user": "Bob is an engineer"}, schema_json="{}"
        )

        assert result == {"name": "Bob", "role": "engineer"}
        mock_client.chat.complete.assert_called_once()

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_with_messages(self, mock_mistral_class, mock_env_vars):
        """Test _call_api with messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model="mistral-large-latest")
        messages = [{"role": "user", "content": "test"}]

        response = client._call_api(messages, temperature=0.1)

        assert response == '{"result": "success"}'
        mock_client.chat.complete.assert_called_once()

        # Verify call arguments
        call_args = mock_client.chat.complete.call_args
        assert call_args[1]["model"] == "mistral-large-latest"
        assert call_args[1]["messages"] == messages
        assert call_args[1]["temperature"] == 0.1

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_empty_response(self, mock_mistral_class, mock_env_vars):
        """Test handling of empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model="mistral-large-latest")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ClientError) as exc_info:
            client._call_api(messages)

        assert "empty content" in str(exc_info.value).lower()
        assert exc_info.value.details["model"] == "mistral-large-latest"

    @patch("docling_graph.llm_clients.mistral.Mistral")
    def test_call_api_exception_handling(self, mock_mistral_class, mock_env_vars):
        """Test API exception handling."""
        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = Exception("API Error")
        mock_mistral_class.return_value = mock_client

        client = MistralClient(model="mistral-large-latest")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ClientError) as exc_info:
            client._call_api(messages)

        assert "API Error" in str(exc_info.value)
        assert exc_info.value.details["model"] == "mistral-large-latest"
