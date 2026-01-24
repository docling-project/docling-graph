"""Tests for BaseLlmClient with template method pattern."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.exceptions import ClientError, ConfigurationError
from docling_graph.llm_clients.base import BaseLlmClient


class MockLlmClient(BaseLlmClient):
    """Mock implementation of BaseLlmClient for testing."""

    def _provider_id(self) -> str:
        return "test_provider"

    def _setup_client(self, **kwargs: Any) -> None:
        self.test_value = kwargs.get("test_value", "default")
        self.api_key = kwargs.get("api_key", "test_key")

    def _call_api(self, messages, **params: Any) -> tuple[str, dict[str, Any]]:
        if hasattr(self, "_mock_response"):
            response = self._mock_response
        else:
            response = '{"test": "response"}'

        # Return tuple with metadata
        metadata = {"finish_reason": "stop", "model": self.model}
        return response, metadata


class TestBaseLlmClient:
    """Test suite for BaseLlmClient."""

    def test_provider_id_implementation(self):
        """Test that _provider_id must be implemented."""
        client = MockLlmClient(model="test-model")
        assert client._provider_id() == "test_provider"

    def test_client_initialization(self):
        """Test client initialization with kwargs."""
        client = MockLlmClient(model="test-model", test_value="custom")
        assert client.model == "test-model"
        assert client.test_value == "custom"

    @patch("docling_graph.llm_clients.config.get_model_config")
    def test_context_limit_property(self, mock_get_model_config):
        """Test context_limit property loads from config."""
        mock_config = MagicMock()
        mock_config.context_limit = 8192
        mock_get_model_config.return_value = mock_config
        client = MockLlmClient(model="test-model")
        assert client.context_limit == 8192
        mock_get_model_config.assert_called_with("test_provider", "test-model")

    @patch("docling_graph.llm_clients.response_handler.ResponseHandler.parse_json_response")
    def test_get_json_response_success(self, mock_parse):
        """Test successful JSON response."""
        mock_parse.return_value = {"result": "success"}
        client = MockLlmClient(model="test-model")

        response = client.get_json_response(prompt="test prompt", schema_json="{}")

        assert response == {"result": "success"}
        mock_parse.assert_called_once()

    def test_prepare_messages_string_prompt(self):
        """Test message preparation with string prompt."""
        client = MockLlmClient(model="test-model")
        messages = client._prepare_messages("test prompt")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test prompt"

    def test_prepare_messages_dict_prompt(self):
        """Test message preparation with dict prompt."""
        client = MockLlmClient(model="test-model")
        messages = client._prepare_messages({"system": "system prompt", "user": "user prompt"})

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user prompt"

    def test_prepare_messages_dict_user_only(self):
        """Test message preparation with dict containing only user."""
        client = MockLlmClient(model="test-model")
        messages = client._prepare_messages({"user": "user prompt"})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "user prompt"

    @patch.dict("os.environ", {"TEST_KEY": "test_value"})
    def test_get_required_env_success(self):
        """Test getting required environment variable."""
        client = MockLlmClient(model="test-model")
        value = client._get_required_env("TEST_KEY")
        assert value == "test_value"

    def test_get_required_env_missing(self):
        """Test missing required environment variable raises error."""
        client = MockLlmClient(model="test-model")

        with pytest.raises(ConfigurationError) as exc_info:
            client._get_required_env("MISSING_KEY")

        assert "MISSING_KEY" in str(exc_info.value)
        assert exc_info.value.details["variable"] == "MISSING_KEY"

    def test_needs_aggressive_cleaning_default(self):
        """Test default aggressive cleaning behavior."""
        client = MockLlmClient(model="test-model")
        assert client._needs_aggressive_cleaning() is False

    @patch("docling_graph.llm_clients.response_handler.ResponseHandler.parse_json_response")
    def test_get_json_response_uses_aggressive_cleaning(self, mock_parse):
        """Test that aggressive cleaning flag is passed correctly."""
        mock_parse.return_value = {"test": "data"}

        class AggressiveClient(MockLlmClient):
            def _needs_aggressive_cleaning(self) -> bool:
                return True

        client = AggressiveClient(model="test-model")
        client.get_json_response("test", "{}")

        # Verify aggressive_clean=True was passed
        call_args = mock_parse.call_args
        assert call_args[1]["aggressive_clean"] is True

    def test_call_api_must_be_implemented(self):
        """Test that _call_api must be implemented."""

        class IncompleteClient(BaseLlmClient):
            def _provider_id(self) -> str:
                return "incomplete"

            def _setup_client(self, **kwargs: Any) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteClient(model="test")

    @patch("docling_graph.llm_clients.response_handler.ResponseHandler.parse_json_response")
    def test_client_error_propagation(self, mock_parse):
        """Test that ClientError is propagated correctly."""
        mock_parse.side_effect = ClientError("Parse failed", details={"error": "invalid"})

        client = MockLlmClient(model="test-model")

        with pytest.raises(ClientError) as exc_info:
            client.get_json_response("test", "{}")

        assert "Parse failed" in str(exc_info.value)
        assert exc_info.value.details["error"] == "invalid"
