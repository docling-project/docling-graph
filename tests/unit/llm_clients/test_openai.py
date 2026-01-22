import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from docling_graph.llm_clients.openai import OpenAIClient


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-123")


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_openai_client_init(mock_get_model_config, mock_openai_class, mock_env_vars):
    """Test OpenAI client initialization."""
    mock_openai_class.return_value = MagicMock()
    mock_config = MagicMock()
    mock_config.context_limit = 4096
    mock_get_model_config.return_value = mock_config

    client = OpenAIClient(model="gpt-4")

    assert client.model == "gpt-4"
    assert client.api_key == "test-api-key-123"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(api_key="test-api-key-123")
