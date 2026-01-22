import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.vllm import VllmClient


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_vllm_client_init(mock_get_model_config, mock_openai_class):
    """Test vLLM client initialization."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = [{"id": "llama-7b"}]
    mock_config = MagicMock()

    mock_config.context_limit = 4096

    mock_get_model_config.return_value = mock_config

    client = VllmClient(model="llama-7b")

    assert client.model == "llama-7b"
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "EMPTY"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="EMPTY")


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_vllm_client_init_custom_url(mock_get_model_config, mock_openai_class):
    """Test vLLM client with custom URL."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_config = MagicMock()

    mock_config.context_limit = 8192

    mock_get_model_config.return_value = mock_config

    client = VllmClient(
        model="mistral-7b", base_url="http://remote-server:8000/v1", api_key="custom-key"
    )

    assert client.base_url == "http://remote-server:8000/v1"
    assert client.api_key == "custom-key"
    mock_openai_class.assert_called_once_with(
        base_url="http://remote-server:8000/v1", api_key="custom-key"
    )


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_get_json_response_empty_all_null(mock_get_model_config, mock_openai_class):
    """Test handling of all-null JSON."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_config = MagicMock()

    mock_config.context_limit = 4096

    mock_get_model_config.return_value = mock_config

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"a": None, "b": None})
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"a": None, "b": None}
