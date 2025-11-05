import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.vllm import VllmClient


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_vllm_client_init(mock_get_context_limit, mock_openai_class):
    """Test vLLM client initialization."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = [{"id": "llama-7b"}]
    mock_get_context_limit.return_value = 4096

    client = VllmClient(model="llama-7b")

    assert client.model == "llama-7b"
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "EMPTY"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="EMPTY")


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_vllm_client_init_custom_url(mock_get_context_limit, mock_openai_class):
    """Test vLLM client with custom URL."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 8192

    client = VllmClient(
        model="mistral-7b", base_url="http://remote-server:8000/v1", api_key="custom-key"
    )

    assert client.base_url == "http://remote-server:8000/v1"
    assert client.api_key == "custom-key"
    mock_openai_class.assert_called_once_with(
        base_url="http://remote-server:8000/v1", api_key="custom-key"
    )


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_vllm_client_init_connection_error(mock_get_context_limit, mock_openai_class):
    """Test handling of connection errors during init."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.side_effect = RuntimeError("Connection refused")
    mock_get_context_limit.return_value = 4096

    with pytest.raises(RuntimeError):
        VllmClient(model="llama-7b")


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_dict_prompt(mock_get_context_limit, mock_openai_class):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    response_data = {"name": "Delta", "score": 95}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(
        prompt={"system": "Extract info", "user": "Process data"}, schema_json="{}"
    )

    assert result == response_data
    mock_client.chat.completions.create.assert_called()


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_string_prompt(mock_get_context_limit, mock_openai_class):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    response_data = {"status": "complete"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="Process", schema_json="{}")

    assert result == response_data


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_empty_content(mock_get_context_limit, mock_openai_class):
    """Test handling of empty response content."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_invalid_json(mock_get_context_limit, mock_openai_class):
    """Test handling of invalid JSON."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json {{"
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_api_error(mock_get_context_limit, mock_openai_class):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    mock_client.chat.completions.create.side_effect = RuntimeError("API Error")

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.vllm.OpenAI")
@patch("docling_graph.llm_clients.vllm.get_context_limit")
def test_get_json_response_empty_all_null(mock_get_context_limit, mock_openai_class):
    """Test handling of all-null JSON."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"a": None, "b": None})
    mock_client.chat.completions.create.return_value = mock_response

    client = VllmClient(model="llama-7b")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"a": None, "b": None}
