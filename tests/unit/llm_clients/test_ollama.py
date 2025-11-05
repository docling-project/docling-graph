import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.ollama import OllamaClient


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_ollama_client_init(mock_get_context_limit, mock_ollama, monkeypatch):
    """Test Ollama client initialization."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    client = OllamaClient(model="llama2")

    assert client.model == "llama2"
    assert client.context_limit == 4096
    mock_ollama.show.assert_called_once_with("llama2")


@patch("docling_graph.llm_clients.ollama.ollama", None)
def test_ollama_client_init_import_error():
    """Test that missing ollama package raises error."""
    with pytest.raises(ImportError, match="Ollama client could not be imported"):
        OllamaClient(model="llama2")


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_ollama_client_init_connection_error(mock_get_context_limit, mock_ollama):
    """Test handling of connection errors during init."""
    mock_ollama.show.side_effect = RuntimeError("Connection refused")
    mock_get_context_limit.return_value = 4096

    with pytest.raises(RuntimeError, match="Connection refused"):
        OllamaClient(model="llama2")


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_dict_prompt(mock_get_context_limit, mock_ollama):
    """Test JSON response with dict-style prompt."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    response_data = {"name": "Charlie", "age": 25}
    mock_ollama.chat.return_value = {"message": {"content": json.dumps(response_data)}}

    client = OllamaClient(model="llama2")
    result = client.get_json_response(
        prompt={"system": "Extract info", "user": "Process this"}, schema_json="{}"
    )

    assert result == response_data
    mock_ollama.chat.assert_called()


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_string_prompt(mock_get_context_limit, mock_ollama):
    """Test JSON response with string prompt."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    response_data = {"result": "done"}
    mock_ollama.chat.return_value = {"message": {"content": json.dumps(response_data)}}

    client = OllamaClient(model="llama2")
    result = client.get_json_response(prompt="Extract", schema_json="{}")

    assert result == response_data

    # Verify messages format
    call_args = mock_ollama.chat.call_args
    assert call_args[1]["messages"][0]["role"] == "user"


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_invalid_json(mock_get_context_limit, mock_ollama):
    """Test handling of invalid JSON."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    mock_ollama.chat.return_value = {"message": {"content": "invalid {json"}}

    client = OllamaClient(model="llama2")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_empty_content(mock_get_context_limit, mock_ollama):
    """Test handling of empty response."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    mock_ollama.chat.return_value = {"message": {"content": ""}}

    client = OllamaClient(model="llama2")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_api_error(mock_get_context_limit, mock_ollama):
    """Test handling of API errors."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    mock_ollama.chat.side_effect = RuntimeError("API Error")

    client = OllamaClient(model="llama2")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.ollama.get_context_limit")
def test_get_json_response_empty_all_null(mock_get_context_limit, mock_ollama):
    """Test handling of all-null JSON."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_get_context_limit.return_value = 4096

    mock_ollama.chat.return_value = {"message": {"content": json.dumps({"a": None, "b": None})}}

    client = OllamaClient(model="llama2")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"a": None, "b": None}
