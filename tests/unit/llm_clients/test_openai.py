import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from docling_graph.llm_clients.openai import OpenAIClient


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-123")


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_openai_client_init(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test OpenAI client initialization."""
    mock_openai_class.return_value = MagicMock()
    mock_get_context_limit.return_value = 4096

    client = OpenAIClient(model="gpt-4")

    assert client.model == "gpt-4"
    assert client.api_key == "test-api-key-123"
    assert client.context_limit == 4096
    mock_openai_class.assert_called_once_with(api_key="test-api-key-123")


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_openai_client_init_no_api_key(mock_get_context_limit, mock_openai_class, monkeypatch):
    """Test that missing API key raises error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
        OpenAIClient(model="gpt-4")


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_dict_prompt(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    # Mock API response
    response_data = {"name": "Alice", "age": 30, "city": "NYC"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(
        prompt={"system": "Extract info", "user": "John from NY"}, schema_json="{}"
    )

    assert result == response_data
    mock_client.chat.completions.create.assert_called_once()


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_string_prompt(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    response_data = {"status": "ok"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(prompt="Extract data", schema_json="{}")

    assert result == response_data


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_empty_content(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test handling of empty API response."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_invalid_json(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test handling of invalid JSON response."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json {{"
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_api_error(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    mock_client.chat.completions.create.side_effect = RuntimeError("API Error")

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.openai.OpenAI")
@patch("docling_graph.llm_clients.openai.get_context_limit")
def test_get_json_response_empty_all_null(mock_get_context_limit, mock_openai_class, mock_env_vars):
    """Test handling of all-null JSON."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_get_context_limit.return_value = 4096

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"a": None, "b": None})
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient(model="gpt-4")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"a": None, "b": None}
