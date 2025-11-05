import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.mistral import MistralClient


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_mistral_client_init(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test Mistral client initialization."""
    mock_mistral_class.return_value = MagicMock()
    mock_get_context_limit.return_value = 32768

    client = MistralClient(model="mistral-medium")

    assert client.model == "mistral-medium"
    assert client.api_key == "test-mistral-key"
    assert client.context_limit == 32768
    mock_mistral_class.assert_called_once_with(api_key="test-mistral-key")


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_mistral_client_init_no_api_key(mock_get_context_limit, mock_mistral_class, monkeypatch):
    """Test that missing API key raises error."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY not set"):
        MistralClient(model="mistral-medium")


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_dict_prompt(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    response_data = {"name": "Bob", "role": "engineer"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.complete.return_value = mock_response

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(
        prompt={"system": "Extract data", "user": "Bob is an engineer"}, schema_json="{}"
    )

    assert result == response_data
    mock_client.chat.complete.assert_called_once()

    # Verify messages format
    call_args = mock_client.chat.complete.call_args
    assert call_args[1]["messages"][0]["role"] == "system"
    assert call_args[1]["messages"][1]["role"] == "user"


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_empty_prompt(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test handling of empty prompts."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    response_data = {"fallback": True}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.complete.return_value = mock_response

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(prompt={"system": "", "user": ""}, schema_json="{}")

    assert result == response_data


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_string_prompt(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    response_data = {"result": "success"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    mock_client.chat.complete.return_value = mock_response

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(prompt="Extract info", schema_json="{}")

    assert result == response_data


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_invalid_json(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test handling of invalid JSON."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "invalid {json"
    mock_client.chat.complete.return_value = mock_response

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_empty_content(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test handling of empty response content."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_client.chat.complete.return_value = mock_response

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.mistral.Mistral")
@patch("docling_graph.llm_clients.mistral.get_context_limit")
def test_get_json_response_api_error(mock_get_context_limit, mock_mistral_class, mock_env_vars):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_mistral_class.return_value = mock_client
    mock_get_context_limit.return_value = 32768

    mock_client.chat.complete.side_effect = RuntimeError("API Error")

    client = MistralClient(model="mistral-medium")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}
