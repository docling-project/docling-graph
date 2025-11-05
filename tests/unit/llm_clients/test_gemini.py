import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.gemini import GeminiClient


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_gemini_client_init(mock_get_context_limit, mock_types, mock_genai, mock_env_vars):
    """Test Gemini client initialization."""
    mock_genai.Client.return_value = MagicMock()
    mock_get_context_limit.return_value = 1000000

    client = GeminiClient(model="gemini-pro")

    assert client.model == "gemini-pro"
    assert client.api_key == "test-gemini-key"
    assert client.context_limit == 1000000
    mock_genai.Client.assert_called_once_with(api_key="test-gemini-key")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_gemini_client_init_no_api_key(mock_get_context_limit, mock_types, mock_genai, monkeypatch):
    """Test that missing API key raises error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
        GeminiClient(model="gemini-pro")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_dict_prompt(
    mock_get_context_limit, mock_types, mock_genai, mock_env_vars
):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    response_data = {"extracted": "data", "value": 42}
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(
        prompt={"system": "Extract info", "user": "Process this"}, schema_json="{}"
    )

    assert result == response_data
    mock_client.models.generate_content.assert_called_once()


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_string_prompt(
    mock_get_context_limit, mock_types, mock_genai, mock_env_vars
):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    response_data = {"status": "ok"}
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="Extract", schema_json="{}")

    assert result == response_data


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_list_result(
    mock_get_context_limit, mock_types, mock_genai, mock_env_vars
):
    """Test normalization of list response to dict."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    mock_response = MagicMock()
    mock_response.text = json.dumps([1, 2, 3])
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"result": [1, 2, 3]}


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_scalar_result(
    mock_get_context_limit, mock_types, mock_genai, mock_env_vars
):
    """Test normalization of scalar response to dict."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    mock_response = MagicMock()
    mock_response.text = json.dumps("simple string")
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {"value": "simple string"}


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_invalid_json(
    mock_get_context_limit, mock_types, mock_genai, mock_env_vars
):
    """Test handling of invalid JSON."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    mock_response = MagicMock()
    mock_response.text = "not json {{"
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.gemini.get_context_limit")
def test_get_json_response_api_error(mock_get_context_limit, mock_types, mock_genai, mock_env_vars):
    """Test handling of API errors."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_get_context_limit.return_value = 1000000

    mock_client.models.generate_content.side_effect = RuntimeError("API Error")

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="test", schema_json="{}")

    assert result == {}
