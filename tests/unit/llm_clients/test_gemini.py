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
@patch("docling_graph.llm_clients.config.get_model_config")
def test_gemini_client_init(mock_get_model_config, mock_types, mock_genai, mock_env_vars):
    """Test Gemini client initialization."""
    mock_genai.Client.return_value = MagicMock()
    mock_config = MagicMock()
    mock_config.context_limit = 1000000
    mock_get_model_config.return_value = mock_config

    client = GeminiClient(model="gemini-pro")

    assert client.model == "gemini-pro"
    assert client.api_key == "test-gemini-key"
    assert client.context_limit == 1000000
    mock_genai.Client.assert_called_once_with(api_key="test-gemini-key")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_gemini_client_init_no_api_key(mock_get_model_config, mock_types, mock_genai, monkeypatch):
    """Test that missing API key raises ConfigurationError."""
    from docling_graph.exceptions import ConfigurationError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mock_genai.Client.return_value = MagicMock()
    mock_config = MagicMock()
    mock_config.context_limit = 1000000
    mock_get_model_config.return_value = mock_config

    with pytest.raises(ConfigurationError, match="Required environment variable not set"):
        GeminiClient(model="gemini-pro")


@patch("docling_graph.llm_clients.gemini.genai")
@patch("docling_graph.llm_clients.gemini.types")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_get_json_response_dict_prompt(
    mock_get_model_config, mock_types, mock_genai, mock_env_vars
):
    """Test JSON response with dict-style prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_config = MagicMock()
    mock_config.context_limit = 1000000
    mock_get_model_config.return_value = mock_config

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
@patch("docling_graph.llm_clients.config.get_model_config")
def test_get_json_response_string_prompt(
    mock_get_model_config, mock_types, mock_genai, mock_env_vars
):
    """Test JSON response with string prompt."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_config = MagicMock()
    mock_config.context_limit = 1000000
    mock_get_model_config.return_value = mock_config

    response_data = {"status": "ok"}
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_data)
    mock_client.models.generate_content.return_value = mock_response

    mock_types.GenerateContentConfig.return_value = MagicMock()

    client = GeminiClient(model="gemini-pro")
    result = client.get_json_response(prompt="Extract", schema_json="{}")

    assert result == response_data


# NOTE: The following tests were removed as they tested obsolete behavior:
# - test_get_json_response_list_result: Lists are now returned as-is, not wrapped
# - test_get_json_response_scalar_result: Scalar normalization tested in response_handler
# - test_get_json_response_invalid_json: Now raises ClientError instead of returning {}
# - test_get_json_response_api_error: Now raises exceptions instead of returning {}
# These behaviors are properly tested in tests/unit/llm_clients/test_response_handler.py

# Made with Bob
