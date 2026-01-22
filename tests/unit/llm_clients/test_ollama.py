import json
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.llm_clients.ollama import OllamaClient


@patch("docling_graph.llm_clients.ollama.ollama")
@patch("docling_graph.llm_clients.config.get_model_config")
def test_ollama_client_init(mock_get_model_config, mock_ollama, monkeypatch):
    """Test Ollama client initialization."""
    mock_ollama.show.return_value = {"name": "llama2"}
    mock_config = MagicMock()

    mock_config.context_limit = 4096

    mock_get_model_config.return_value = mock_config

    client = OllamaClient(model="llama2")

    assert client.model == "llama2"
    assert client.context_limit == 4096
    mock_ollama.show.assert_called_once_with("llama2")
