"""
Shared test constants and fixtures for LLM clients.
"""

import types as py_types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ==================== Context Limit Constants ====================

OPENAI_TEST_MODELS = [
    ("gpt-4-turbo", 128000),
    ("gpt-4", 8192),
    ("gpt-3.5-turbo", 16000),
]

MISTRAL_TEST_MODELS = [
    ("mistral-large-latest", 128000),
    ("mistral-medium-latest", 128000),
    ("mistral-small-latest", 32000),
]

OLLAMA_TEST_MODELS = [
    ("llama3.1:8b", 128000),
    ("llama3.2:3b", 128000),
    ("mixtral:8x7b", 32000),
]

VLLM_TEST_MODELS = [
    ("meta-llama/Llama-3.1-8B", 128000),
    ("mistralai/Mixtral-8x7B-v0.1", 32000),
]

GEMINI_TEST_MODELS = [
    ("gemini-1.5-pro", 1000000),
    ("gemini-1.5-flash", 1000000),
]

# ==================== Shared Fixtures ====================


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client."""
    with patch("docling_graph.llm_clients.openai.OpenAI") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_ollama_module():
    """Mock Ollama module."""
    with patch("docling_graph.llm_clients.ollama.ollama") as mock:
        mock.show = MagicMock(return_value={"name": "test-model"})
        yield mock


@pytest.fixture
def mock_mistral_api():
    """Mock Mistral API module."""
    with patch("docling_graph.llm_clients.mistral.Mistral") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gemini():
    """Mock Gemini API and its types module."""
    with (
        patch("docling_graph.llm_clients.gemini.genai") as mock_genai,
        patch("docling_graph.llm_clients.gemini.types") as mock_types,
    ):
        # Minimal config object with attribute access for assertions
        def make_config(**kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        mock_types.GenerateContentConfig = lambda **kwargs: make_config(**kwargs)

        # Provide a Client() with .models.generate_content
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content = MagicMock()

        yield mock_genai


@pytest.fixture
def mock_vllm_openai_client():
    """Mock OpenAI client for vLLM."""
    with patch("docling_graph.llm_clients.vllm.OpenAI") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        mock_instance.models.list.return_value = MagicMock()
        yield mock_instance
