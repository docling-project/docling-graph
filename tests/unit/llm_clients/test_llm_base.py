"""
Tests for LLM base client interface.
"""

from abc import ABC
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from docling_graph.llm_clients.llm_base import BaseLlmClient


class ConcreteClient(BaseLlmClient):
    """Concrete implementation for testing."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._context_limit = 4096

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        return {"result": "test"}

    @property
    def context_limit(self) -> int:
        return self._context_limit


class TestBaseLlmClient:
    """Test BaseLlmClient abstract class."""

    def test_base_client_is_abstract(self):
        """Should not be able to instantiate BaseLlmClient."""
        with pytest.raises(TypeError):
            BaseLlmClient()

    def test_concrete_client_can_be_instantiated(self):
        """Concrete implementation should be instantiable."""
        client = ConcreteClient(model="test-model")
        assert client is not None

    def test_client_requires_model_parameter(self):
        """Client should accept model parameter."""
        client = ConcreteClient(model="gpt-4")
        assert client.model == "gpt-4"

    def test_client_context_limit_property(self):
        """Client should have context_limit property."""
        client = ConcreteClient(model="test-model")
        assert hasattr(client, "context_limit")
        assert isinstance(client.context_limit, int)
        assert client.context_limit > 0

    def test_get_json_response_method(self):
        """Client should have get_json_response method."""
        client = ConcreteClient(model="test-model")
        result = client.get_json_response("prompt", "{}")

        assert isinstance(result, dict)
        assert result.get("result") == "test"

    def test_accepts_string_or_dict_prompt(self):
        """get_json_response should accept string or dict prompt."""
        client = ConcreteClient(model="test-model")

        # String prompt
        result1 = client.get_json_response("test prompt", "{}")
        assert isinstance(result1, dict)

        # Dict prompt
        result2 = client.get_json_response({"system": "sys", "user": "msg"}, "{}")
        assert isinstance(result2, dict)
