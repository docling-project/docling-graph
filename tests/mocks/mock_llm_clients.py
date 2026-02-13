"""
Mock LLM clients for testing extraction.
"""

from typing import Any
from unittest.mock import MagicMock


class MockLLMClient:
    """Mock LLM client with configurable responses."""

    def __init__(self) -> None:
        self.model = "mock-model"
        self.context_limit = 4096
        self.responses = []
        self.call_count = 0

    def get_json_response(
        self,
        prompt: dict | str,
        schema_json: str,
        structured_output: bool = True,
        response_top_level: str = "object",
        response_schema_name: str = "extraction_result",
    ) -> dict | None:
        """Return mock JSON response."""
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return None

    def set_responses(self, responses: list[dict]):
        """Set mock responses."""
        self.responses = responses.copy()

    def reset(self):
        """Reset client state."""
        self.responses = []
        self.call_count = 0


def create_mock_llm_client():
    """Factory function to create a mock LLM client."""
    return MockLLMClient()
