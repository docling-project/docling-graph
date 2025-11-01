"""
Mock API clients for testing without actual API calls.
"""

from typing import Any, Dict, List, Optional


class MockAPIClient:
    """Base mock API client."""

    def __init__(self, api_key: str = "mock-api-key"):
        """Initialize mock API client.

        Args:
            api_key: API key (mock).
        """
        self.api_key = api_key
        self.call_count = 0
        self.last_request = None

    def make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock API request.

        Args:
            endpoint: API endpoint.
            data: Request data.

        Returns:
            Mock response.
        """
        self.call_count += 1
        self.last_request = {"endpoint": endpoint, "data": data}

        return {"status": "success", "data": {"result": "mock_result"}}


class MockOpenAIClient:
    """Mock OpenAI API client."""

    def __init__(self, api_key: str = "mock-openai-key"):
        """Initialize mock OpenAI client.

        Args:
            api_key: API key (mock).
        """
        self.api_key = api_key
        self.call_count = 0

    def chat_completion(self, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Mock chat completion.

        Args:
            model: Model name.
            messages: Chat messages.

        Returns:
            Mock completion response.
        """
        self.call_count += 1

        return {"choices": [{"message": {"role": "assistant", "content": '{"extracted": "data"}'}}]}


class MockGeminiClient:
    """Mock Google Gemini API client."""

    def __init__(self, api_key: str = "mock-gemini-key"):
        """Initialize mock Gemini client.

        Args:
            api_key: API key (mock).
        """
        self.api_key = api_key
        self.call_count = 0

    def generate_content(self, prompt: str) -> Any:
        """Mock content generation.

        Args:
            prompt: Input prompt.

        Returns:
            Mock response.
        """
        self.call_count += 1

        class MockResponse:
            text = '{"field": "value"}'

        return MockResponse()


class RateLimitedMockClient:
    """Mock client that simulates rate limiting."""

    def __init__(self, rate_limit: int = 3):
        """Initialize rate-limited mock client.

        Args:
            rate_limit: Number of calls before rate limit error.
        """
        self.rate_limit = rate_limit
        self.call_count = 0

    def make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock request with rate limiting.

        Args:
            data: Request data.

        Returns:
            Mock response or raises rate limit error.
        """
        self.call_count += 1

        if self.call_count > self.rate_limit:
            raise Exception("Rate limit exceeded")

        return {"status": "success"}

    def reset(self):
        """Reset call counter."""
        self.call_count = 0
