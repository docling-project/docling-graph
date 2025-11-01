"""
Mock backends for testing without actual LLM/VLM dependencies.
"""

from typing import Any, Dict, List

from pydantic import BaseModel


class MockLLMBackend:
    """Mock LLM backend for testing."""

    def __init__(self, model_name: str = "mock-llm"):
        """Initialize mock LLM backend.

        Args:
            model_name: Name of the mock model.
        """
        self.model_name = model_name
        self.call_count = 0
        self.last_input = None

    def extract(self, text: str, template: type[BaseModel]) -> Dict[str, Any]:
        """Mock extraction from text.

        Args:
            text: Input text to extract from.
            template: Pydantic model class.

        Returns:
            Dictionary matching template structure.
        """
        self.call_count += 1
        self.last_input = text

        # Return mock data based on template fields
        mock_data = {}
        for field_name, field_info in template.model_fields.items():
            # Generate appropriate mock data based on field type
            field_type = field_info.annotation
            if field_type == str:
                mock_data[field_name] = f"mock_{field_name}"
            elif field_type == int:
                mock_data[field_name] = 42
            elif field_type == float:
                mock_data[field_name] = 3.14
            elif field_type == bool:
                mock_data[field_name] = True
            elif field_type == list:
                mock_data[field_name] = []
            else:
                mock_data[field_name] = None

        return mock_data

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_input = None


class MockVLMBackend:
    """Mock VLM (Vision Language Model) backend for testing."""

    def __init__(self, model_name: str = "mock-vlm"):
        """Initialize mock VLM backend.

        Args:
            model_name: Name of the mock model.
        """
        self.model_name = model_name
        self.name = "Mock"  # Add name attribute
        self.call_count = 0
        self.last_image = None

    def extract_from_document(self, doc_path, template):
        """Return proper list, not Mock object.

        Args:
            doc_path: Path to document file.
            template: Pydantic model class.

        Returns:
            List of dictionaries matching template structure.
        """
        self.call_count += 1
        # Return actual list with mock data
        return [
            {"name": "John Doe", "age": 30},
            {"name": "Jane Smith", "age": 25},
        ]  # NOT just Mock()

    def extract_from_images(self, images, template):
        """Return proper list for image extraction.

        Args:
            images: List of image paths.
            template: Pydantic model class.

        Returns:
            List of dictionaries matching template structure.
        """
        return [{"name": "Person1", "age": 40}]

    def extract(self, image_path: str, template: type[BaseModel]) -> List[Dict[str, Any]]:
        """Mock extraction from image.

        Args:
            image_path: Path to image file.
            template: Pydantic model class.

        Returns:
            List of dictionaries matching template structure.
        """
        self.call_count += 1
        self.last_image = image_path

        # Return mock data
        mock_data = {}
        for field_name, field_info in template.model_fields.items():
            field_type = field_info.annotation
            if field_type == str:
                mock_data[field_name] = f"mock_{field_name}"
            elif field_type == int:
                mock_data[field_name] = 99
            elif field_type == float:
                mock_data[field_name] = 2.71
            elif field_type == bool:
                mock_data[field_name] = False
            elif field_type == list:
                mock_data[field_name] = []
            else:
                mock_data[field_name] = None

        # VLM returns list
        return [mock_data]

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_image = None


class MockOllamaClient:
    """Mock Ollama client for testing."""

    def __init__(self, model: str = "llama3:8b"):
        """Initialize mock Ollama client.

        Args:
            model: Model name.
        """
        self.model = model
        self.call_count = 0

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Mock chat completion.

        Args:
            messages: List of chat messages.

        Returns:
            Mock response.
        """
        self.call_count += 1
        return {
            "message": {
                "role": "assistant",
                "content": '{"name": "mock_name", "value": "mock_value"}',
            }
        }

    def extract(self, text: str, template: type[BaseModel]) -> Dict[str, Any]:
        """Mock extraction.

        Args:
            text: Input text.
            template: Pydantic model.

        Returns:
            Mock data.
        """
        return MockLLMBackend().extract(text, template)


class MockMistralClient:
    """Mock Mistral API client for testing."""

    def __init__(self, api_key: str = "mock-key"):
        """Initialize mock Mistral client.

        Args:
            api_key: API key (mock).
        """
        self.api_key = api_key
        self.call_count = 0

    def chat(self, model: str, messages: List[Dict[str, str]]) -> Any:
        """Mock chat completion.

        Args:
            model: Model name.
            messages: Chat messages.

        Returns:
            Mock response.
        """
        self.call_count += 1

        # Mock response object
        class MockResponse:
            class MockChoice:
                class MockMessage:
                    content = '{"field": "value"}'
                    role = "assistant"

                message = MockMessage()

            choices = [MockChoice()]

        return MockResponse()

    def extract(self, text: str, template: type[BaseModel]) -> Dict[str, Any]:
        """Mock extraction.

        Args:
            text: Input text.
            template: Pydantic model.

        Returns:
            Mock data.
        """
        return MockLLMBackend().extract(text, template)


class ConfigurableMockBackend:
    """Configurable mock backend for advanced testing."""

    def __init__(self, responses: List[Dict[str, Any]] = None):
        """Initialize configurable mock.

        Args:
            responses: List of responses to return sequentially.
        """
        self.responses = responses or []
        self.call_index = 0
        self.call_count = 0
        self.calls = []

    def extract(self, text: str, template: type[BaseModel]) -> Dict[str, Any]:
        """Mock extraction with configured responses.

        Args:
            text: Input text.
            template: Pydantic model.

        Returns:
            Next configured response or default mock data.
        """
        self.calls.append({"text": text, "template": template})
        self.call_count += 1

        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        else:
            # Fall back to default mock
            return MockLLMBackend().extract(text, template)

    def reset(self):
        """Reset mock state."""
        self.call_index = 0
        self.call_count = 0
        self.calls = []


class FailingMockBackend:
    """Mock backend that simulates failures."""

    def __init__(self, failure_mode: str = "exception"):
        """Initialize failing mock.

        Args:
            failure_mode: Type of failure ("exception", "timeout", "invalid_data").
        """
        self.failure_mode = failure_mode
        self.call_count = 0

    def extract(self, text: str, template: type[BaseModel]) -> Dict[str, Any]:
        """Mock extraction that fails.

        Args:
            text: Input text.
            template: Pydantic model.

        Raises:
            Various exceptions based on failure_mode.
        """
        self.call_count += 1

        if self.failure_mode == "exception":
            raise Exception("Mock backend error")
        elif self.failure_mode == "timeout":
            import time

            time.sleep(10)  # Simulate timeout
        elif self.failure_mode == "invalid_data":
            return {"invalid": "does not match template"}
        else:
            raise ValueError(f"Unknown failure mode: {self.failure_mode}")


# Convenience functions
def create_mock_llm_backend() -> MockLLMBackend:
    """Create a mock LLM backend.

    Returns:
        Configured MockLLMBackend instance.
    """
    return MockLLMBackend()


def create_mock_vlm_backend() -> MockVLMBackend:
    """Create a mock VLM backend.

    Returns:
        Configured MockVLMBackend instance.
    """
    return MockVLMBackend()


def create_failing_backend(failure_mode: str = "exception") -> FailingMockBackend:
    """Create a failing mock backend.

    Args:
        failure_mode: Type of failure to simulate.

    Returns:
        FailingMockBackend instance.
    """
    return FailingMockBackend(failure_mode=failure_mode)
