from unittest.mock import patch

import pytest
from pydantic import SecretStr

from docling_graph.exceptions import ClientError
from docling_graph.llm_clients.config import (
    EffectiveModelConfig,
    GenerationDefaults,
    ReliabilityDefaults,
    ResolvedConnection,
)
from docling_graph.llm_clients.litellm import LiteLLMClient


def _make_effective_config() -> EffectiveModelConfig:
    """Create a test EffectiveModelConfig without capability field (removed)."""
    return EffectiveModelConfig(
        model_id="mistral-large-latest",
        provider_id="mistral",
        litellm_model="mistral/mistral-large-latest",
        context_limit=128000,
        max_output_tokens=4096,
        # Note: capability field removed
        generation=GenerationDefaults(max_tokens=512, temperature=0.1),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key=SecretStr("test-mistral-key")),
        tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
        merge_threshold=0.95,
    )


def _make_vllm_effective_config() -> EffectiveModelConfig:
    return EffectiveModelConfig(
        model_id="ibm-granite/granite-4.0-1b",
        provider_id="vllm",
        litellm_model="vllm/ibm-granite/granite-4.0-1b",
        context_limit=32768,
        max_output_tokens=4096,
        generation=GenerationDefaults(max_tokens=512, temperature=0.1),
        reliability=ReliabilityDefaults(timeout_s=30, max_retries=0),
        connection=ResolvedConnection(api_key=SecretStr("test-vllm-key")),
        tokenizer="dummy",
        merge_threshold=0.95,
    )


@patch("docling_graph.llm_clients.litellm.litellm")
def test_litellm_client_builds_request(mock_litellm):
    mock_litellm.get_supported_openai_params.return_value = [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "response_format",
        "timeout",
        "drop_params",
        "api_key",
    ]
    mock_litellm.completion.return_value = {
        "model": "mistral/mistral-large-latest",
        "choices": [
            {
                "message": {"content": '{"ok": true}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 12},
    }

    client = LiteLLMClient(model_config=_make_effective_config())
    result = client.get_json_response(prompt="Extract", schema_json="{}")

    assert result == {"ok": True}
    mock_litellm.completion.assert_called_once()
    request = mock_litellm.completion.call_args.kwargs
    assert request["model"] == "mistral/mistral-large-latest"
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["drop_params"] is True


@patch("docling_graph.llm_clients.litellm.litellm")
def test_litellm_client_supports_legacy_json_object_mode(mock_litellm):
    mock_litellm.get_supported_openai_params.return_value = [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "response_format",
        "timeout",
        "drop_params",
        "api_key",
    ]
    mock_litellm.completion.return_value = {
        "model": "mistral/mistral-large-latest",
        "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 12},
    }
    client = LiteLLMClient(model_config=_make_effective_config())
    result = client.get_json_response(prompt="Extract", schema_json="{}", structured_output=False)
    assert result == {"ok": True}
    request = mock_litellm.completion.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}


@patch("docling_graph.llm_clients.litellm.litellm")
def test_litellm_client_empty_choices_raises(mock_litellm):
    mock_litellm.get_supported_openai_params.return_value = []
    mock_litellm.completion.return_value = {"choices": []}

    client = LiteLLMClient(model_config=_make_effective_config())
    with pytest.raises(ClientError, match="no choices"):
        client.get_json_response(prompt="Extract", schema_json="{}")


@patch("docling_graph.llm_clients.litellm.litellm")
def test_litellm_structured_attempt_applies_to_vllm(mock_litellm):
    mock_litellm.get_supported_openai_params.return_value = [
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "response_format",
        "timeout",
        "drop_params",
        "api_key",
    ]
    mock_litellm.completion.return_value = {
        "model": "vllm/ibm-granite/granite-4.0-1b",
        "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 12},
    }
    client = LiteLLMClient(model_config=_make_vllm_effective_config())
    client.get_json_response(prompt="Extract", schema_json="{}")
    request = mock_litellm.completion.call_args.kwargs
    assert request["response_format"]["type"] == "json_schema"


@patch("docling_graph.llm_clients.litellm.litellm")
def test_litellm_records_structured_failure_diagnostics(mock_litellm):
    mock_litellm.get_supported_openai_params.return_value = []
    mock_litellm.completion.side_effect = RuntimeError("unsupported response_format")
    client = LiteLLMClient(model_config=_make_vllm_effective_config())
    with pytest.raises(ClientError):
        client.get_json_response(prompt="Extract", schema_json="{}", structured_output=True)
    assert client.last_call_diagnostics["structured_attempted"] is True
    assert client.last_call_diagnostics["provider"] == "vllm"


# ============================================================================
# Streaming Tests
# ============================================================================


class MockStreamPayload:
    """Mock payload object for streaming chunks."""

    def __init__(self, content: str | None) -> None:
        self.content = content


class MockStreamChoice:
    """Mock choice object for streaming chunks."""

    def __init__(self, content: str | None = None, finish_reason: str | None = None) -> None:
        self.delta = MockStreamPayload(content)
        self.finish_reason = finish_reason


class MockStreamChunk:
    """Mock streaming chunk from litellm."""

    def __init__(
        self, content: str | None = None, finish_reason: str | None = None, model: str | None = None
    ) -> None:
        self.choices = [MockStreamChoice(content, finish_reason)]
        self.model = model


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_successful_response(mock_litellm):
    """Test successful streaming response with single chunk."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(
                content='{"result": "success"}', finish_reason=None, model="test-model"
            ),
            MockStreamChunk(content=None, finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))

    assert len(results) == 1
    assert results[0] == {"result": "success"}
    assert client.last_call_diagnostics["streaming"] is True
    assert client.last_call_diagnostics["finish_reason"] == "stop"
    assert client.last_call_diagnostics["model"] == "test-model"


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_multiple_chunks(mock_litellm):
    """Test streaming with multiple content chunks that need to be accumulated."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response with multiple chunks
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"items": [', model="test-model"),
            MockStreamChunk(content='"item1", '),
            MockStreamChunk(content='"item2", '),
            MockStreamChunk(content='"item3"'),
            MockStreamChunk(content="]}"),
            MockStreamChunk(content=None, finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))

    assert len(results) == 1
    assert results[0] == {"items": ["item1", "item2", "item3"]}
    assert client.last_call_diagnostics["streaming"] is True
    assert client.last_call_diagnostics["raw_response"] == '{"items": ["item1", "item2", "item3"]}'


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_empty_content_raises_error(mock_litellm):
    """Test that streaming with no content raises ClientError."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response with no content
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content=None, finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())

    with pytest.raises(ClientError, match="No content received from streaming response"):
        list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_error_handling(mock_litellm):
    """Test that streaming errors are properly caught and wrapped."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response that raises an exception
    mock_litellm.completion.side_effect = RuntimeError("Connection error")

    client = LiteLLMClient(model_config=_make_effective_config())

    # With structured_output=False, should get generic streaming error
    with pytest.raises(ClientError, match="Streaming API call failed"):
        list(
            client.get_json_response_stream(
                prompt="Extract", schema_json="{}", structured_output=False
            )
        )


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_structured_output_error(mock_litellm):
    """Test that structured output errors in streaming provide helpful message."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response that raises an exception
    mock_litellm.completion.side_effect = RuntimeError("response_format not supported")

    client = LiteLLMClient(model_config=_make_effective_config())

    with pytest.raises(
        ClientError,
        match=r"Streaming structured output request failed.*Disable structured output",
    ):
        list(
            client.get_json_response_stream(
                prompt="Extract", schema_json="{}", structured_output=True
            )
        )


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_with_structured_output_enabled(mock_litellm):
    """Test streaming with structured_output=True includes schema in request."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"name": "test"}', finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'

    results = list(
        client.get_json_response_stream(
            prompt="Extract", schema_json=schema, structured_output=True
        )
    )

    assert len(results) == 1
    assert results[0] == {"name": "test"}

    # Verify request includes response_format with json_schema
    request = mock_litellm.completion.call_args.kwargs
    assert request["stream"] is True
    assert request["response_format"]["type"] == "json_schema"
    assert "json_schema" in request["response_format"]


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_diagnostics_complete(mock_litellm):
    """Test that streaming updates diagnostics with all required fields."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"data": "value"}', model="gpt-4"),
            MockStreamChunk(content=None, finish_reason="stop", model="gpt-4"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))

    assert len(results) == 1

    # Verify all diagnostic fields are present
    diagnostics = client.last_call_diagnostics
    assert diagnostics["streaming"] is True
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["model"] == "gpt-4"
    assert diagnostics["provider"] == "mistral"
    assert diagnostics["raw_response"] == '{"data": "value"}'
    assert diagnostics["parsed_json"] == {"data": "value"}


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_truncation_detection(mock_litellm):
    """Test that streaming detects truncation via finish_reason=length."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response with length finish_reason
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"partial": "data"}', model="test-model"),
            MockStreamChunk(content=None, finish_reason="length", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))

    assert len(results) == 1
    assert client.last_call_diagnostics["finish_reason"] == "length"


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_with_system_and_user_messages(mock_litellm):
    """Test streaming with both system and user messages."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"ok": true}', finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    prompt = {"system": "You are a helpful assistant", "user": "Extract data"}

    results = list(client.get_json_response_stream(prompt=prompt, schema_json="{}"))

    assert len(results) == 1
    assert results[0] == {"ok": True}

    # Verify messages were properly formatted
    request = mock_litellm.completion.call_args.kwargs
    messages = request["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Extract data"


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_with_array_response(mock_litellm):
    """Test streaming with array as top-level response."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response with array
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(
                content='[{"id": 1}, {"id": 2}]', finish_reason="stop", model="test-model"
            ),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(
        client.get_json_response_stream(
            prompt="Extract", schema_json="{}", response_top_level="array"
        )
    )

    assert len(results) == 1
    assert results[0] == [{"id": 1}, {"id": 2}]
    assert isinstance(results[0], list)


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_without_structured_output(mock_litellm):
    """Test streaming with structured_output=False uses json_object mode."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"result": "ok"}', finish_reason="stop", model="test-model"),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(
        client.get_json_response_stream(prompt="Extract", schema_json="{}", structured_output=False)
    )

    assert len(results) == 1
    assert results[0] == {"result": "ok"}

    # Verify request uses json_object mode
    request = mock_litellm.completion.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}


@patch("docling_graph.llm_clients.litellm.litellm")
def test_streaming_model_fallback_in_diagnostics(mock_litellm):
    """Test that diagnostics use configured model if chunk doesn't provide one."""
    mock_litellm.get_supported_openai_params.return_value = []

    # Mock streaming response without model in chunks
    mock_litellm.completion.return_value = iter(
        [
            MockStreamChunk(content='{"data": "test"}', finish_reason="stop", model=None),
        ]
    )

    client = LiteLLMClient(model_config=_make_effective_config())
    results = list(client.get_json_response_stream(prompt="Extract", schema_json="{}"))

    assert len(results) == 1
    # Should fall back to configured model
    assert client.last_call_diagnostics["model"] == "mistral-large-latest"


# ============================================================================
# Streaming Configuration Tests
# ============================================================================


def test_litellm_client_streaming_flag():
    """Test that LiteLLMClient stores streaming configuration."""
    config = _make_effective_config()
    config.streaming = True

    client = LiteLLMClient(model_config=config)

    assert client.streaming is True


def test_litellm_client_streaming_default():
    """Test that streaming defaults to False in client."""
    config = _make_effective_config()
    # streaming should default to False in config
    assert config.streaming is False

    client = LiteLLMClient(model_config=config)

    assert client.streaming is False


def test_litellm_client_streaming_from_config():
    """Test that client respects streaming flag from config."""
    # Test with streaming=False
    config_no_stream = _make_effective_config()
    config_no_stream.streaming = False
    client_no_stream = LiteLLMClient(model_config=config_no_stream)
    assert client_no_stream.streaming is False

    # Test with streaming=True
    config_stream = _make_effective_config()
    config_stream.streaming = True
    client_stream = LiteLLMClient(model_config=config_stream)
    assert client_stream.streaming is True
