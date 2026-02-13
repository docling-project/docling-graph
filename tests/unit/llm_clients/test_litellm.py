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
