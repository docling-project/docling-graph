from unittest.mock import patch

import pytest
from pydantic import ValidationError

from docling_graph.llm_clients.config import (
    LlmRuntimeOverrides,
    ProviderDefinition,
    resolve_effective_model_config,
)


def test_invalid_merge_threshold_rejected():
    with pytest.raises(ValidationError):
        ProviderDefinition(merge_threshold=1.5)


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=128000)
@patch(
    "docling_graph.llm_clients.config._get_litellm_model_info",
    return_value={"max_output_tokens": 4096},
)
def test_defaults_apply_in_effective_config(_info, _max_tokens):
    # Test that defaults are applied when no overrides are provided
    effective = resolve_effective_model_config("openai", "gpt-4o")

    assert effective.context_limit == 128000
    assert effective.max_output_tokens == 4096
    assert effective.generation.temperature == 0.1
    assert effective.generation.max_tokens == 4096
    assert effective.reliability.timeout_s == 300


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_runtime_overrides_take_precedence(_info, _max_tokens):
    overrides = LlmRuntimeOverrides(
        generation={"temperature": 0.3, "max_tokens": 1024},
        reliability={"timeout_s": 10},
        connection={"base_url": "https://proxy.example.com"},
        token_density=2.4,
    )

    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)

    assert effective.generation.temperature == 0.3
    assert effective.generation.max_tokens == 1024
    assert effective.reliability.timeout_s == 10
    assert effective.connection.base_url == "https://proxy.example.com"
    assert effective.token_density == 2.4


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=None)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_context_limit_and_max_output_tokens_overrides(_info, _max_tokens):
    """Test that context_limit and max_output_tokens can be overridden."""
    overrides = LlmRuntimeOverrides(
        context_limit=16384,
        max_output_tokens=4096,
    )

    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)

    assert effective.context_limit == 16384
    assert effective.max_output_tokens == 4096


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=64000)
@patch(
    "docling_graph.llm_clients.config._get_litellm_model_info",
    return_value={"max_output_tokens": 4096},
)
def test_models_yaml_precedence(_info, _max_tokens):
    """Test that model configuration is resolved from LiteLLM metadata."""
    # Since models.yaml was removed, we test that LiteLLM metadata is used
    effective = resolve_effective_model_config("mistral", "mistral-large-latest")
    assert effective.context_limit == 64000
    assert effective.max_output_tokens == 4096


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=None)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_context_limit_and_max_output_tokens_dict_overrides(_info, _max_tokens):
    """Test that context_limit and max_output_tokens can be overridden via dict."""
    overrides = {
        "context_limit": 32768,
        "max_output_tokens": 8192,
    }

    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)

    assert effective.context_limit == 32768
    assert effective.max_output_tokens == 8192
