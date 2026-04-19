import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from docling_graph.llm_clients.config import (
    LlmRuntimeOverrides,
    ProviderDefinition,
    build_litellm_model_name,
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


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_connection_api_key_override(_info, _max_tokens):
    """Override api_key via ConnectionOverrides is used in resolved config."""
    overrides = LlmRuntimeOverrides(
        connection={"api_key": SecretStr("custom-key-from-config")},
    )
    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.connection.api_key is not None
    assert effective.connection.api_key.get_secret_value() == "custom-key-from-config"


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_fixed_custom_env_resolution_for_openai(_info, _max_tokens):
    """Fixed CUSTOM_LLM_* env vars are resolved for openai provider."""
    with patch.dict(
        os.environ,
        {
            "CUSTOM_LLM_API_KEY": "env-api-key",
            "CUSTOM_LLM_BASE_URL": "https://onprem.example.com/v1",
        },
        clear=False,
    ):
        effective = resolve_effective_model_config("openai", "gpt-4o")
    assert effective.connection.api_key is not None
    assert effective.connection.api_key.get_secret_value() == "env-api-key"
    assert effective.connection.base_url == "https://onprem.example.com/v1"


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_connection_explicit_override_beats_env(_info, _max_tokens):
    """Explicit connection.api_key and base_url override env-based values."""
    overrides = LlmRuntimeOverrides(
        connection={
            "api_key": SecretStr("explicit-key"),
            "base_url": "https://explicit.example.com/v1",
        },
    )
    with patch.dict(
        os.environ,
        {"CUSTOM_LLM_API_KEY": "env-key", "CUSTOM_LLM_BASE_URL": "https://env.example.com/v1"},
        clear=False,
    ):
        effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.connection.api_key.get_secret_value() == "explicit-key"
    assert effective.connection.base_url == "https://explicit.example.com/v1"


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_connection_overrides_backward_compat(_info, _max_tokens):
    """Only base_url/api_key (no env names) still works."""
    overrides = LlmRuntimeOverrides(
        connection={"base_url": "https://legacy.example.com"},
    )
    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.connection.base_url == "https://legacy.example.com"


def test_build_litellm_model_name_lmstudio():
    """LM Studio provider produces lm_studio/<model_id> for LiteLLM."""
    assert build_litellm_model_name("lmstudio", "my-model", None) == "lm_studio/my-model"
    assert build_litellm_model_name("lmstudio", "llama-3.2-3b", None) == "lm_studio/llama-3.2-3b"


def test_build_litellm_model_name_lmstudio_strips_prefix():
    """LM Studio model_id with existing lm_studio/ prefix is normalized."""
    assert build_litellm_model_name("lmstudio", "lm_studio/my-model", None) == "lm_studio/my-model"


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_resolve_effective_model_config_lmstudio_produces_litellm_model(_info, _max_tokens):
    """Resolving lmstudio provider yields litellm_model with lm_studio/ prefix."""
    effective = resolve_effective_model_config("lmstudio", "my-local-model")
    assert effective.litellm_model == "lm_studio/my-local-model"
    assert effective.provider_id == "lmstudio"


# ============================================================================
# Streaming Configuration Tests
# ============================================================================


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_streaming_config_default(_info, _max_tokens):
    """Test that streaming defaults to False."""
    effective = resolve_effective_model_config("openai", "gpt-4o")
    assert effective.streaming is False


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_streaming_config_override(_info, _max_tokens):
    """Test that streaming can be overridden."""
    overrides = LlmRuntimeOverrides(streaming=True)
    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.streaming is True


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_streaming_config_in_effective_config(_info, _max_tokens):
    """Test that streaming appears in EffectiveModelConfig."""
    # Test with streaming=False (default)
    effective_default = resolve_effective_model_config("openai", "gpt-4o")
    assert hasattr(effective_default, "streaming")
    assert effective_default.streaming is False

    # Test with streaming=True
    overrides = LlmRuntimeOverrides(streaming=True)
    effective_enabled = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective_enabled.streaming is True


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_streaming_config_via_dict_override(_info, _max_tokens):
    """Test that streaming can be overridden via dict."""
    overrides = {"streaming": True}
    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.streaming is True


@patch("docling_graph.llm_clients.config._get_litellm_max_tokens", return_value=8192)
@patch("docling_graph.llm_clients.config._get_litellm_model_info", return_value=None)
def test_streaming_config_with_other_overrides(_info, _max_tokens):
    """Test that streaming works alongside other overrides."""
    overrides = LlmRuntimeOverrides(
        streaming=True,
        generation={"temperature": 0.5},
        reliability={"timeout_s": 60},
    )
    effective = resolve_effective_model_config("openai", "gpt-4o", overrides=overrides)
    assert effective.streaming is True
    assert effective.generation.temperature == 0.5
    assert effective.reliability.timeout_s == 60
