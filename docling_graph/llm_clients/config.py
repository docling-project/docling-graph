"""
Lightweight LLM configuration registry and resolver.

This module keeps only provider infrastructure settings that LiteLLM does not
provide (tokenizer and merge_threshold), plus connection defaults/overrides.
Model limits and capabilities are resolved dynamically via LiteLLM metadata.
"""

from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from rich import print as rich_print
from typing_extensions import Self

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Load .env for provider credentials
load_dotenv()


class GenerationDefaults(BaseModel):
    """Default generation parameters."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.1
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    min_tokens: int | None = None
    repetition_penalty: float | None = None
    decoding_method: str | None = None


class GenerationOverrides(BaseModel):
    """Overrides for generation parameters (None means 'no override')."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    min_tokens: int | None = None
    repetition_penalty: float | None = None
    decoding_method: str | None = None


class BackoffDefaults(BaseModel):
    """Retry backoff configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_s: float = 1.0
    max_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.1


class BackoffOverrides(BaseModel):
    """Override retry backoff configuration."""

    model_config = ConfigDict(extra="forbid")

    initial_s: float | None = None
    max_s: float | None = None
    multiplier: float | None = None
    jitter: float | None = None


class ReliabilityDefaults(BaseModel):
    """Default reliability parameters."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: int = 300
    max_retries: int = 2
    backoff: BackoffDefaults = Field(default_factory=BackoffDefaults)


class ReliabilityOverrides(BaseModel):
    """Overrides for reliability parameters."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: int | None = None
    max_retries: int | None = None
    backoff: BackoffOverrides | None = None


class ProviderConnection(BaseModel):
    """Provider connection and auth settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    organization: str | None = None
    project_id: str | None = None
    project_id_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class ConnectionOverrides(BaseModel):
    """Runtime connection overrides."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project_id: str | None = None
    headers: dict[str, str] | None = None


class ProviderDefinition(BaseModel):
    """Provider configuration with infrastructure defaults."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = ""
    requires_api_key: bool = False
    requires_project_id: bool = False
    connection: ProviderConnection = Field(default_factory=ProviderConnection)
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    merge_threshold: float = 0.95

    @field_validator("merge_threshold")
    @classmethod
    def _validate_ratio(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("Must be between 0 (exclusive) and 1 (inclusive).")
        return value


class ProviderRegistry(BaseModel):
    """Registry of provider definitions only (no static models)."""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderDefinition]

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        normalized: dict[str, ProviderDefinition] = {}
        for provider_id, provider in self.providers.items():
            provider.provider_id = provider_id.lower()
            normalized[provider_id.lower()] = provider
        self.providers = normalized
        return self

    def get_provider(self, provider_id: str) -> ProviderDefinition | None:
        return self.providers.get(provider_id.lower())


class ResolvedConnection(BaseModel):
    """Connection values after env + override resolution."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    project_id: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)


class EffectiveModelConfig(BaseModel):
    """Fully-resolved runtime configuration for a model.

    No capability tier, no context_limit.
    LiteLLM handles token limits dynamically.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    provider_id: str
    litellm_model: str
    context_limit: int  # Keep for backward compat, but use LiteLLM's value
    max_output_tokens: int  # Keep for backward compat, but use LiteLLM's value
    generation: GenerationDefaults
    reliability: ReliabilityDefaults
    connection: ResolvedConnection
    tokenizer: str
    merge_threshold: float
    token_density: float | None = None


class LlmRuntimeOverrides(BaseModel):
    """Runtime overrides for a resolved model configuration."""

    model_config = ConfigDict(extra="forbid")

    generation: GenerationOverrides = Field(default_factory=GenerationOverrides)
    reliability: ReliabilityOverrides = Field(default_factory=ReliabilityOverrides)
    connection: ConnectionOverrides = Field(default_factory=ConnectionOverrides)
    context_limit: int | None = None
    max_output_tokens: int | None = None
    token_density: float | None = None


_DEFAULT_CONTEXT_LIMIT = 32000
_DEFAULT_MAX_OUTPUT_TOKENS = 4092

# Track which models have already been warned to avoid duplicate warnings
_warned_models: set[str] = set()


def _build_default_registry() -> ProviderRegistry:
    return ProviderRegistry(
        providers={
            "mistral": ProviderDefinition(
                requires_api_key=True,
                connection=ProviderConnection(api_key_env="MISTRAL_API_KEY"),
                tokenizer="mistralai/Mistral-7B-Instruct-v0.2",
                merge_threshold=0.95,
            ),
            "openai": ProviderDefinition(
                requires_api_key=True,
                connection=ProviderConnection(api_key_env="OPENAI_API_KEY"),
                tokenizer="tiktoken",
                merge_threshold=0.95,
            ),
            "gemini": ProviderDefinition(
                requires_api_key=True,
                connection=ProviderConnection(api_key_env="GEMINI_API_KEY"),
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                merge_threshold=0.95,
            ),
            "watsonx": ProviderDefinition(
                requires_api_key=True,
                requires_project_id=True,
                connection=ProviderConnection(
                    api_key_env="WATSONX_API_KEY",
                    project_id_env="WATSONX_PROJECT_ID",
                    base_url="https://us-south.ml.cloud.ibm.com",
                    base_url_env="WATSONX_URL",
                ),
                tokenizer="ibm-granite/granite-embedding-278m-multilingual",
                merge_threshold=0.95,
            ),
            "vllm": ProviderDefinition(
                requires_api_key=False,
                connection=ProviderConnection(
                    base_url="http://localhost:8000/v1",
                    base_url_env="VLLM_BASE_URL",
                    api_key=SecretStr("EMPTY"),
                ),
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                merge_threshold=0.95,
            ),
            "ollama": ProviderDefinition(
                requires_api_key=False,
                connection=ProviderConnection(
                    base_url="http://localhost:11434",
                    base_url_env="OLLAMA_BASE_URL",
                ),
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                merge_threshold=0.95,
            ),
        }
    )


_registry: ProviderRegistry | None = None


def set_registry(registry: ProviderRegistry) -> None:
    global _registry
    _registry = registry


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


def _get_litellm() -> Any | None:
    try:
        import litellm
    except Exception:
        return None
    return litellm


def _get_litellm_model_info(model_name: str | None) -> dict[str, Any] | None:
    if not model_name:
        return None
    litellm = _get_litellm()
    if not litellm:
        return None
    get_info = getattr(litellm, "get_model_info", None)
    if callable(get_info):
        try:
            info = get_info(model_name)
            return info if isinstance(info, dict) else None
        except Exception:
            return None
    return None


def _get_litellm_max_tokens(model_name: str | None) -> int | None:
    if not model_name:
        return None
    litellm = _get_litellm()
    if not litellm:
        return None
    get_max = getattr(litellm, "get_max_tokens", None)
    if callable(get_max):
        try:
            value = get_max(model_name)
            return int(value) if value else None
        except Exception:
            return None
    return None


def build_litellm_model_name(
    provider_id: str, model_id: str, connection: ResolvedConnection | None = None
) -> str:
    model_name = model_id
    provider_id = provider_id.lower()
    base_url = connection.base_url if connection else None

    if provider_id == "vllm" and base_url:
        if model_name.startswith("vllm/"):
            model_name = model_name.removeprefix("vllm/")
        if model_name.startswith("hosted_vllm/"):
            model_name = model_name.removeprefix("hosted_vllm/")
        model_name = f"hosted_vllm/{model_name}"
    elif provider_id not in {"openai"} and not model_name.startswith(f"{provider_id}/"):
        model_name = f"{provider_id}/{model_name}"
    return model_name


def _merge_generation(base: GenerationDefaults, runtime: GenerationOverrides) -> GenerationDefaults:
    data = base.model_dump()
    data.update(runtime.model_dump(exclude_none=True))
    return GenerationDefaults(**data)


def _merge_reliability(
    base: ReliabilityDefaults, runtime: ReliabilityOverrides
) -> ReliabilityDefaults:
    data = base.model_dump()
    data.update(runtime.model_dump(exclude_none=True))

    if runtime.backoff:
        backoff_data = base.backoff.model_dump()
        backoff_data.update(runtime.backoff.model_dump(exclude_none=True))
        data["backoff"] = BackoffDefaults(**backoff_data)

    return ReliabilityDefaults(**data)


def _resolve_connection(
    provider: ProviderDefinition, overrides: ConnectionOverrides | None, provider_id: str
) -> ResolvedConnection:
    connection = provider.connection

    def _env_value(key: str | None) -> str | None:
        if not key:
            return None
        value = os.getenv(key)
        return value or None

    env_api_key = _env_value(connection.api_key_env)
    api_key = connection.api_key or (SecretStr(env_api_key) if env_api_key is not None else None)
    base_url = connection.base_url or _env_value(connection.base_url_env)
    organization = connection.organization
    project_id = connection.project_id or _env_value(connection.project_id_env)
    headers = dict(connection.headers)

    if overrides:
        if overrides.api_key is not None:
            api_key = overrides.api_key
        if overrides.base_url is not None:
            base_url = overrides.base_url
        if overrides.organization is not None:
            organization = overrides.organization
        if overrides.project_id is not None:
            project_id = overrides.project_id
        if overrides.headers:
            headers.update(overrides.headers)

    if provider.requires_api_key and not api_key:
        # In library and test contexts we allow configuration to be
        # constructed without credentials; actual API calls will still
        # fail fast if the key is missing.
        logger.warning(
            "Missing required API key for provider '%s' (env hint: %s); "
            "proceeding with empty credentials.",
            provider_id,
            connection.api_key_env,
        )
    if provider.requires_project_id and not project_id:
        logger.warning(
            "Missing required project ID for provider '%s' (env hint: %s); "
            "proceeding with empty project configuration.",
            provider_id,
            connection.project_id_env,
        )

    return ResolvedConnection(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project_id=project_id,
        headers=headers,
    )


def _resolve_context_limit(litellm_model: str, override: int | None = None) -> int:
    """
    Resolve context limit for a model.

    Context limits are not critical for extraction (direct single-call extraction).
    These values
    are kept for backward compatibility but warnings are removed as they're not actionable
    in the new architecture.

    Args:
        litellm_model: LiteLLM model identifier
        override: Optional override value

    Returns:
        Context limit in tokens
    """
    if override is not None:
        return override
    context_limit = _get_litellm_max_tokens(litellm_model)
    if context_limit:
        return int(context_limit)
    # Silently fall back to default - not critical for extraction
    return _DEFAULT_CONTEXT_LIMIT


def _resolve_max_output_tokens(litellm_model: str, override: int | None = None) -> int:
    """
    Resolve max output tokens for a model.

    Max output tokens are not critical for extraction (direct single-call extraction).
    These values
    are kept for backward compatibility but warnings are removed as they're not actionable
    in the new architecture.

    Args:
        litellm_model: LiteLLM model identifier
        override: Optional override value

    Returns:
        Max output tokens
    """
    if override is not None:
        return override
    info = _get_litellm_model_info(litellm_model)
    if info and info.get("max_output_tokens"):
        return int(info["max_output_tokens"])
    # Silently fall back to default - not critical for extraction
    return _DEFAULT_MAX_OUTPUT_TOKENS


def get_provider_config(provider_id: str) -> ProviderDefinition | None:
    return get_registry().get_provider(provider_id)


def list_providers() -> list[str]:
    return sorted(get_registry().providers.keys())


def get_tokenizer_for_provider(provider_id: str) -> str:
    """Get tokenizer name for a provider (for DocumentChunker)."""
    provider = get_provider_config(provider_id)
    if provider:
        return provider.tokenizer
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_merge_threshold_for_provider(provider_id: str) -> float:
    """Get merge threshold for a provider (for DocumentChunker)."""
    provider = get_provider_config(provider_id)
    if provider:
        return provider.merge_threshold
    return 0.95


def resolve_effective_model_config(
    provider_id: str,
    model_id: str,
    overrides: LlmRuntimeOverrides | dict[str, Any] | None = None,
) -> EffectiveModelConfig:
    """Resolve effective model configuration WITHOUT registry/capability.

    No models.yaml registry, no capability detection.
    Uses LiteLLM's own metadata for context limits.
    """
    provider = get_provider_config(provider_id)
    if not provider:
        logger.warning(
            "Unknown provider '%s'; using generic defaults for tokenizer and merge threshold.",
            provider_id,
        )
        provider = ProviderDefinition()

    if overrides is None:
        overrides = LlmRuntimeOverrides()
    elif isinstance(overrides, dict):
        overrides = LlmRuntimeOverrides(**overrides)

    connection = _resolve_connection(provider, overrides.connection, provider_id)
    litellm_model = build_litellm_model_name(provider_id, model_id, connection)

    # Use overrides or LiteLLM defaults (no registry lookup)
    context_limit = _resolve_context_limit(litellm_model, overrides.context_limit)
    max_output_tokens = _resolve_max_output_tokens(litellm_model, overrides.max_output_tokens)

    generation = _merge_generation(GenerationDefaults(), overrides.generation)
    reliability = _merge_reliability(ReliabilityDefaults(), overrides.reliability)

    if generation.max_tokens is None:
        generation = generation.model_copy(update={"max_tokens": max_output_tokens})
    assert generation.max_tokens is not None
    if generation.max_tokens > max_output_tokens:
        raise ConfigurationError(
            "max_tokens exceeds model limit",
            details={
                "model": model_id,
                "max_tokens": generation.max_tokens,
                "model_max_output_tokens": max_output_tokens,
            },
        )

    effective = EffectiveModelConfig(
        model_id=model_id,
        provider_id=provider_id.lower(),
        litellm_model=litellm_model,
        context_limit=context_limit,
        max_output_tokens=max_output_tokens,
        generation=generation,
        reliability=reliability,
        connection=connection,
        tokenizer=provider.tokenizer,
        merge_threshold=provider.merge_threshold,
        token_density=overrides.token_density,  # Use override directly, no registry
    )

    return effective
