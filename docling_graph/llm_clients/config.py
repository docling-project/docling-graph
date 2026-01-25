"""
Centralized LLM provider configuration registry (YAML-based).

This module replaces the previous 547-line Python configuration with a
clean YAML-based approach, reducing code to ~304 lines while maintaining
all functionality.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capability tiers for adaptive extraction strategies."""

    SIMPLE = "simple"  # 1B-7B models: simplified prompts, no Chain of Density
    STANDARD = "standard"  # 7B-13B models: standard prompts, single-pass consolidation
    ADVANCED = "advanced"  # 13B+ models: complex prompts, Chain of Density support


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    context_limit: int
    max_new_tokens: int = 4096
    max_tokens: int | None = None
    timeout: int | None = None
    capability: ModelCapability = ModelCapability.STANDARD
    description: str = ""
    notes: str = ""

    @property
    def supports_chain_of_density(self) -> bool:
        """Check if model supports multi-turn consolidation."""
        return self.capability == ModelCapability.ADVANCED

    @property
    def requires_strict_schema(self) -> bool:
        """Check if model needs strict schema compliance."""
        return self.capability == ModelCapability.SIMPLE

    def __repr__(self) -> str:
        return (
            f"ModelConfig({self.model_id}, context={self.context_limit}, "
            f"max_tokens={self.max_new_tokens}, capability={self.capability.value})"
        )


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider_id: str
    models: Dict[str, ModelConfig]
    tokenizer: str
    content_ratio: float = 0.8

    # Batching configuration
    merge_threshold: float = 0.85
    """Merge batches if below this % of context (0.0-1.0)."""

    rate_limit_rpm: int | None = None
    """Rate limit in requests per minute (None = no limit)."""

    supports_batching: bool = True
    """Whether provider supports efficient batching."""

    default_max_tokens: int = 8192
    """Default maximum tokens to generate for responses."""

    timeout_seconds: int = 300
    """Default request timeout in seconds."""

    def get_model(self, model_name: str) -> ModelConfig | None:
        """Get a specific model from this provider."""
        return self.models.get(model_name)

    def list_models(self) -> list[str]:
        """List all available models for this provider."""
        return list(self.models.keys())

    def get_max_tokens(self, model_name: str) -> int:
        """Get max_tokens for a model (model-specific or provider default)."""
        model = self.get_model(model_name)
        if model and model.max_tokens is not None:
            return model.max_tokens
        return self.default_max_tokens

    def get_timeout(self, model_name: str) -> int:
        """Get timeout for a model (model-specific or provider default)."""
        model = self.get_model(model_name)
        if model and model.timeout is not None:
            return model.timeout
        return self.timeout_seconds

    def get_recommended_chunk_size(self, model_name: str, schema_size: int = 0) -> int:
        """
        Get recommended chunk size for a model in this provider.

        Uses dynamic adjustment based on schema complexity to prevent output overflow.

        Args:
            model_name: Name of the model
            schema_size: Size of the Pydantic schema JSON (optional, for dynamic adjustment)

        Returns:
            Recommended max_tokens for DocumentChunker
        """
        model = self.get_model(model_name)
        if not model:
            return 5120  # Default fallback

        # Estimate output density from schema complexity
        if schema_size > 10000:
            output_ratio = 0.8  # Complex schema: 80% of input becomes output
        elif schema_size > 5000:
            output_ratio = 0.5  # Medium schema: 50% expansion
        elif schema_size > 0:
            output_ratio = 0.3  # Simple schema: 30% expansion
        else:
            output_ratio = 0.4  # No schema info: conservative default

        system_prompt_tokens = 500
        safety_buffer = 0.8  # 20% safety margin

        # Strategy 1: Output-constrained sizing
        max_safe_chunk = int(model.max_new_tokens / output_ratio * safety_buffer)

        # Strategy 2: Context-constrained sizing
        max_by_context = int((model.context_limit - system_prompt_tokens) * 0.7)

        # Use the smaller of the two (most conservative)
        chunk_size = min(max_safe_chunk, max_by_context)

        return max(1024, chunk_size)  # Minimum 1024 tokens


class ConfigRegistry:
    """
    Centralized configuration registry.

    Loads model configurations from YAML file, providing a clean
    separation between code and configuration data.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize registry and load configuration.

        Args:
            config_path: Optional path to YAML config file.
                        Defaults to models.yaml in this directory.
        """
        self._providers: Dict[str, ProviderConfig] = {}

        if config_path is None:
            config_path = Path(__file__).parent / "models.yaml"

        self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e

        # Parse providers
        for provider_id, provider_data in data.get("providers", {}).items():
            models = {}

            # Parse models for this provider
            for model_id, model_data in provider_data.get("models", {}).items():
                # Parse capability field
                capability_str = model_data.get("capability", "standard")
                try:
                    capability = ModelCapability(capability_str)
                except ValueError:
                    logger.warning(
                        f"Invalid capability '{capability_str}' for {model_id}, "
                        "defaulting to 'standard'"
                    )
                    capability = ModelCapability.STANDARD

                models[model_id] = ModelConfig(
                    model_id=model_id,
                    context_limit=model_data["context_limit"],
                    max_new_tokens=model_data.get("max_new_tokens", 4096),
                    max_tokens=model_data.get("max_tokens"),
                    timeout=model_data.get("timeout"),
                    capability=capability,
                    description=model_data.get("description", ""),
                    notes=model_data.get("notes", ""),
                )

            # Create provider config
            self._providers[provider_id] = ProviderConfig(
                provider_id=provider_id,
                models=models,
                tokenizer=provider_data.get("tokenizer", "sentence-transformers/all-MiniLM-L6-v2"),
                content_ratio=provider_data.get("content_ratio", 0.8),
                merge_threshold=provider_data.get("merge_threshold", 0.85),
                rate_limit_rpm=provider_data.get("rate_limit_rpm"),
                supports_batching=provider_data.get("supports_batching", True),
                default_max_tokens=provider_data.get("default_max_tokens", 8192),
                timeout_seconds=provider_data.get("timeout_seconds", 300),
            )

        logger.info(
            f"Loaded {len(self._providers)} providers with {sum(len(p.models) for p in self._providers.values())} models"
        )

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """
        Get provider configuration by ID.

        Args:
            provider_id: Provider identifier (e.g., "mistral", "openai")

        Returns:
            ProviderConfig or None if not found
        """
        return self._providers.get(provider_id.lower())

    def get_model(self, provider_id: str, model_id: str) -> ModelConfig | None:
        """
        Get model configuration by provider and model ID.

        Args:
            provider_id: Provider identifier
            model_id: Model identifier

        Returns:
            ModelConfig or None if not found
        """
        provider = self.get_provider(provider_id)
        return provider.get_model(model_id) if provider else None

    def list_providers(self) -> list[str]:
        """List all available provider IDs."""
        return list(self._providers.keys())

    def list_models(self, provider_id: str) -> list[str] | None:
        """
        List all models for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            List of model IDs or None if provider not found
        """
        provider = self.get_provider(provider_id)
        return provider.list_models() if provider else None


# Global registry instance
_registry: ConfigRegistry | None = None


def _get_registry() -> ConfigRegistry:
    """Get or create global registry instance."""
    global _registry
    if _registry is None:
        _registry = ConfigRegistry()
    return _registry


# Public API functions (backward compatible with old config.py)


def get_provider_config(provider: str) -> ProviderConfig | None:
    """
    Get provider configuration by name.

    Args:
        provider: Provider ID (e.g., "mistral", "openai", "gemini")

    Returns:
        ProviderConfig or None if not found
    """
    return _get_registry().get_provider(provider)


def get_model_config(provider: str, model_name: str) -> ModelConfig | None:
    """
    Get model configuration by provider and model name.

    Args:
        provider: Provider ID (e.g., "mistral", "openai")
        model_name: Model name (e.g., "mistral-large-latest")

    Returns:
        ModelConfig or None if not found
    """
    return _get_registry().get_model(provider, model_name)


def get_context_limit(provider: str, model: str) -> int:
    """
    Get context window size for a model.

    Args:
        provider: Provider ID
        model: Model name

    Returns:
        Context limit in tokens (defaults to 8000 if not found)
    """
    config = get_model_config(provider, model)
    return config.context_limit if config else 8000


def get_tokenizer_for_provider(provider: str) -> str:
    """
    Get recommended tokenizer for a provider.

    Args:
        provider: Provider ID

    Returns:
        Tokenizer name (HuggingFace model or special name like "tiktoken")
    """
    provider_config = get_provider_config(provider)
    if provider_config:
        return provider_config.tokenizer
    return "sentence-transformers/all-MiniLM-L6-v2"  # Default fallback


def get_recommended_chunk_size(provider: str, model: str, schema_size: int = 0) -> int:
    """
    Get recommended chunk size for chunker based on model's context window.

    Args:
        provider: Provider ID
        model: Model name
        schema_size: Size of the Pydantic schema JSON (optional, for dynamic adjustment)

    Returns:
        Recommended max_tokens for DocumentChunker
    """
    provider_config = get_provider_config(provider)
    if provider_config:
        return provider_config.get_recommended_chunk_size(model, schema_size)
    return 5120  # Default fallback


def list_providers() -> list[str]:
    """List all available provider IDs."""
    return _get_registry().list_providers()


def list_models(provider: str) -> list[str] | None:
    """List all models for a provider."""
    return _get_registry().list_models(provider)


def detect_model_capability(
    context_limit: int, model_name: str = "", max_new_tokens: int | None = None
) -> ModelCapability:
    """
    Auto-detect model capability from model characteristics.

    Used as fallback when model not in registry.

    Priority:
    1. Model name (most reliable indicator)
    2. max_new_tokens (better proxy than context for reasoning capability)
    3. Context limit (last resort, can be misleading)

    Args:
        context_limit: Model's context window size
        model_name: Optional model name for heuristic detection
        max_new_tokens: Optional max output tokens (better capability indicator)

    Returns:
        Detected ModelCapability
    """
    name_lower = model_name.lower()

    # Priority 1: Check model name for explicit size hints (most reliable)
    # Small models (1B-3B parameters)
    if any(size in name_lower for size in ["1b", "350m", "500m", "2b", "3b"]):
        logger.info(
            f"Detected SIMPLE capability from model name: {model_name} (small parameter count)"
        )
        return ModelCapability.SIMPLE

    # Large models (70B+ parameters)
    if any(size in name_lower for size in ["70b", "65b", "405b"]):
        logger.info(
            f"Detected ADVANCED capability from model name: {model_name} (large parameter count)"
        )
        return ModelCapability.ADVANCED

    # Priority 2: Use max_new_tokens if available (better proxy than context)
    # Small models often have limited output capacity regardless of context size
    if max_new_tokens is not None:
        if max_new_tokens <= 2048:
            logger.info(
                f"Detected SIMPLE capability from max_new_tokens: {max_new_tokens} "
                "(limited output capacity)"
            )
            return ModelCapability.SIMPLE
        elif max_new_tokens <= 4096:
            logger.info(f"Detected STANDARD capability from max_new_tokens: {max_new_tokens}")
            return ModelCapability.STANDARD
        else:
            logger.info(
                f"Detected ADVANCED capability from max_new_tokens: {max_new_tokens} "
                "(high output capacity)"
            )
            return ModelCapability.ADVANCED

    # Priority 3: Fall back to context limit (least reliable)
    # Note: Modern small models can have large contexts (e.g., Granite 1B with 128K)
    if context_limit <= 4096:
        logger.warning(
            f"Detected SIMPLE capability from context_limit: {context_limit} "
            "(fallback heuristic, may be inaccurate)"
        )
        return ModelCapability.SIMPLE
    elif context_limit <= 32768:
        logger.warning(
            f"Detected STANDARD capability from context_limit: {context_limit} "
            "(fallback heuristic, may be inaccurate)"
        )
        return ModelCapability.STANDARD
    else:
        logger.warning(
            f"Detected ADVANCED capability from context_limit: {context_limit} "
            "(fallback heuristic, may be inaccurate for small models with large contexts)"
        )
        return ModelCapability.ADVANCED
