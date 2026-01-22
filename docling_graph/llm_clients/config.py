"""
Centralized LLM provider configuration registry (YAML-based).

This module replaces the previous 547-line Python configuration with a
clean YAML-based approach, reducing code to ~304 lines while maintaining
all functionality.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""

    model_id: str
    context_limit: int
    max_new_tokens: int = 4096
    description: str = ""
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"ModelConfig({self.model_id}, context={self.context_limit}, "
            f"max_new_tokens={self.max_new_tokens})"
        )


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider_id: str
    models: Dict[str, ModelConfig]
    tokenizer: str
    content_ratio: float = 0.8

    def get_model(self, model_name: str) -> ModelConfig | None:
        """Get a specific model from this provider."""
        return self.models.get(model_name)

    def list_models(self) -> list[str]:
        """List all available models for this provider."""
        return list(self.models.keys())

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
                models[model_id] = ModelConfig(
                    model_id=model_id,
                    context_limit=model_data["context_limit"],
                    max_new_tokens=model_data.get("max_new_tokens", 4096),
                    description=model_data.get("description", ""),
                    notes=model_data.get("notes", ""),
                )

            # Create provider config
            self._providers[provider_id] = ProviderConfig(
                provider_id=provider_id,
                models=models,
                tokenizer=provider_data.get("tokenizer", "sentence-transformers/all-MiniLM-L6-v2"),
                content_ratio=provider_data.get("content_ratio", 0.8),
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
