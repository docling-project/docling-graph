"""
LLM Clients module with LiteLLM as the default execution path.
"""

from typing import Type

from .litellm import LiteLLMClient

__all__ = ["LiteLLMClient", "get_client"]


def get_client(provider: str) -> Type[LiteLLMClient]:
    """
    Get LLM client class for the specified provider.

    Uses lazy imports, so client packages are only loaded when actually used.

    Args:
        provider: Provider name (mistral, ollama, vllm, openai, gemini, watsonx)

    Returns:
        The client class for the provider

    Raises:
        ValueError: If provider is not recognized
        ImportError: If provider package is not installed
    """
    return LiteLLMClient
