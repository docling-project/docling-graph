"""
LLM Clients module with LiteLLM as the default execution path.
"""

from typing import Type

from .litellm import LiteLLMClient

__all__ = ["LiteLLMClient", "get_client"]


def get_client(provider: str) -> Type[LiteLLMClient]:
    """
    Get LLM client class for the specified provider.

    Every provider is served by the single LiteLLM-backed client; the provider
    string is resolved later by ``resolve_effective_model_config`` (unknown
    providers fall back to generic defaults with a warning). This function
    itself never raises.

    Args:
        provider: Provider name (mistral, ollama, vllm, openai, gemini, watsonx)

    Returns:
        The client class for the provider
    """
    return LiteLLMClient
