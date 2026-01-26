"""
Unit tests for max_tokens and timeout configuration.

Tests the new max_tokens and timeout parameters added to fix the VllmClient hanging issue.
"""

import pytest

from docling_graph.llm_clients.config import get_model_config, get_provider_config


class TestMaxTokensConfiguration:
    """Test max_tokens configuration at provider and model level."""

    def test_provider_has_default_max_tokens(self):
        """Verify all providers have default_max_tokens configured."""
        providers = ["mistral", "openai", "google", "anthropic", "watsonx", "vllm", "ollama"]
        # vllm has higher default (16384) for large schemas like rheology research
        expected_defaults = {
            "vllm": 16384,
        }

        for provider_id in providers:
            provider = get_provider_config(provider_id)
            assert provider is not None, f"Provider {provider_id} not found"
            assert hasattr(provider, "default_max_tokens"), (
                f"Provider {provider_id} missing default_max_tokens"
            )
            expected = expected_defaults.get(provider_id, 8192)
            assert provider.default_max_tokens == expected, (
                f"Provider {provider_id} has wrong default_max_tokens: "
                f"expected {expected}, got {provider.default_max_tokens}"
            )

    def test_provider_has_timeout_seconds(self):
        """Verify all providers have timeout_seconds configured."""
        providers = ["mistral", "openai", "google", "anthropic", "watsonx", "vllm", "ollama"]

        for provider_id in providers:
            provider = get_provider_config(provider_id)
            assert provider is not None, f"Provider {provider_id} not found"
            assert hasattr(provider, "timeout_seconds"), (
                f"Provider {provider_id} missing timeout_seconds"
            )
            assert provider.timeout_seconds in [300, 600], (
                f"Provider {provider_id} has unexpected timeout"
            )

    def test_vllm_provider_has_correct_timeout(self):
        """Verify vLLM provider has 600s timeout (10 minutes)."""
        provider = get_provider_config("vllm")
        assert provider is not None
        assert provider.timeout_seconds == 600, "vLLM should have 600s timeout"

    def test_api_providers_have_shorter_timeout(self):
        """Verify API providers have 300s timeout (5 minutes)."""
        api_providers = ["mistral", "openai", "google", "anthropic"]

        for provider_id in api_providers:
            provider = get_provider_config(provider_id)
            assert provider is not None
            assert provider.timeout_seconds == 300, f"{provider_id} should have 300s timeout"

    def test_provider_get_max_tokens_returns_default(self):
        """Test ProviderConfig.get_max_tokens() returns default when model has no override."""
        provider = get_provider_config("vllm")
        assert provider is not None

        # Model without max_tokens override should return provider default (16384 for vllm)
        max_tokens = provider.get_max_tokens("meta-llama/Llama-3.1-8B")
        assert max_tokens == 16384

    def test_provider_get_timeout_returns_default(self):
        """Test ProviderConfig.get_timeout() returns default when model has no override."""
        provider = get_provider_config("vllm")
        assert provider is not None

        # Model without timeout override should return provider default
        timeout = provider.get_timeout("meta-llama/Llama-3.1-8B")
        assert timeout == 600


class TestModelConfigFields:
    """Test ModelConfig has max_tokens and timeout fields."""

    def test_model_config_has_max_tokens_field(self):
        """Verify ModelConfig has max_tokens field."""
        config = get_model_config("vllm", "meta-llama/Llama-3.1-8B")
        assert config is not None
        assert hasattr(config, "max_tokens")

    def test_model_config_has_timeout_field(self):
        """Verify ModelConfig has timeout field."""
        config = get_model_config("vllm", "meta-llama/Llama-3.1-8B")
        assert config is not None
        assert hasattr(config, "timeout")

    def test_model_config_max_tokens_can_be_none(self):
        """Verify max_tokens can be None (uses provider default)."""
        config = get_model_config("vllm", "meta-llama/Llama-3.1-8B")
        assert config is not None
        # Most models don't have explicit max_tokens, should be None
        assert config.max_tokens is None

    def test_model_config_timeout_can_be_none(self):
        """Verify timeout can be None (uses provider default)."""
        config = get_model_config("vllm", "meta-llama/Llama-3.1-8B")
        assert config is not None
        # Most models don't have explicit timeout, should be None
        assert config.timeout is None


class TestBaseLlmClientIntegration:
    """Test BaseLlmClient properly loads max_tokens and timeout."""

    def test_client_properties_exist(self):
        """Verify BaseLlmClient has max_tokens and timeout properties."""
        from docling_graph.llm_clients.base import BaseLlmClient

        # Check that the properties are defined
        assert hasattr(BaseLlmClient, "max_tokens")
        assert hasattr(BaseLlmClient, "timeout")

    def test_client_init_accepts_max_tokens(self):
        """Verify BaseLlmClient.__init__ accepts max_tokens parameter."""
        import inspect

        from docling_graph.llm_clients.base import BaseLlmClient

        sig = inspect.signature(BaseLlmClient.__init__)
        assert "max_tokens" in sig.parameters

    def test_client_init_accepts_timeout(self):
        """Verify BaseLlmClient.__init__ accepts timeout parameter."""
        import inspect

        from docling_graph.llm_clients.base import BaseLlmClient

        sig = inspect.signature(BaseLlmClient.__init__)
        assert "timeout" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
