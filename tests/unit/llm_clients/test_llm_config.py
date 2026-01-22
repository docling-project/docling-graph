"""Tests for YAML-based configuration system."""

import pytest

from docling_graph.llm_clients import config


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        mc = config.ModelConfig(model_id="test-model", context_limit=10000, max_new_tokens=2048)
        assert mc.model_id == "test-model"
        assert mc.context_limit == 10000
        assert mc.max_new_tokens == 2048

    def test_model_config_repr(self):
        """Test ModelConfig string representation."""
        mc = config.ModelConfig(model_id="test-model", context_limit=10000, max_new_tokens=2048)
        repr_str = repr(mc)
        assert "test-model" in repr_str
        assert "10000" in repr_str
        assert "2048" in repr_str


class TestProviderConfig:
    """Test suite for ProviderConfig."""

    def test_provider_config_creation(self):
        """Test ProviderConfig creation."""
        model_conf = config.ModelConfig(model_id="test-model", context_limit=10000)
        provider_conf = config.ProviderConfig(
            provider_id="test_provider",
            models={"test-model": model_conf},
            tokenizer="test-tokenizer",
        )
        assert provider_conf.provider_id == "test_provider"
        assert "test-model" in provider_conf.models

    def test_get_model_success(self):
        """Test getting existing model."""
        model_conf = config.ModelConfig(model_id="test-model", context_limit=10000)
        provider_conf = config.ProviderConfig(
            provider_id="test_provider",
            models={"test-model": model_conf},
            tokenizer="test-tokenizer",
        )
        result = provider_conf.get_model("test-model")
        assert result == model_conf

    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        provider_conf = config.ProviderConfig(
            provider_id="test_provider", models={}, tokenizer="test-tokenizer"
        )
        result = provider_conf.get_model("unknown-model")
        assert result is None

    def test_list_models(self):
        """Test listing all models."""
        model1 = config.ModelConfig(model_id="model1", context_limit=8000)
        model2 = config.ModelConfig(model_id="model2", context_limit=16000)
        provider_conf = config.ProviderConfig(
            provider_id="test_provider",
            models={"model1": model1, "model2": model2},
            tokenizer="test-tokenizer",
        )
        models = provider_conf.list_models()
        assert set(models) == {"model1", "model2"}


class TestConfigLookupFunctions:
    """Test suite for configuration lookup functions."""

    def test_get_provider_config_success(self):
        """Test getting provider config."""
        openai_config = config.get_provider_config("openai")
        assert openai_config is not None
        assert openai_config.provider_id == "openai"
        assert len(openai_config.models) > 0

    def test_get_provider_config_case_insensitive(self):
        """Test case-insensitive provider lookup."""
        config1 = config.get_provider_config("openai")
        config2 = config.get_provider_config("OPENAI")
        config3 = config.get_provider_config("OpenAI")
        assert config1 == config2 == config3

    def test_get_provider_config_not_found(self):
        """Test getting non-existent provider."""
        result = config.get_provider_config("unknown_provider")
        assert result is None

    def test_get_model_config_success(self):
        """Test getting model config."""
        gpt4 = config.get_model_config("openai", "gpt-4")
        assert gpt4 is not None
        assert gpt4.model_id == "gpt-4"
        assert gpt4.context_limit > 0

    def test_get_model_config_not_found(self):
        """Test getting non-existent model."""
        result = config.get_model_config("openai", "unknown-model")
        assert result is None

    def test_get_model_config_unknown_provider(self):
        """Test getting model from unknown provider."""
        result = config.get_model_config("unknown-provider", "gpt-4")
        assert result is None

    def test_get_context_limit_success(self):
        """Test getting context limit."""
        limit = config.get_context_limit("openai", "gpt-4")
        assert limit == 8192

    def test_get_context_limit_large_model(self):
        """Test getting context limit for large context model."""
        limit = config.get_context_limit("openai", "gpt-4o")
        assert limit == 128000

    def test_get_context_limit_default_fallback(self):
        """Test default fallback for unknown model."""
        limit = config.get_context_limit("openai", "unknown-model")
        assert limit == 8000  # Default fallback

    def test_get_context_limit_unknown_provider(self):
        """Test default fallback for unknown provider."""
        limit = config.get_context_limit("unknown-provider", "any-model")
        assert limit == 8000  # Default fallback


class TestYAMLConfiguration:
    """Test suite for YAML configuration loading."""

    def test_mistral_models_loaded(self):
        """Test that Mistral models are loaded from YAML."""
        mistral_config = config.get_provider_config("mistral")
        assert mistral_config is not None
        assert "mistral-large-latest" in mistral_config.models
        assert "mistral-small-latest" in mistral_config.models

    def test_openai_models_loaded(self):
        """Test that OpenAI models are loaded from YAML."""
        openai_config = config.get_provider_config("openai")
        assert openai_config is not None
        assert "gpt-4" in openai_config.models
        assert "gpt-4o" in openai_config.models

    def test_gemini_models_loaded(self):
        """Test that Gemini models are loaded from YAML."""
        gemini_config = config.get_provider_config("google")
        assert gemini_config is not None
        assert len(gemini_config.models) > 0

    def test_watsonx_models_loaded(self):
        """Test that WatsonX models are loaded from YAML."""
        watsonx_config = config.get_provider_config("watsonx")
        assert watsonx_config is not None
        assert len(watsonx_config.models) > 0

    def test_ollama_models_loaded(self):
        """Test that Ollama models are loaded from YAML."""
        ollama_config = config.get_provider_config("ollama")
        assert ollama_config is not None
        assert len(ollama_config.models) > 0

    def test_vllm_models_loaded(self):
        """Test that vLLM models are loaded from YAML."""
        vllm_config = config.get_provider_config("vllm")
        assert vllm_config is not None
        assert len(vllm_config.models) > 0

    def test_all_models_have_context_limits(self):
        """Test that all models have context limits defined."""
        providers = ["mistral", "openai", "google", "watsonx", "ollama", "vllm"]
        for provider_id in providers:
            provider = config.get_provider_config(provider_id)
            if provider:
                for model_id, model_config in provider.models.items():
                    assert model_config.context_limit > 0, (
                        f"{provider_id}/{model_id} missing context_limit"
                    )

    def test_all_models_have_max_tokens(self):
        """Test that all models have max_new_tokens defined."""
        providers = ["mistral", "openai", "google", "watsonx", "ollama", "vllm"]
        for provider_id in providers:
            provider = config.get_provider_config(provider_id)
            if provider:
                for model_id, model_config in provider.models.items():
                    assert model_config.max_new_tokens > 0, (
                        f"{provider_id}/{model_id} missing max_new_tokens"
                    )


class TestConfigRegistry:
    """Test suite for ConfigRegistry."""

    def test_registry_multiple_instances(self):
        """Test that ConfigRegistry can be instantiated multiple times."""
        from docling_graph.llm_clients.config import ConfigRegistry

        registry1 = ConfigRegistry()
        registry2 = ConfigRegistry()
        # ConfigRegistry is NOT a singleton - each instance loads config independently
        assert registry1 is not registry2
        # But both should have the same providers loaded
        assert registry1.list_providers() == registry2.list_providers()

    def test_registry_has_providers(self):
        """Test that registry has providers loaded."""
        from docling_graph.llm_clients.config import ConfigRegistry

        registry = ConfigRegistry()
        assert len(registry.list_providers()) > 0

    def test_registry_get_provider(self):
        """Test getting provider from registry."""
        from docling_graph.llm_clients.config import ConfigRegistry

        registry = ConfigRegistry()
        openai = registry.get_provider("openai")
        assert openai is not None
        assert openai.provider_id == "openai"


# Made with Bob
