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


class TestModelCapability:
    """Test suite for ModelCapability enum and detection (Phase 1 Fix 2)."""

    def test_model_capability_enum_values(self):
        """Test ModelCapability enum has correct values."""
        from docling_graph.llm_clients.config import ModelCapability

        assert ModelCapability.SIMPLE.value == "simple"
        assert ModelCapability.STANDARD.value == "standard"
        assert ModelCapability.ADVANCED.value == "advanced"

    def test_detect_model_capability_simple(self):
        """Test capability detection for simple models (1B-5B)."""
        from docling_graph.llm_clients.config import ModelCapability, detect_model_capability

        # Small context limits indicate simple models
        assert detect_model_capability(2048) == ModelCapability.SIMPLE
        assert detect_model_capability(4096) == ModelCapability.SIMPLE

    def test_detect_model_capability_standard(self):
        """Test capability detection for standard models (7B-13B)."""
        from docling_graph.llm_clients.config import ModelCapability, detect_model_capability

        # Medium context limits indicate standard models
        assert detect_model_capability(8192) == ModelCapability.STANDARD
        assert detect_model_capability(16384) == ModelCapability.STANDARD
        assert detect_model_capability(32768) == ModelCapability.STANDARD

    def test_detect_model_capability_advanced(self):
        """Test capability detection for advanced models (13B+)."""
        from docling_graph.llm_clients.config import ModelCapability, detect_model_capability

        # Large context limits indicate advanced models
        assert detect_model_capability(65536) == ModelCapability.ADVANCED
        assert detect_model_capability(128000) == ModelCapability.ADVANCED
        assert detect_model_capability(200000) == ModelCapability.ADVANCED

    def test_model_config_capability_properties(self):
        """Test ModelConfig capability helper properties."""
        from docling_graph.llm_clients.config import ModelCapability, ModelConfig

        # Simple model
        simple_model = ModelConfig(
            model_id="phi-3", context_limit=4096, capability=ModelCapability.SIMPLE
        )
        assert simple_model.requires_strict_schema is True
        assert simple_model.supports_chain_of_density is False

        # Standard model
        standard_model = ModelConfig(
            model_id="mistral-7b", context_limit=8192, capability=ModelCapability.STANDARD
        )
        assert standard_model.requires_strict_schema is False
        assert standard_model.supports_chain_of_density is False

        # Advanced model
        advanced_model = ModelConfig(
            model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
        )
        assert advanced_model.requires_strict_schema is False
        assert advanced_model.supports_chain_of_density is True

    def test_model_config_without_capability(self):
        """Test ModelConfig helper properties when capability is not provided."""
        from docling_graph.llm_clients.config import ModelConfig

        # Create model without capability (will be None by default if optional)
        model = ModelConfig(model_id="unknown", context_limit=8192)
        # When capability is None, properties should return False for safety
        assert model.requires_strict_schema is False
        assert model.supports_chain_of_density is False


class TestYAMLCapabilityConfiguration:
    """Test suite for YAML capability configuration (Phase 1 Fix 1)."""

    def test_all_models_have_capability(self):
        """Test that all models have capability field defined."""
        providers = ["mistral", "openai", "google", "watsonx", "ollama", "vllm"]
        for provider_id in providers:
            provider = config.get_provider_config(provider_id)
            if provider:
                for model_id, model_config in provider.models.items():
                    assert model_config.capability is not None, (
                        f"{provider_id}/{model_id} missing capability"
                    )

    def test_simple_models_have_simple_capability(self):
        """Test that small models are classified as simple."""
        # Check Phi models
        ollama_config = config.get_provider_config("ollama")
        if ollama_config and "phi3" in ollama_config.models:
            phi3 = ollama_config.models["phi3"]
            assert phi3.capability.value == "simple"

    def test_standard_models_have_standard_capability(self):
        """Test that medium models are classified as standard."""
        # Check Mistral 7B
        mistral_config = config.get_provider_config("mistral")
        if mistral_config and "mistral-small-latest" in mistral_config.models:
            mistral_small = mistral_config.models["mistral-small-latest"]
            assert mistral_small.capability.value == "standard"

    def test_advanced_models_have_advanced_capability(self):
        """Test that large models are classified as advanced."""
        # Check GPT-4
        openai_config = config.get_provider_config("openai")
        if openai_config and "gpt-4" in openai_config.models:
            gpt4 = openai_config.models["gpt-4"]
            assert gpt4.capability.value == "advanced"

        # Check Claude
        anthropic_config = config.get_provider_config("anthropic")
        if anthropic_config and "claude-3-5-sonnet-20241022" in anthropic_config.models:
            claude = anthropic_config.models["claude-3-5-sonnet-20241022"]
            assert claude.capability.value == "advanced"


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
