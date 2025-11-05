import pytest

from docling_graph.llm_clients import config


# Test ModelConfig
def test_model_config_repr():
    """Tests the __repr__ method for a clear debug output."""
    mc = config.ModelConfig(model_id="test-model", context_limit=10000)
    # Recommended chunk size = 10000 * 0.8 * 0.8 = 6400
    expected_repr = "ModelConfig(test-model, context=10000, tokens_per_chunk=6400)"
    assert repr(mc) == expected_repr


# Test ProviderConfig
def test_provider_config_methods():
    """Tests the helper methods within the ProviderConfig dataclass."""
    model_conf = config.ModelConfig(model_id="test-model", context_limit=10000)
    provider_conf = config.ProviderConfig(
        provider_id="test_provider",
        models={"test-model": model_conf},
        tokenizer="test_tokenizer",
        content_ratio=0.8,
    )

    assert provider_conf.get_model("test-model") == model_conf
    assert provider_conf.get_model("unknown-model") is None
    assert provider_conf.list_models() == ["test-model"]

    # 10000 * 0.8 * 0.8 = 6400
    assert provider_conf.get_recommended_chunk_size("test-model") == 6400

    # Test minimum chunk size enforcement
    small_model_conf = config.ModelConfig(model_id="small-model", context_limit=1000)
    provider_conf.models["small-model"] = small_model_conf
    # 1000 * 0.8 * 0.8 = 640, but min is 1024
    assert provider_conf.get_recommended_chunk_size("small-model") == 1024

    # Test fallback for unknown model
    assert provider_conf.get_recommended_chunk_size("unknown-model") == 5120


# Test lookup functions
def test_get_provider_config():
    """Tests the top-level provider config lookup."""
    openai_config = config.get_provider_config("openai")
    assert openai_config is not None
    assert openai_config.provider_id == "openai"
    assert "gpt-4o" in openai_config.models

    # Test case insensitivity
    assert config.get_provider_config("OPENAI") == openai_config
    # Test not found
    assert config.get_provider_config("unknown_provider") is None


def test_get_model_config():
    """Tests the top-level model config lookup."""
    gpt_4o = config.get_model_config("openai", "gpt-4o")
    assert gpt_4o is not None
    assert gpt_4o.model_id == "gpt-4o"
    assert gpt_4o.context_limit == 128000

    assert config.get_model_config("openai", "unknown-model") is None
    assert config.get_model_config("unknown-provider", "gpt-4o") is None


def test_get_context_limit():
    """Tests the context limit retrieval helper."""
    assert config.get_context_limit("openai", "gpt-4o") == 128000
    assert config.get_context_limit("openai", "gpt-4") == 8192
    assert config.get_context_limit("mistral", "mistral-small-latest") == 32000

    # Test default fallback
    assert config.get_context_limit("openai", "unknown-model") == 8000
    assert config.get_context_limit("unknown-provider", "gpt-4o") == 8000


def test_get_tokenizer_for_provider():
    """Tests the tokenizer retrieval helper."""
    assert config.get_tokenizer_for_provider("openai") == "tiktoken"
    assert config.get_tokenizer_for_provider("mistral") == "mistralai/Mistral-7B-Instruct-v0.2"
    assert config.get_tokenizer_for_provider("google") == "sentence-transformers/all-MiniLM-L6-v2"

    # Test default fallback
    assert (
        config.get_tokenizer_for_provider("unknown-provider")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_get_recommended_chunk_size():
    """Tests the recommended chunk size retrieval helper."""
    # gpt-4o: 128000 * 0.8 (content_ratio) * 0.8 (safety) = 81920
    assert config.get_recommended_chunk_size("openai", "gpt-4o") == 81920
    # granite-20b-multilingual: 4096 * 0.75 * 0.8 = 2457.6 -> 2457
    assert config.get_recommended_chunk_size("ibm", "granite-20b-multilingual") == 2457

    # Test default fallback
    assert config.get_recommended_chunk_size("openai", "unknown-model") == 5120
    assert config.get_recommended_chunk_size("unknown-provider", "gpt-4o") == 5120


def test_list_providers():
    """Tests that list_providers returns a list of known provider keys."""
    providers = config.list_providers()
    assert isinstance(providers, list)
    assert "openai" in providers
    assert "mistral" in providers
    assert "gemini" in providers
    assert "ollama" in providers


def test_list_models():
    """Tests that list_models returns a list of model keys for a provider."""
    openai_models = config.list_models("openai")
    assert isinstance(openai_models, list)
    assert "gpt-4o" in openai_models
    assert "gpt-3.5-turbo" in openai_models

    assert config.list_models("unknown-provider") is None
