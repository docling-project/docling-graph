import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.llm_clients.base import BaseLlmClient


# A simple Pydantic model for testing
class MockTemplate(BaseModel):
    name: str
    age: int


# Fixture for a mock LLM client
@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=BaseLlmClient)
    client.__class__.__name__ = "MockLlmClient"
    client.context_limit = 8000  # Set as integer, not MagicMock
    return client


# Fixture for the LlmBackend
@pytest.fixture
def llm_backend(mock_llm_client):
    return LlmBackend(llm_client=mock_llm_client)


def test_init(llm_backend, mock_llm_client):
    """Test that the backend initializes with the client."""
    assert llm_backend.client == mock_llm_client


@patch("docling_graph.core.extractors.backends.llm_backend.get_extraction_prompt")
def test_extract_from_markdown_success(mock_get_prompt, llm_backend, mock_llm_client):
    """Test successful extraction and validation."""
    markdown = "This is a test."
    context = "test context"
    expected_json = {"name": "Test", "age": 30}
    schema_json = json.dumps(MockTemplate.model_json_schema(), indent=2)

    # Configure mock client
    mock_llm_client.get_json_response.return_value = expected_json
    mock_get_prompt.return_value = {"system": "sys", "user": "user"}

    # Run extraction
    result = llm_backend.extract_from_markdown(
        markdown=markdown, template=MockTemplate, context=context
    )

    # Assertions
    assert isinstance(result, MockTemplate)
    assert result.name == "Test"
    assert result.age == 30
    # Check that prompt was called (model_config is now passed automatically)
    mock_get_prompt.assert_called_once()
    call_kwargs = mock_get_prompt.call_args[1]
    assert call_kwargs["markdown_content"] == markdown
    assert call_kwargs["schema_json"] == schema_json
    assert not call_kwargs["is_partial"]
    assert "model_config" in call_kwargs  # New parameter
    mock_llm_client.get_json_response.assert_called_with(
        prompt={"system": "sys", "user": "user"}, schema_json=schema_json
    )


def test_extract_from_markdown_empty_input(llm_backend):
    """Test that empty or whitespace-only markdown returns None."""
    result_empty = llm_backend.extract_from_markdown(markdown="", template=MockTemplate)
    result_whitespace = llm_backend.extract_from_markdown(markdown="   \n ", template=MockTemplate)

    assert result_empty is None
    assert result_whitespace is None


def test_extract_from_markdown_no_json_returned(llm_backend, mock_llm_client):
    """Test when the LLM client returns no valid JSON (e.g., None)."""
    mock_llm_client.get_json_response.return_value = None

    result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)
    assert result is None


@patch("docling_graph.core.extractors.backends.llm_backend.rich_print")
def test_extract_from_markdown_validation_error(mock_rich_print, llm_backend, mock_llm_client):
    """Test when the LLM returns JSON that fails Pydantic validation."""
    # 'age' is missing, which is a required field
    invalid_json = {"name": "Test Only"}
    mock_llm_client.get_json_response.return_value = invalid_json

    result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)

    # Should fail validation and return None
    assert result is None
    # Check that a validation error was printed
    mock_rich_print.assert_any_call(
        "[blue][LlmBackend][/blue] [yellow]Validation Error for document:[/yellow]"
    )


@patch("docling_graph.core.extractors.backends.llm_backend.get_consolidation_prompt")
def test_consolidate_success(mock_get_prompt, llm_backend, mock_llm_client):
    """Test successful consolidation."""
    raw_models = [MockTemplate(name="Test", age=30)]
    programmatic_model = MockTemplate(name="Test", age=30)
    expected_json = {"name": "Consolidated", "age": 31}

    mock_llm_client.get_json_response.return_value = expected_json
    mock_get_prompt.return_value = "consolidation_prompt"

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models,
        programmatic_model=programmatic_model,
        template=MockTemplate,
    )

    assert isinstance(result, MockTemplate)
    assert result.name == "Consolidated"
    assert result.age == 31


@patch("docling_graph.core.extractors.backends.llm_backend.rich_print")
def test_consolidate_validation_error(mock_rich_print, llm_backend, mock_llm_client):
    """Test consolidation with a Pydantic validation error."""
    raw_models = [MockTemplate(name="Test", age=30)]
    programmatic_model = MockTemplate(name="Test", age=30)
    # 'age' is missing
    invalid_json = {"name": "Consolidated"}

    mock_llm_client.get_json_response.return_value = invalid_json

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models,
        programmatic_model=programmatic_model,
        template=MockTemplate,
    )

    assert result is None
    mock_rich_print.assert_any_call(
        "[blue][LlmBackend][/blue] [yellow]Validation Error during consolidation:[/yellow]"
    )


@patch("gc.collect")
def test_cleanup(mock_gc_collect, mock_llm_client):
    """Test that cleanup removes the client and calls garbage collection."""
    # Add a mock cleanup method to the client
    mock_llm_client.cleanup = MagicMock()

    backend = LlmBackend(llm_client=mock_llm_client)
    assert hasattr(backend, "client")

    backend.cleanup()

    # Check client's cleanup was called
    mock_llm_client.cleanup.assert_called_once()
    # Check client attribute was deleted
    assert not hasattr(backend, "client")
    # Check gc was called
    mock_gc_collect.assert_called_once()


# --- Tests for Phase 1 Fix 5: Model-Aware Backend ---


@patch("docling_graph.core.extractors.backends.llm_backend.get_model_config")
def test_init_with_model_config(mock_get_model_config, mock_llm_client):
    """Test backend initialization with model configuration."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    mock_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )
    mock_get_model_config.return_value = mock_config
    mock_llm_client.provider = "openai"
    mock_llm_client.model_id = "gpt-4"

    backend = LlmBackend(llm_client=mock_llm_client)

    assert backend.model_config == mock_config
    assert backend.model_config.capability == ModelCapability.ADVANCED


@patch("docling_graph.core.extractors.backends.llm_backend.get_model_config")
@patch("docling_graph.core.extractors.backends.llm_backend.detect_model_capability")
def test_init_with_fallback_detection(mock_detect, mock_get_model_config, mock_llm_client):
    """Test backend falls back to auto-detection when model not in registry."""
    from docling_graph.llm_clients.config import ModelCapability

    mock_get_model_config.return_value = None  # Not in registry
    mock_detect.return_value = ModelCapability.STANDARD
    mock_llm_client.context_limit = 8192

    backend = LlmBackend(llm_client=mock_llm_client)

    assert backend.model_config is not None
    assert backend.model_config.capability == ModelCapability.STANDARD
    mock_detect.assert_called_once_with(8192, "")


@patch("docling_graph.core.extractors.backends.llm_backend.get_extraction_prompt")
def test_extract_passes_model_config_to_prompt(mock_get_prompt, llm_backend, mock_llm_client):
    """Test that model_config is passed to prompt generation."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    llm_backend.model_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )

    markdown = "Test content"
    expected_json = {"name": "Test", "age": 30}
    mock_llm_client.get_json_response.return_value = expected_json
    mock_get_prompt.return_value = {"system": "sys", "user": "user"}

    llm_backend.extract_from_markdown(markdown=markdown, template=MockTemplate)

    # Verify model_config was passed to prompt generation
    mock_get_prompt.assert_called_once()
    call_kwargs = mock_get_prompt.call_args[1]
    assert "model_config" in call_kwargs
    assert call_kwargs["model_config"] == llm_backend.model_config


@patch("docling_graph.core.extractors.backends.llm_backend.get_consolidation_prompt")
def test_consolidate_chain_of_density(mock_get_prompt, llm_backend, mock_llm_client):
    """Test Chain of Density consolidation for advanced models."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    llm_backend.model_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )

    raw_models = [MockTemplate(name="Test1", age=30), MockTemplate(name="Test2", age=31)]
    programmatic_model = MockTemplate(name="Merged", age=30)

    # Mock Chain of Density prompts (list of 2 prompts)
    stage1_prompt = "Stage 1: Initial merge"
    stage2_prompt = "Stage 2: Refine with {stage1_result} and {originals}"
    mock_get_prompt.return_value = [stage1_prompt, stage2_prompt]

    # Mock LLM responses for both stages
    stage1_result = {"name": "Stage1", "age": 30}
    stage2_result = {"name": "Final", "age": 32}
    mock_llm_client.get_json_response.side_effect = [stage1_result, stage2_result]

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models, programmatic_model=programmatic_model, template=MockTemplate
    )

    # Should call LLM twice (stage 1 and stage 2)
    assert mock_llm_client.get_json_response.call_count == 2
    assert result.name == "Final"
    assert result.age == 32


@patch("docling_graph.core.extractors.backends.llm_backend.get_consolidation_prompt")
def test_consolidate_chain_of_density_stage2_fails(mock_get_prompt, llm_backend, mock_llm_client):
    """Test Chain of Density falls back to stage 1 if stage 2 fails."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    llm_backend.model_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )

    raw_models = [MockTemplate(name="Test1", age=30)]
    programmatic_model = MockTemplate(name="Merged", age=30)

    mock_get_prompt.return_value = ["Stage 1", "Stage 2: {stage1_result}"]

    # Stage 1 succeeds, Stage 2 fails
    stage1_result = {"name": "Stage1", "age": 30}
    mock_llm_client.get_json_response.side_effect = [stage1_result, None]

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models, programmatic_model=programmatic_model, template=MockTemplate
    )

    # Should use stage 1 result
    assert result.name == "Stage1"
    assert result.age == 30


@patch("docling_graph.core.extractors.backends.llm_backend.get_consolidation_prompt")
def test_consolidate_single_turn_simple_model(mock_get_prompt, llm_backend, mock_llm_client):
    """Test single-turn consolidation for simple models."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    llm_backend.model_config = ModelConfig(
        model_id="phi-3", context_limit=4096, capability=ModelCapability.SIMPLE
    )

    raw_models = [MockTemplate(name="Test1", age=30)]
    programmatic_model = MockTemplate(name="Merged", age=30)

    # Mock single prompt (string, not list)
    mock_get_prompt.return_value = "Single consolidation prompt"
    mock_llm_client.get_json_response.return_value = {"name": "Consolidated", "age": 31}

    result = llm_backend.consolidate_from_pydantic_models(
        raw_models=raw_models, programmatic_model=programmatic_model, template=MockTemplate
    )

    # Should call LLM once
    assert mock_llm_client.get_json_response.call_count == 1
    assert result.name == "Consolidated"
    assert result.age == 31


def test_cleanup_without_cleanup_method(mock_llm_client):
    """Test cleanup when client doesn't have cleanup method."""
    # Client without cleanup method
    mock_llm_client_no_cleanup = MagicMock(spec=BaseLlmClient)
    mock_llm_client_no_cleanup.__class__.__name__ = "MockClient"
    mock_llm_client_no_cleanup.context_limit = 8192  # Set as integer
    # Explicitly remove cleanup if it exists
    if hasattr(mock_llm_client_no_cleanup, "cleanup"):
        delattr(mock_llm_client_no_cleanup, "cleanup")

    backend = LlmBackend(llm_client=mock_llm_client_no_cleanup)

    # Should not raise error
    backend.cleanup()

    # Client should be deleted
    assert not hasattr(backend, "client")
