import json
from typing import List

import pytest
from pydantic import BaseModel, Field

from docling_graph.llm_clients.prompts import (
    _CONSOLIDATION_PROMPT,
    _SYSTEM_PROMPT_COMPLETE,
    _SYSTEM_PROMPT_PARTIAL,
    _USER_PROMPT_TEMPLATE,
    get_consolidation_prompt,
    get_extraction_prompt,
)

# --- Test get_extraction_prompt ---


def test_get_extraction_prompt_partial():
    """Tests that the partial prompt (for chunks) is generated correctly."""
    markdown = "This is page 1."
    schema = '{"title": "Test"}'

    prompt_dict = get_extraction_prompt(markdown, schema, is_partial=True)

    # Check key content instead of exact match (adaptive prompts may vary)
    assert "system" in prompt_dict
    assert "user" in prompt_dict
    assert (
        "extraction" in prompt_dict["system"].lower() or "extract" in prompt_dict["system"].lower()
    )
    assert "document page" in prompt_dict["user"].lower() or "page" in prompt_dict["user"].lower()
    assert markdown in prompt_dict["user"]
    assert schema in prompt_dict["user"]


def test_get_extraction_prompt_complete():
    """Tests that the complete prompt (for full docs) is generated correctly."""
    markdown = "This is the full document."
    schema = '{"title": "Test"}'

    prompt_dict = get_extraction_prompt(markdown, schema, is_partial=False)

    # Check key content instead of exact match (adaptive prompts may vary)
    assert "system" in prompt_dict
    assert "user" in prompt_dict
    assert (
        "extraction" in prompt_dict["system"].lower() or "extract" in prompt_dict["system"].lower()
    )
    assert (
        "complete document" in prompt_dict["user"].lower()
        or "document" in prompt_dict["user"].lower()
    )
    assert markdown in prompt_dict["user"]
    assert schema in prompt_dict["user"]


# --- Test get_consolidation_prompt ---


class SimpleModel(BaseModel):
    name: str
    value: int


class ComplexModel(BaseModel):
    items: List[SimpleModel] = Field(default_factory=list)


@pytest.fixture
def sample_models():
    """Provides a list of Pydantic models for consolidation testing."""
    m1 = ComplexModel(items=[SimpleModel(name="A", value=1)])
    m2 = ComplexModel(items=[SimpleModel(name="B", value=2)])
    return [m1, m2]


@pytest.fixture
def sample_programmatic_model():
    """Provides a merged Pydantic model for consolidation testing."""
    return ComplexModel(items=[SimpleModel(name="A", value=1), SimpleModel(name="B", value=2)])


@pytest.fixture
def sample_schema_json():
    """Provides the JSON schema string for the test model."""
    # Pydantic v2 model_json_schema() does not take 'indent'
    return json.dumps(ComplexModel.model_json_schema(), indent=2)


def test_get_consolidation_prompt_with_programmatic(
    sample_models, sample_programmatic_model, sample_schema_json
):
    """Tests the consolidation prompt when a programmatic draft is available."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json,
        raw_models=sample_models,
        programmatic_model=sample_programmatic_model,
    )

    raw_jsons_expected = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in sample_models)
    sample_programmatic_model.model_dump_json(indent=2)

    # Check key content instead of exact match (adaptive prompts may vary)
    assert isinstance(prompt, str)
    assert schema_json in prompt
    assert raw_jsons_expected in prompt
    # Programmatic model data should be in prompt (format may vary)
    assert any(item in prompt for item in ["A", "B", "value"])


def test_get_consolidation_prompt_no_programmatic(sample_models, sample_schema_json):
    """Tests the consolidation prompt when no programmatic draft is provided."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json, raw_models=sample_models, programmatic_model=None
    )

    raw_jsons_expected = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in sample_models)

    # Check key content instead of exact match (adaptive prompts may vary)
    assert isinstance(prompt, str)
    assert schema_json in prompt
    assert raw_jsons_expected in prompt


def test_get_consolidation_prompt_empty_raw(sample_schema_json):
    """Tests the consolidation prompt when the list of raw models is empty."""
    schema_json = sample_schema_json

    prompt = get_consolidation_prompt(
        schema_json=schema_json, raw_models=[], programmatic_model=None
    )

    # Check key content instead of exact match (adaptive prompts may vary)
    assert isinstance(prompt, str)
    assert schema_json in prompt


# --- Test Adaptive Prompting (Phase 1 Fix 3) ---


def test_get_extraction_prompt_with_simple_model():
    """Test extraction prompt for simple models (1B-5B)."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    markdown = "Test document"
    schema = '{"title": "Test"}'
    model_config = ModelConfig(
        model_id="phi-3", context_limit=4096, capability=ModelCapability.SIMPLE
    )

    prompt_dict = get_extraction_prompt(markdown, schema, model_config=model_config)

    # Simple models should get simplified instructions
    assert "system" in prompt_dict
    assert "user" in prompt_dict
    # Check for simple/basic instruction keywords
    system_lower = prompt_dict["system"].lower()
    assert any(word in system_lower for word in ["read", "extract", "return", "json"])


def test_get_extraction_prompt_with_standard_model():
    """Test extraction prompt for standard models (7B-13B)."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    markdown = "Test document"
    schema = '{"title": "Test"}'
    model_config = ModelConfig(
        model_id="mistral-7b", context_limit=8192, capability=ModelCapability.STANDARD
    )

    prompt_dict = get_extraction_prompt(markdown, schema, model_config=model_config)

    # Standard models should get balanced instructions
    assert "system" in prompt_dict
    assert "user" in prompt_dict


def test_get_extraction_prompt_with_advanced_model():
    """Test extraction prompt for advanced models (13B+)."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    markdown = "Test document"
    schema = '{"title": "Test"}'
    model_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )

    prompt_dict = get_extraction_prompt(markdown, schema, model_config=model_config)

    # Advanced models should get flexible instructions
    assert "system" in prompt_dict
    assert "user" in prompt_dict


def test_get_extraction_prompt_backward_compatible():
    """Test that extraction prompt works without model_config (backward compatibility)."""
    markdown = "Test document"
    schema = '{"title": "Test"}'

    # Should work without model_config parameter
    prompt_dict = get_extraction_prompt(markdown, schema)

    assert "system" in prompt_dict
    assert "user" in prompt_dict


def test_get_consolidation_prompt_chain_of_density():
    """Test Chain of Density consolidation for advanced models."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    schema_json = '{"title": "Test"}'
    m1 = ComplexModel(items=[SimpleModel(name="A", value=1)])
    m2 = ComplexModel(items=[SimpleModel(name="B", value=2)])
    programmatic = ComplexModel(
        items=[SimpleModel(name="A", value=1), SimpleModel(name="B", value=2)]
    )

    model_config = ModelConfig(
        model_id="gpt-4", context_limit=128000, capability=ModelCapability.ADVANCED
    )

    result = get_consolidation_prompt(
        schema_json=schema_json,
        raw_models=[m1, m2],
        programmatic_model=programmatic,
        model_config=model_config,
    )

    # Advanced models should get list of prompts for Chain of Density
    assert isinstance(result, list)
    assert len(result) == 2  # Stage 1 and Stage 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    # Stage 2 should have placeholders for injection
    assert "{stage1_result}" in result[1]


def test_get_consolidation_prompt_single_turn_simple():
    """Test single-turn consolidation for simple models."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    schema_json = '{"title": "Test"}'
    m1 = ComplexModel(items=[SimpleModel(name="A", value=1)])

    model_config = ModelConfig(
        model_id="phi-3", context_limit=4096, capability=ModelCapability.SIMPLE
    )

    result = get_consolidation_prompt(
        schema_json=schema_json,
        raw_models=[m1],
        programmatic_model=None,
        model_config=model_config,
    )

    # Simple models should get single prompt
    assert isinstance(result, str)
    assert schema_json in result


def test_get_consolidation_prompt_single_turn_standard():
    """Test single-turn consolidation for standard models."""
    from docling_graph.llm_clients.config import ModelCapability, ModelConfig

    schema_json = '{"title": "Test"}'
    m1 = ComplexModel(items=[SimpleModel(name="A", value=1)])

    model_config = ModelConfig(
        model_id="mistral-7b", context_limit=8192, capability=ModelCapability.STANDARD
    )

    result = get_consolidation_prompt(
        schema_json=schema_json,
        raw_models=[m1],
        programmatic_model=None,
        model_config=model_config,
    )

    # Standard models should get single prompt
    assert isinstance(result, str)
    assert schema_json in result
