"""
Unit tests for LLM backend.

Tests the LLM backend for direct extraction:
- extract_from_markdown() for direct extraction
- cleanup() for resource management
- QuantityWithUnit coercion and best-effort prune salvage
- Template-level relaxed QuantityWithUnit input (rheology template)
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.backends.llm_backend import LlmBackend


def _load_rheology_quantity_with_unit() -> type[BaseModel] | None:
    """Load QuantityWithUnit from docs/examples/templates/rheology_research.py."""
    repo_root = Path(__file__).resolve().parents[5]
    template_path = repo_root / "docs" / "examples" / "templates" / "rheology_research.py"
    if not template_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("rheology_research", template_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rheology_research"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "QuantityWithUnit", None)


# Simple Pydantic model for testing
class MockTemplate(BaseModel):
    name: str
    age: int


# Minimal QuantityWithUnit and template for coercion tests (model name must be QuantityWithUnit)
class QuantityWithUnit(BaseModel):
    numeric_value: float | None = None
    text_value: str | None = None


class TemplateWithQuantity(BaseModel):
    gap: QuantityWithUnit | None = None


# For prune salvage: nested structure with optional invalid field
class Inner(BaseModel):
    a: str
    b: int | None = None


class Outer(BaseModel):
    name: str
    inner: Inner | None = None


# Fixtures
@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.__class__.__name__ = "MockLlmClient"
    client.context_limit = 8000
    return client


@pytest.fixture
def llm_backend(mock_llm_client):
    """Create an LlmBackend instance with mock client."""
    return LlmBackend(llm_client=mock_llm_client)


class TestInitialization:
    """Test backend initialization."""

    def test_init_with_client(self, llm_backend, mock_llm_client):
        """Test that backend initializes with the client."""
        assert llm_backend.client == mock_llm_client

    def test_init_logs_client_info(self, mock_llm_client):
        """Test that initialization logs client information."""
        # Should not raise any errors
        backend = LlmBackend(llm_client=mock_llm_client)
        assert backend.client == mock_llm_client


class TestExtractFromMarkdown:
    """Test extract_from_markdown() method (direct extraction)."""

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_successful_extraction(self, mock_get_prompt, llm_backend, mock_llm_client):
        """Test successful extraction and validation."""
        markdown = "This is a test document."
        context = "test context"
        expected_json = {"name": "Test", "age": 30}
        schema_json = json.dumps(MockTemplate.model_json_schema(), indent=2)

        # Configure mocks
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

        # Verify prompt generation
        mock_get_prompt.assert_called_once()
        call_kwargs = mock_get_prompt.call_args[1]
        assert call_kwargs["markdown_content"] == markdown
        assert call_kwargs["schema_json"] == schema_json
        assert not call_kwargs["is_partial"]

        # Verify LLM call
        mock_llm_client.get_json_response.assert_called_with(
            prompt={"system": "sys", "user": "user"}, schema_json=schema_json
        )

    def test_empty_markdown_returns_none(self, llm_backend):
        """Test that empty or whitespace-only markdown returns None."""
        result_empty = llm_backend.extract_from_markdown(markdown="", template=MockTemplate)
        result_whitespace = llm_backend.extract_from_markdown(
            markdown="   \n ", template=MockTemplate
        )

        assert result_empty is None
        assert result_whitespace is None

    def test_no_json_returned(self, llm_backend, mock_llm_client):
        """Test when LLM client returns no valid JSON."""
        mock_llm_client.get_json_response.return_value = None

        result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)

        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.rich_print")
    def test_validation_error(self, mock_rich_print, llm_backend, mock_llm_client):
        """Test when LLM returns JSON that fails Pydantic validation."""
        # Missing required field 'age'
        invalid_json = {"name": "Test Only"}
        mock_llm_client.get_json_response.return_value = invalid_json

        result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)

        # Should fail validation and return None
        assert result is None

        # Check that validation error was printed
        mock_rich_print.assert_any_call(
            "[blue][LlmBackend][/blue] [yellow]Validation Error for document:[/yellow]"
        )

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_partial_extraction(self, mock_get_prompt, llm_backend, mock_llm_client):
        """Test extraction with is_partial=True."""
        markdown = "This is a page."
        expected_json = {"name": "Test", "age": 30}

        mock_llm_client.get_json_response.return_value = expected_json
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        result = llm_backend.extract_from_markdown(
            markdown=markdown, template=MockTemplate, is_partial=True
        )

        assert isinstance(result, MockTemplate)

        # Verify is_partial was passed to prompt generation
        call_kwargs = mock_get_prompt.call_args[1]
        assert call_kwargs["is_partial"] is True

    def test_exception_handling(self, llm_backend, mock_llm_client):
        """Test that exceptions are handled gracefully."""
        mock_llm_client.get_json_response.side_effect = Exception("Test error")

        result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)

        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.StagedOrchestrator")
    def test_staged_contract_uses_multi_pass_flow(self, mock_orchestrator_cls, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="staged")
        orchestrator = mock_orchestrator_cls.return_value
        orchestrator.extract.return_value = {"name": "Alice", "age": 21}

        result = backend.extract_from_markdown(markdown="x", template=MockTemplate)

        assert result is not None
        assert result.name == "Alice"
        assert result.age == 21
        orchestrator.extract.assert_called_once()

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_staged_contract_partial_extraction_falls_back_to_direct(
        self, mock_get_prompt, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="staged")
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}
        mock_llm_client.get_json_response.return_value = {"name": "Test", "age": 30}

        result = backend.extract_from_markdown(
            markdown="partial page", template=MockTemplate, is_partial=True
        )

        assert result is not None
        mock_get_prompt.assert_called_once()


class TestQuantityCoercionAndPruneSalvage:
    """Test best-effort validation: QuantityWithUnit coercion and prune salvage."""

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_quantity_coercion_scalar_returns_valid_model(
        self, mock_get_prompt, llm_backend, mock_llm_client
    ):
        """Scalar for QuantityWithUnit field is coerced to object; extraction returns valid model."""
        # LLM returns scalar for gap (schema expects QuantityWithUnit object)
        mock_llm_client.get_json_response.return_value = {"gap": 1.0}
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        result = llm_backend.extract_from_markdown(
            markdown="Gap 1 mm.",
            template=TemplateWithQuantity,
            context="document",
        )

        assert result is not None
        assert isinstance(result, TemplateWithQuantity)
        assert result.gap is not None
        assert result.gap.numeric_value == 1.0

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_prune_salvage_returns_valid_model_with_remaining_content(
        self, mock_get_prompt, llm_backend, mock_llm_client
    ):
        """Invalid nested field is pruned; valid model returned with remaining content."""
        # inner.b is invalid (str "bad" instead of int); pruning should remove b and keep a
        mock_llm_client.get_json_response.return_value = {
            "name": "Test",
            "inner": {"a": "ok", "b": "bad"},
        }
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        result = llm_backend.extract_from_markdown(
            markdown="Some content",
            template=Outer,
            context="document",
        )

        assert result is not None
        assert isinstance(result, Outer)
        assert result.name == "Test"
        assert result.inner is not None
        assert result.inner.a == "ok"
        assert result.inner.b is None


class TestFillMissingRequiredFieldsStableSyntheticIds:
    """Stable synthetic ID generation: same entity content => same generated ID."""

    def test_content_fingerprint_deterministic(self, llm_backend):
        """_content_fingerprint is deterministic for same entity."""
        entity = {"objective": "Study colloidal stability", "experiments": [{"experiment_id": "E1"}]}
        fp1 = llm_backend._content_fingerprint(entity, exclude_keys=set())
        fp2 = llm_backend._content_fingerprint(entity, exclude_keys=set())
        assert fp1 == fp2

    def test_fill_missing_study_id_same_content_same_id(self, llm_backend):
        """Filling missing study_id for two entities with same content yields same synthetic ID."""
        data1 = {
            "studies": [
                {"objective": "Same objective", "experiments": [{"experiment_id": "EXP-1"}]},
                {"objective": "Same objective", "experiments": [{"experiment_id": "EXP-1"}]},
            ]
        }
        errors1 = [
            {"type": "missing", "loc": ("studies", 0, "study_id")},
            {"type": "missing", "loc": ("studies", 1, "study_id")},
        ]
        llm_backend._fill_missing_required_fields(data1, errors1)
        id0_a = data1["studies"][0]["study_id"]
        id1_a = data1["studies"][1]["study_id"]
        # Same content => same fingerprint => same synthetic ID
        assert id0_a == id1_a
        assert id0_a.startswith("STUDY-")

    def test_fill_missing_study_id_different_content_different_id(self, llm_backend):
        """Filling missing study_id for entities with different content yields different IDs."""
        data = {
            "studies": [
                {"objective": "Objective A", "experiments": []},
                {"objective": "Objective B", "experiments": []},
            ]
        }
        errors = [
            {"type": "missing", "loc": ("studies", 0, "study_id")},
            {"type": "missing", "loc": ("studies", 1, "study_id")},
        ]
        llm_backend._fill_missing_required_fields(data, errors)
        assert data["studies"][0]["study_id"] != data["studies"][1]["study_id"]
        assert data["studies"][0]["study_id"].startswith("STUDY-")
        assert data["studies"][1]["study_id"].startswith("STUDY-")


class TestRheologyQuantityWithUnitRelaxedInput:
    """Test template-level coercion: rheology QuantityWithUnit accepts scalars and strings."""

    @pytest.fixture(scope="class")
    def rheology_quantity_with_unit(self) -> type[BaseModel] | None:
        """Load QuantityWithUnit from rheology_research template if available."""
        return _load_rheology_quantity_with_unit()

    def test_scalar_int_normalizes_to_numeric_value(self, rheology_quantity_with_unit):
        """Scalar int is accepted and normalized to numeric_value."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        m = rheology_quantity_with_unit.model_validate(1)
        assert m.numeric_value == 1.0
        assert m.text_value is None

    def test_scalar_float_normalizes_to_numeric_value(self, rheology_quantity_with_unit):
        """Scalar float is accepted and normalized to numeric_value."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        m = rheology_quantity_with_unit.model_validate(1.0)
        assert m.numeric_value == 1.0
        assert m.text_value is None

    def test_string_numeric_only_normalizes_to_numeric_value(self, rheology_quantity_with_unit):
        """String that is only a number (e.g. '0.95') normalizes to numeric_value."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        m = rheology_quantity_with_unit.model_validate("0.95")
        assert m.numeric_value == 0.95
        assert m.text_value is None

    def test_string_numeric_with_unit_normalizes_to_numeric_and_unit(
        self, rheology_quantity_with_unit
    ):
        """String like '25 C' normalizes to numeric_value and unit."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        m = rheology_quantity_with_unit.model_validate("25 C")
        assert m.numeric_value == 25.0
        assert m.unit == "C"
        assert m.text_value is None

    def test_string_qualitative_normalizes_to_text_value(self, rheology_quantity_with_unit):
        """Qualitative string (no leading number) normalizes to text_value."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        m = rheology_quantity_with_unit.model_validate("High")
        assert m.text_value == "High"
        assert m.numeric_value is None

    def test_dict_unchanged(self, rheology_quantity_with_unit):
        """Dict input is passed through unchanged by the before validator."""
        if rheology_quantity_with_unit is None:
            pytest.skip("rheology_research template not found")
        d = {"numeric_value": 40.0, "unit": "mm"}
        m = rheology_quantity_with_unit.model_validate(d)
        assert m.numeric_value == 40.0
        assert m.unit == "mm"


class TestJSONRepair:
    """Test JSON repair functionality"""

    def test_removal_of_invalid_control_characters(self, llm_backend):
        """Test removal of invalid control characters."""
        # Test with various control characters
        raw_json = '{"key": "value\x00\x01\x02"}'
        repaired = llm_backend._repair_json(raw_json)

        # Control chars should be removed
        assert "\x00" not in repaired
        assert "\x01" not in repaired
        assert "\x02" not in repaired
        assert "value" in repaired

    def test_removal_of_trailing_commas(self, llm_backend):
        """Test removal of trailing commas."""
        # Trailing comma before closing brace
        raw_json = '{"key": "value",}'
        repaired = llm_backend._repair_json(raw_json)
        assert repaired == '{"key": "value"}'

        # Trailing comma before closing bracket
        raw_json = '["item1", "item2",]'
        repaired = llm_backend._repair_json(raw_json)
        assert repaired == '["item1", "item2"]'

    def test_bracket_balancing_add_missing_closing(self, llm_backend):
        """Test adding missing closing brackets."""
        # Missing closing brace
        raw_json = '{"key": "value"'
        repaired = llm_backend._repair_json(raw_json)
        assert repaired == '{"key": "value"}'

        # Missing closing bracket
        raw_json = '["item1", "item2"'
        repaired = llm_backend._repair_json(raw_json)
        assert repaired == '["item1", "item2"]'

        # Missing multiple closing brackets
        raw_json = '{"array": [1, 2, 3'
        repaired = llm_backend._repair_json(raw_json)
        assert repaired.count("]") == 1
        assert repaired.count("}") == 1

    def test_valid_json_unchanged(self, llm_backend):
        """Test that valid JSON is unchanged."""
        valid_json = '{"key": "value", "array": [1, 2, 3], "nested": {"a": 1}}'
        repaired = llm_backend._repair_json(valid_json)

        # Should be unchanged
        assert repaired == valid_json

    def test_preserve_valid_control_characters(self, llm_backend):
        """Test that valid control characters (newline, tab, CR) are preserved."""
        raw_json = '{"key": "line1\nline2\ttabbed\rcarriage"}'
        repaired = llm_backend._repair_json(raw_json)

        # Valid control chars should be preserved
        assert "\n" in repaired
        assert "\t" in repaired
        assert "\r" in repaired

    def test_complex_repair_scenario(self, llm_backend):
        """Test complex repair with multiple issues."""
        # Multiple issues: control chars, trailing comma, missing bracket
        raw_json = '{"key": "value\x00", "array": [1, 2,]'
        repaired = llm_backend._repair_json(raw_json)

        # Should be valid JSON after repair
        parsed = json.loads(repaired)
        assert "key" in parsed
        assert "array" in parsed
        assert parsed["array"] == [1, 2]


class TestCleanup:
    """Test cleanup() method."""

    @patch("gc.collect")
    def test_cleanup_with_client_cleanup_method(self, mock_gc_collect, mock_llm_client):
        """Test cleanup when client has cleanup method."""
        # Add cleanup method to client
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

    @patch("gc.collect")
    def test_cleanup_without_client_cleanup_method(self, mock_gc_collect):
        """Test cleanup when client doesn't have cleanup method."""
        # Create client without cleanup method
        mock_client = MagicMock()
        mock_client.__class__.__name__ = "MockClient"
        mock_client.context_limit = 8192
        # Explicitly remove cleanup if it exists
        if hasattr(mock_client, "cleanup"):
            delattr(mock_client, "cleanup")

        backend = LlmBackend(llm_client=mock_client)

        # Should not raise error
        backend.cleanup()

        # Client should be deleted
        assert not hasattr(backend, "client")

        # GC should still be called
        mock_gc_collect.assert_called_once()
