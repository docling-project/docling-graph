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
from typing import Any, NoReturn
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.backends.llm_backend import LlmBackend
from docling_graph.exceptions import ClientError


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


class LargeTemplate(BaseModel):
    f1: str | None = None
    f2: str | None = None
    f3: str | None = None
    f4: str | None = None
    f5: str | None = None
    f6: str | None = None
    f7: str | None = None
    f8: str | None = None
    f9: str | None = None
    f10: str | None = None
    f11: str | None = None
    f12: str | None = None


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
        called_kwargs = mock_llm_client.get_json_response.call_args.kwargs
        assert called_kwargs["prompt"] == {"system": "sys", "user": "user"}
        assert called_kwargs["schema_json"] == schema_json
        assert called_kwargs["structured_output"] is True

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

    def test_validation_error(self, llm_backend, mock_llm_client, caplog):
        """Test when LLM returns JSON that fails Pydantic validation."""
        import logging

        # Missing required field 'age'
        invalid_json = {"name": "Test Only"}
        mock_llm_client.get_json_response.return_value = invalid_json

        with caplog.at_level(logging.WARNING, logger="docling_graph"):
            result = llm_backend.extract_from_markdown(
                markdown="Some content", template=MockTemplate
            )

        # Should fail validation and return None
        assert result is None

        # Check that validation error was logged
        assert any("Validation error for document" in r.getMessage() for r in caplog.records)

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

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_structured_failure_falls_back_to_legacy_prompt_schema(
        self, mock_get_prompt, llm_backend, mock_llm_client
    ):
        mock_get_prompt.side_effect = [
            {"system": "sys", "user": "compact"},
            {"system": "sys", "user": "legacy"},
        ]
        mock_llm_client.get_json_response.side_effect = [
            ClientError("structured failed"),
            {"name": "Test", "age": 30},
        ]
        result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)
        assert result is not None
        assert mock_llm_client.get_json_response.call_count == 2
        first = mock_llm_client.get_json_response.call_args_list[0].kwargs
        second = mock_llm_client.get_json_response.call_args_list[1].kwargs
        assert first["structured_output"] is True
        assert second["structured_output"] is False
        assert llm_backend.last_call_diagnostics["structured_attempted"] is True
        assert llm_backend.last_call_diagnostics["structured_failed"] is True
        assert llm_backend.last_call_diagnostics["fallback_used"] is True

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_sparse_structured_result_triggers_legacy_retry(
        self, mock_get_prompt, llm_backend, mock_llm_client
    ):
        mock_get_prompt.side_effect = [
            {"system": "sys", "user": "compact"},
            {"system": "sys", "user": "legacy"},
        ]
        sparse = {"f1": "only one"}
        rich = {"f1": "a", "f2": "b", "f3": "c", "f4": "d", "f5": "e"}
        mock_llm_client.get_json_response.side_effect = [sparse, rich]
        mock_llm_client.last_call_diagnostics = {"raw_response": '{"f1":"only one"}'}
        llm_backend.trace_data = MagicMock()  # Simulate debug mode trace capture enabled
        markdown = "x" * 1200
        result = llm_backend.extract_from_markdown(markdown=markdown, template=LargeTemplate)
        assert result is not None
        assert result.f5 == "e"
        assert mock_llm_client.get_json_response.call_count == 2
        assert llm_backend.last_call_diagnostics["fallback_used"] is True
        assert "structured_primary_attempt_parsed_json" in llm_backend.last_call_diagnostics

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_sparse_structured_result_does_not_retry_when_check_disabled(
        self, mock_get_prompt, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, structured_sparse_check=False)
        mock_get_prompt.return_value = {"system": "sys", "user": "compact"}
        sparse = {"f1": "only one"}
        mock_llm_client.get_json_response.return_value = sparse
        markdown = "x" * 1200
        result = backend.extract_from_markdown(markdown=markdown, template=LargeTemplate)
        assert result is not None
        assert mock_llm_client.get_json_response.call_count == 1
        assert backend.last_call_diagnostics["fallback_used"] is False


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
        entity = {
            "objective": "Study colloidal stability",
            "experiments": [{"experiment_id": "E1"}],
        }
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
        # Same content => same fingerprint => same synthetic ID (prefix shortened generically from field name)
        assert id0_a == id1_a
        assert id0_a.startswith("STUD-")

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
        assert data["studies"][0]["study_id"].startswith("STUD-")
        assert data["studies"][1]["study_id"].startswith("STUD-")


class TestIdentityGuardSalvage:
    """Salvage must never fabricate a value for a graph_id_fields member: the
    offending instance is dropped instead (phantom-hub guard, RC3)."""

    def _template(self) -> type:
        from pydantic import ConfigDict

        class Garantie(BaseModel):
            model_config = ConfigDict(graph_id_fields=["nom"])
            nom: str
            description: str | None = None

        class Root(BaseModel):
            title: str | None = None
            garanties: list[Garantie] = Field(default_factory=list)

        return Root

    def test_instance_missing_required_identity_is_dropped_not_blanked(self, llm_backend):
        root_model = self._template()
        data = {
            "title": "T",
            "garanties": [
                {"nom": "Vol", "description": "d1"},
                {"description": "bucket junk with no name"},
                {"nom": "Incendie"},
            ],
        }
        result = llm_backend._validate_extraction(data, root_model, context="test")
        assert result is not None
        noms = [g.nom for g in result.garanties]
        assert noms == ["Vol", "Incendie"]
        assert "" not in noms  # no blank-identity phantom

    def test_multiple_drops_in_one_list_do_not_shift_each_other(self, llm_backend):
        root_model = self._template()
        data = {
            "garanties": [
                {"description": "junk 1"},
                {"nom": "Keep A"},
                {"description": "junk 2"},
                {"nom": "Keep B"},
                {"description": "junk 3"},
            ]
        }
        result = llm_backend._validate_extraction(data, root_model, context="test")
        assert result is not None
        assert [g.nom for g in result.garanties] == ["Keep A", "Keep B"]

    def test_non_identity_required_fields_still_get_filled(self, llm_backend):
        """The guard is identity-specific: other required strings keep the
        legacy fill-with-generated-value salvage."""
        from pydantic import ConfigDict

        class Item(BaseModel):
            model_config = ConfigDict(graph_id_fields=["item_id"])
            item_id: str
            label: str  # required NON-identity

        class Root(BaseModel):
            items: list[Item] = Field(default_factory=list)

        data = {"items": [{"item_id": "IT-1"}]}
        result = llm_backend._validate_extraction(data, Root, context="test")
        assert result is not None
        assert result.items[0].item_id == "IT-1"
        assert result.items[0].label == ""  # legacy salvage fill

    def test_root_document_is_never_dropped(self, llm_backend):
        """A missing root-level required field keeps legacy behavior (filled),
        never deletes the whole document."""
        from pydantic import ConfigDict

        class Root(BaseModel):
            model_config = ConfigDict(graph_id_fields=["reference_document"])
            reference_document: str

        result = llm_backend._validate_extraction({}, Root, context="test")
        assert result is not None


class TestCoerceStringTypeErrors:
    """Test that int/float/bool in string fields are coerced so validation can pass."""

    def test_validate_extraction_coerces_study_id_int_to_string(self, llm_backend):
        """When a projected study_id is int (e.g. 3), coercion pass converts it to '3' and validation passes."""
        from pydantic import BaseModel, Field

        class Study(BaseModel):
            study_id: str = Field(description="ID")
            objective: str | None = None

        class Root(BaseModel):
            studies: list[Study] = Field(default_factory=list)

        data = {"studies": [{"study_id": 3, "objective": "Analyze flow curves"}]}
        result = llm_backend._validate_extraction(data, Root, context="test")
        assert result is not None
        assert len(result.studies) == 1
        assert result.studies[0].study_id == "3"
        assert result.studies[0].objective == "Analyze flow curves"

    def test_validate_extraction_coerces_string_field_from_list_or_dict(self, llm_backend):
        """When a string field (e.g. name) is list/dict (LLM misuse), coerce to string and keep list items."""
        from pydantic import BaseModel, Field

        class Item(BaseModel):
            name: str = Field(description="Identity")

        class Root(BaseModel):
            items: list[Item] = Field(default_factory=list)

        # First item: name is list of dicts (common LLM mistake); second: name is dict with nom key
        data = {
            "items": [
                {"name": [{"description": "Long text", "nom": "First"}]},
                {"name": {"nom": "Second", "extra": 1}},
                {"name": "Third"},
            ]
        }
        result = llm_backend._validate_extraction(data, Root, context="test")
        assert result is not None
        assert len(result.items) == 3
        assert result.items[0].name == "First"
        assert result.items[1].name == "Second"
        assert result.items[2].name == "Third"

    def test_coerce_string_fallback_when_list_dict_yields_no_string(self, llm_backend):
        """When schema expects string but value is list/dict with no extractable string, use '' so validation passes."""
        from pydantic import BaseModel, Field

        class Dataset(BaseModel):
            dataset_id: str = Field(description="ID")

        class Root(BaseModel):
            datasets: list[Dataset] = Field(default_factory=list)

        data = {"datasets": [{"dataset_id": [{}]}]}
        result = llm_backend._validate_extraction(data, Root, context="test")
        assert result is not None
        assert len(result.datasets) == 1
        assert result.datasets[0].dataset_id == ""


class TestCoerceListTypeErrors:
    """Test that scalar in list field is coerced to single-element list so validation can pass."""

    def test_validate_extraction_coerces_statut_occupation_string_to_list(self, llm_backend):
        """When statut_occupation is returned as a string for a list[str] field, coercion wraps it in a list."""
        from pydantic import BaseModel, Field

        class Offre(BaseModel):
            nom: str = Field(description="Name")
            statut_occupation: list[str] = Field(default_factory=list, description="Status")

        class Root(BaseModel):
            offres: list[Offre] = Field(default_factory=list)

        data = {
            "offres": [
                {"nom": "PNO", "statut_occupation": "Propriétaire Non Occupant"},
            ]
        }
        result = llm_backend._validate_extraction(data, Root, context="test")
        assert result is not None
        assert len(result.offres) == 1
        assert result.offres[0].nom == "PNO"
        assert result.offres[0].statut_occupation == ["Propriétaire Non Occupant"]


class TestRheologyQuantityWithUnitRelaxedInput:
    """Test template-level coercion: rheology QuantityWithUnit accepts scalars and strings."""

    # Function-scoped: class-scoped fixtures as instance methods are deprecated
    # (PytestRemovedIn10Warning) and the loader is import-cached anyway.
    @pytest.fixture
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

    def test_generate_returns_empty_response_on_exception(self, mock_llm_client):
        """When client.get_json_response raises in generate(), return EmptyResponse with text '{}'."""
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.get_json_response.side_effect = RuntimeError("network error")
        response = backend.generate(system_prompt="s", user_prompt="u")
        assert response.text == "{}"

    def test_cleanup_handles_client_cleanup_exception(self, mock_llm_client):
        """When client.cleanup() raises, backend catches and does not propagate (exception path covered)."""
        mock_llm_client.cleanup = MagicMock(side_effect=RuntimeError("cleanup failed"))
        backend = LlmBackend(llm_client=mock_llm_client)
        backend.cleanup()
        mock_llm_client.cleanup.assert_called_once()
        # Exception was caught so no crash; del self.client is not reached when cleanup raises


class TestDirectExtractionTraceAndDiagnostics:
    """Direct extraction path: trace_data, last_call_diagnostics merge, sparse fallback, truncation retry, gleaning."""

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_extraction_with_trace_data_and_client_diagnostics(
        self, mock_get_prompt, mock_llm_client
    ):
        """Direct path: success with trace_data set and client last_call_diagnostics merged (837-866)."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            structured_output=True,
        )
        backend.trace_data = MagicMock()
        mock_get_prompt.return_value = {"system": "s", "user": "u"}
        mock_llm_client.get_json_response.return_value = {"name": "A", "age": 1}
        mock_llm_client.last_call_diagnostics = {
            "provider": "openai",
            "model": "gpt-4",
            "structured_attempted": True,
        }
        result = backend.extract_from_markdown(markdown="doc", template=MockTemplate, context="doc")
        assert result is not None
        assert result.name == "A"
        assert backend.last_call_diagnostics.get("provider") == "openai"
        assert backend.last_call_diagnostics.get("model") == "gpt-4"
        assert backend.last_call_diagnostics.get("structured_attempted") is True

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_extraction_client_error_emits_trace_and_captures_raw(
        self, mock_get_prompt, mock_llm_client
    ):
        """Direct path: ClientError triggers emit and primary_diag/raw_value branches (728-733, 737-755, 750-760)."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            structured_output=True,
        )
        backend.trace_data = MagicMock()
        mock_get_prompt.side_effect = [
            {"system": "s", "user": "u"},
            {"system": "s", "user": "legacy"},
        ]
        mock_llm_client.get_json_response.side_effect = [
            ClientError("structured failed"),
            {"name": "B", "age": 2},
        ]
        mock_llm_client.last_call_diagnostics = {"raw_response": '{"name":"B","age":2}'}
        result = backend.extract_from_markdown(markdown="doc", template=MockTemplate, context="doc")
        assert result is not None
        backend.trace_data.emit.assert_called()
        call_args_list = [c[0][0] for c in backend.trace_data.emit.call_args_list]
        assert "structured_output_fallback_triggered" in call_args_list
        for call in backend.trace_data.emit.call_args_list:
            if call[0][0] == "structured_output_fallback_triggered" and len(call[0]) >= 3:
                payload = call[0][2]
                assert "error_message" in payload
                assert payload["error_message"] == "structured failed"
                assert "details" in payload
                break
        else:
            pytest.fail(
                "Expected structured_output_fallback_triggered emit with error_message in payload"
            )

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_extraction_sparse_fallback_emits_trace(self, mock_get_prompt, mock_llm_client):
        """Direct path: sparse structured result triggers trace emit and legacy retry (791-827)."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            structured_output=True,
            structured_sparse_check=True,
        )
        backend.trace_data = MagicMock()
        mock_get_prompt.side_effect = [
            {"system": "s", "user": "compact"},
            {"system": "s", "user": "legacy"},
        ]
        sparse = {"f1": "only"}
        rich = {"f1": "a", "f2": "b", "f3": "c", "f4": "d", "f5": "e"}
        mock_llm_client.get_json_response.side_effect = [sparse, rich]
        mock_llm_client.last_call_diagnostics = {}
        markdown = "x" * 1200
        result = backend.extract_from_markdown(
            markdown=markdown, template=LargeTemplate, context="doc"
        )
        assert result is not None
        emit_calls = [c[0][0] for c in backend.trace_data.emit.call_args_list]
        assert "structured_output_fallback_triggered" in emit_calls
        # Second call args should include reason SparseStructuredOutput
        for call in backend.trace_data.emit.call_args_list:
            if len(call[0]) >= 3 and isinstance(call[0][2], dict):
                if call[0][2].get("reason") == "SparseStructuredOutput":
                    break
        else:
            pytest.fail("Expected emit with reason SparseStructuredOutput")

    def test_call_prompt_truncation_retry_uses_details_max_tokens(self, mock_llm_client):
        """When max_tokens is not passed, retry uses context_max from details['max_tokens']."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            dense_config={"retry_on_truncation": True},
        )
        context = "custom_context_xyz"
        details = {"truncated": True, "max_tokens": 512}
        mock_llm_client.get_json_response.side_effect = [
            ClientError("truncated", details=details),
            {"id": "ok"},
        ]
        out = backend._call_prompt({"system": "s", "user": "u"}, "{}", context)
        assert out == {"id": "ok"}
        assert mock_llm_client.get_json_response.call_count == 2

    def test_call_prompt_allow_truncation_retry_false_skips_escalation(self, mock_llm_client):
        """P2 (split-before-escalate): the dense orchestrator's _dense_llm wrapper
        always calls with structured_output_override=False (legacy-only, matching
        dense's single-call-path contract) and, for a multi-chunk batch, also
        allow_truncation_retry=False so it can split instead of escalating output
        tokens. _call_prompt must honor that and never attempt the escalation
        retry call, even though the client reports truncation."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            dense_config={"retry_on_truncation": True},
        )
        details = {"truncated": True, "max_tokens": 512}
        mock_llm_client.get_json_response.side_effect = [
            ClientError("truncated", details=details),
        ]
        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
            allow_truncation_retry=False,
        )
        assert out is None
        # No escalation retry attempted: exactly the one (truncated) call.
        assert mock_llm_client.get_json_response.call_count == 1

    def test_call_prompt_allow_truncation_retry_true_still_escalates(self, mock_llm_client):
        """Contrast case: with allow_truncation_retry left at its default (True),
        the same truncation DOES trigger the escalation retry call."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            dense_config={"retry_on_truncation": True},
        )
        details = {"truncated": True, "max_tokens": 512}
        mock_llm_client.get_json_response.side_effect = [
            ClientError("truncated", details=details),
            {"id": "ok"},
        ]
        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
        )
        assert out == {"id": "ok"}
        assert mock_llm_client.get_json_response.call_count == 2

    @patch("docling_graph.core.extractors.backends.llm_backend.run_gleaning_pass_direct")
    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_extraction_gleaning_pass_invoked(
        self, mock_get_prompt, mock_gleaning, mock_llm_client
    ):
        """Gleaning block (871+) runs when gleaning_enabled and full-doc direct extraction."""
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            dense_config={"gleaning_enabled": True},
        )
        mock_get_prompt.return_value = {"system": "s", "user": "u"}
        mock_llm_client.get_json_response.side_effect = [
            {"name": "Pre", "age": 10},
            {"name": "Pre", "age": 10, "description": "Gleaned desc"},
        ]
        mock_gleaning.return_value = {"name": "Pre", "age": 10, "description": "Gleaned desc"}
        result = backend.extract_from_markdown(
            markdown="Full doc content.", template=MockTemplate, context="doc"
        )
        assert result is not None
        mock_gleaning.assert_called_once()


# ============================================================================
# Streaming Configuration Tests
# ============================================================================


class TestStreamingConfiguration:
    """Test that streaming configuration flows through the backend correctly."""

    def test_llm_backend_uses_streaming_when_enabled(self, mock_llm_client):
        """Test that backend uses streaming method when enabled."""
        # Setup mock client with streaming enabled
        mock_llm_client.streaming = True
        mock_llm_client.get_json_response_stream.return_value = iter([{"name": "Alice", "age": 30}])

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call _get_json_response
        result = backend._get_json_response(
            prompt="Extract data",
            schema_json='{"type": "object"}',
        )

        # Verify streaming method was called
        mock_llm_client.get_json_response_stream.assert_called_once()
        mock_llm_client.get_json_response.assert_not_called()
        assert result == {"name": "Alice", "age": 30}

    def test_llm_backend_uses_non_streaming_by_default(self, mock_llm_client):
        """Test that backend uses non-streaming method by default."""
        # Setup mock client with streaming disabled (default)
        mock_llm_client.streaming = False
        mock_llm_client.get_json_response.return_value = {"name": "Bob", "age": 25}

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call _get_json_response
        result = backend._get_json_response(
            prompt="Extract data",
            schema_json='{"type": "object"}',
        )

        # Verify non-streaming method was called
        mock_llm_client.get_json_response.assert_called_once()
        mock_llm_client.get_json_response_stream.assert_not_called()
        assert result == {"name": "Bob", "age": 25}

    def test_llm_backend_handles_streaming_iterator(self, mock_llm_client):
        """Test that backend properly extracts result from streaming iterator."""
        # Setup mock client with streaming enabled
        mock_llm_client.streaming = True
        expected_result = {"entities": ["entity1", "entity2"], "count": 2}
        mock_llm_client.get_json_response_stream.return_value = iter([expected_result])

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call _get_json_response
        result = backend._get_json_response(
            prompt="Extract entities",
            schema_json='{"type": "object"}',
        )

        # Verify next() was used to get result from iterator
        assert result == expected_result

    def test_llm_backend_streaming_with_structured_output(self, mock_llm_client):
        """Test that streaming works with structured output enabled."""
        # Setup mock client with streaming enabled
        mock_llm_client.streaming = True
        mock_llm_client.get_json_response_stream.return_value = iter([{"data": "value"}])

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call with structured_output=True
        result = backend._get_json_response(
            prompt="Extract",
            schema_json='{"type": "object"}',
            structured_output=True,
        )

        # Verify streaming method was called with structured_output
        call_kwargs = mock_llm_client.get_json_response_stream.call_args.kwargs
        assert call_kwargs["structured_output"] is True
        assert result == {"data": "value"}

    def test_llm_backend_streaming_with_array_response(self, mock_llm_client):
        """Test that streaming works with array response type."""
        # Setup mock client with streaming enabled
        mock_llm_client.streaming = True
        expected_array = [{"id": 1}, {"id": 2}]
        mock_llm_client.get_json_response_stream.return_value = iter([expected_array])

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call with response_top_level="array"
        result = backend._get_json_response(
            prompt="Extract list",
            schema_json='{"type": "array"}',
            response_top_level="array",
        )

        # Verify streaming method was called with correct parameters
        call_kwargs = mock_llm_client.get_json_response_stream.call_args.kwargs
        assert call_kwargs["response_top_level"] == "array"
        assert result == expected_array

    def test_llm_backend_without_streaming_attribute(self, mock_llm_client):
        """Test that backend handles clients without streaming attribute."""
        # Remove streaming attribute to simulate older client
        if hasattr(mock_llm_client, "streaming"):
            delattr(mock_llm_client, "streaming")

        mock_llm_client.get_json_response.return_value = {"fallback": "works"}

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call _get_json_response
        result = backend._get_json_response(
            prompt="Extract",
            schema_json='{"type": "object"}',
        )

        # Should fall back to non-streaming method
        mock_llm_client.get_json_response.assert_called_once()
        assert result == {"fallback": "works"}

    def test_llm_backend_streaming_passes_all_parameters(self, mock_llm_client):
        """Test that all parameters are passed through to streaming method."""
        # Setup mock client with streaming enabled
        mock_llm_client.streaming = True
        mock_llm_client.get_json_response_stream.return_value = iter([{"result": "ok"}])

        backend = LlmBackend(llm_client=mock_llm_client)

        # Call with all parameters
        result = backend._get_json_response(
            prompt={"system": "You are helpful", "user": "Extract data"},
            schema_json='{"type": "object", "properties": {}}',
            structured_output=False,
            response_top_level="object",
            response_schema_name="custom_schema",
        )

        # Verify all parameters were passed
        call_kwargs = mock_llm_client.get_json_response_stream.call_args.kwargs
        assert call_kwargs["prompt"] == {"system": "You are helpful", "user": "Extract data"}
        assert call_kwargs["schema_json"] == '{"type": "object", "properties": {}}'
        assert call_kwargs["structured_output"] is False
        assert call_kwargs["response_top_level"] == "object"
        assert call_kwargs["response_schema_name"] == "custom_schema"
        assert result == {"result": "ok"}


class TestTruncationPropagation:
    """_call_prompt surfaces the client's truncation flag to callers."""

    def test_call_prompt_propagates_truncated(self, llm_backend, mock_llm_client):
        mock_llm_client.get_json_response.return_value = {"ok": True}
        mock_llm_client.last_call_diagnostics = {"truncated": True}
        diag: dict = {}
        out = llm_backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
            _diagnostics_out=diag,
        )
        assert out == {"ok": True}
        assert diag.get("truncated") is True

    def test_call_prompt_truncated_false_when_client_not_truncated(
        self, llm_backend, mock_llm_client
    ):
        mock_llm_client.get_json_response.return_value = {"ok": True}
        mock_llm_client.last_call_diagnostics = {"truncated": False}
        diag: dict = {}
        llm_backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
            _diagnostics_out=diag,
        )
        assert diag.get("truncated") is False


class TestChunkBatchesKeepsSparseResult:
    """extract_from_chunk_batches must not discard sparse-but-valid dense output."""

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_sparse_dense_result_is_validated_not_discarded(self, mock_orch, mock_llm_client):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            structured_sparse_check=True,
        )
        mock_orch.return_value = ({"f1": "only one value"}, {"skeleton_nodes": 1}, None)
        chunks = ["word " * 200]  # long document, sparse fill

        result = backend.extract_from_chunk_batches(
            chunks=chunks, chunk_metadata=None, template=LargeTemplate
        )

        assert result is not None
        assert result.f1 == "only one value"


class TestHonestOutputBudget:
    """Truncation escalation is bounded by real context and learned futility."""

    def test_escalation_ceiling_uses_client_context_limit(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.context_limit = 6000
        # current 4000 * 2.0 = 8000, but ceiling is context // 2 = 3000 -> capped at max(current, 3000)
        retry = backend._retry_max_tokens_for_truncation(4000)
        assert retry is None  # ceiling (4000) not above current

        mock_llm_client.context_limit = 32000
        retry = backend._retry_max_tokens_for_truncation(4000)
        assert retry == 8000  # 4000 * 2 within ceiling 16000

    def test_escalation_disabled_after_marked_futile(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.context_limit = 64000
        assert backend._retry_max_tokens_for_truncation(4000) == 8000
        backend._mark_escalation_futile(8000)
        assert backend._retry_max_tokens_for_truncation(4000) is None

    def test_truncated_escalation_retry_marks_futility(self, mock_llm_client):
        """A retry that still truncates disables escalation for the rest of the run."""
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.context_limit = 64000
        generation = MagicMock()
        generation.max_tokens = 4000
        mock_llm_client._generation = generation

        first_error = ClientError("truncated", details={"truncated": True, "max_tokens": 4000})
        # First call raises (truncated); retry succeeds but is itself truncated.
        mock_llm_client.get_json_response.side_effect = [first_error, {"nodes": []}]
        mock_llm_client.last_call_diagnostics = {"truncated": True}

        diag: dict = {}
        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
            _diagnostics_out=diag,
        )
        assert out == {"nodes": []}
        assert backend._escalation_futile is True
        assert diag.get("truncated") is True
        # Subsequent truncations skip the escalation retry entirely.
        assert backend._retry_max_tokens_for_truncation(4000) is None


class TestTruncationRetryPreservesMode:
    """The truncation retry must replicate the mode of the call that truncated."""

    def test_legacy_only_retry_stays_legacy(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.context_limit = 64000
        generation = MagicMock()
        generation.max_tokens = 4000
        mock_llm_client._generation = generation

        first_error = ClientError("truncated", details={"truncated": True, "max_tokens": 4000})
        mock_llm_client.get_json_response.side_effect = [first_error, {"nodes": []}]
        mock_llm_client.last_call_diagnostics = {"truncated": False}

        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            '{"type": "object"}',
            "ctx",
            structured_output_override=False,
        )
        assert out == {"nodes": []}
        retry_kwargs = mock_llm_client.get_json_response.call_args.kwargs
        # Retry must not silently switch to API structured output...
        assert retry_kwargs["structured_output"] is False
        # ...and must keep the schema embedded in the prompt, like the original call.
        assert "TARGET SCHEMA" in retry_kwargs["prompt"]["user"]


class TestLargeDirectDocHint:
    """Direct extraction hints toward dense when the document dwarfs the budget."""

    def test_large_direct_doc_emits_hint_once(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="direct")
        gen = MagicMock()
        gen.max_tokens = 4096
        mock_llm_client._generation = gen
        big = "x" * 60000  # ~60k chars >> 4096*4*2 = ~32k capacity

        with patch.object(backend, "_log_warning") as warn:
            backend._maybe_hint_dense_for_large_direct(big)
            backend._maybe_hint_dense_for_large_direct(big)
        assert warn.call_count == 1
        assert "dense" in warn.call_args[0][0]

    def test_small_direct_doc_no_hint(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="direct")
        gen = MagicMock()
        gen.max_tokens = 4096
        mock_llm_client._generation = gen
        with patch.object(backend, "_log_warning") as warn:
            backend._maybe_hint_dense_for_large_direct("x" * 5000)
        assert warn.call_count == 0

    def test_dense_contract_never_hints(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        gen = MagicMock()
        gen.max_tokens = 4096
        mock_llm_client._generation = gen
        with patch.object(backend, "_log_warning") as warn:
            backend._maybe_hint_dense_for_large_direct("x" * 60000)
        assert warn.call_count == 0

    def test_output_budget_falls_back_to_default(self, mock_llm_client):
        from docling_graph.llm_clients.config import _DEFAULT_MAX_OUTPUT_TOKENS

        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="direct")
        mock_llm_client._generation = None
        if hasattr(mock_llm_client, "max_tokens"):
            del mock_llm_client.max_tokens
        if hasattr(mock_llm_client, "_max_output_tokens"):
            del mock_llm_client._max_output_tokens
        assert backend._estimated_output_token_budget() == _DEFAULT_MAX_OUTPUT_TOKENS


class TestDirectContractProvenance:
    """Direct contract produces a document-level provenance ledger (spec §5)."""

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_sets_document_level_ledger(self, mock_get_prompt, mock_llm_client):
        from docling_graph.core.provenance import ProvenanceLedger

        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            dense_config={"provenance": "standard"},
        )
        mock_llm_client.get_json_response.return_value = {"name": "Test", "age": 30}
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        result = backend.extract_from_markdown(
            markdown="full document body", template=MockTemplate, context="full document"
        )

        assert isinstance(result, MockTemplate)
        ledger = backend.last_provenance
        assert isinstance(ledger, ProvenanceLedger)
        assert ledger.resolution == "document"
        assert ledger.nodes == {}
        assert ledger.chunks[0].char_length == len("full document body")

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_direct_provenance_off_leaves_no_ledger(self, mock_get_prompt, mock_llm_client):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            dense_config={"provenance": "off"},
        )
        mock_llm_client.get_json_response.return_value = {"name": "Test", "age": 30}
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        backend.extract_from_markdown(
            markdown="full document body", template=MockTemplate, context="full document"
        )
        assert backend.last_provenance is None

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_partial_call_does_not_set_document_ledger(self, mock_get_prompt, mock_llm_client):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            dense_config={"provenance": "standard"},
        )
        mock_llm_client.get_json_response.return_value = {"name": "Test", "age": 30}
        mock_get_prompt.return_value = {"system": "sys", "user": "user"}

        backend.extract_from_markdown(
            markdown="a chunk", template=MockTemplate, context="chunk", is_partial=True
        )
        assert backend.last_provenance is None


class TestLogInfoFormatting:
    """_log_info formats kwargs into the logged message."""

    def test_log_info_with_kwargs_appends_formatted_pairs(self, llm_backend, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="docling_graph"):
            llm_backend._log_info("Processing", chunks=3, tokens=42)
        printed = caplog.records[-1].getMessage()
        assert "Processing" in printed
        assert "3" in printed and "chunks" in printed
        assert "42" in printed and "tokens" in printed

    def test_log_info_without_kwargs(self, llm_backend, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="docling_graph"):
            llm_backend._log_info("Simple message")
        assert caplog.records[-1].getMessage() == "Simple message"


class TestValidationErrorTraceEmit:
    """_log_validation_error emits trace data when trace_data is set (line 192)."""

    def test_validation_error_emits_trace_when_trace_data_set(self, llm_backend, mock_llm_client):
        llm_backend.trace_data = MagicMock()
        invalid_json = {"name": "Test Only"}
        mock_llm_client.get_json_response.return_value = invalid_json
        result = llm_backend.extract_from_markdown(markdown="Some content", template=MockTemplate)
        assert result is None
        llm_backend.trace_data.emit.assert_any_call(
            "validation_error_raw_data",
            "extraction",
            {"context": "document", "raw_data": invalid_json},
        )


class TestScalarCoercionEdgeCases:
    """_coerce_scalar_to_quantity_with_unit non-numeric-string and other-type branches (212-218)."""

    def test_string_non_numeric_becomes_text_value(self, llm_backend):
        result = llm_backend._coerce_scalar_to_quantity_with_unit("not-a-number")
        assert result == {"text_value": "not-a-number"}

    def test_non_str_non_numeric_value_stringified(self, llm_backend):
        result = llm_backend._coerce_scalar_to_quantity_with_unit(None)
        assert result == {"numeric_value": None, "text_value": "None"}


class TestPathHelpersEdgeCases:
    """_set_at_path / _delete_at_path edge cases (234, 243, 246, 250-251)."""

    def test_set_at_path_empty_loc_is_noop(self, llm_backend):
        data = {"a": 1}
        LlmBackend._set_at_path(data, (), "ignored")
        assert data == {"a": 1}

    def test_delete_at_path_empty_loc_is_noop(self, llm_backend):
        data = {"a": 1}
        LlmBackend._delete_at_path(data, ())
        assert data == {"a": 1}

    def test_delete_at_path_parent_none_is_noop(self, llm_backend):
        data = {"a": None}
        # loc[:-1] resolves to data["a"] which is None, so the parent-is-None guard fires.
        LlmBackend._delete_at_path(data, ("a", "leaf"))
        assert data == {"a": None}

    def test_delete_at_path_removes_list_element_by_index(self, llm_backend):
        data = {"items": ["x", "y", "z"]}
        LlmBackend._delete_at_path(data, ("items", 1))
        assert data["items"] == ["x", "z"]

    def test_delete_at_path_list_index_out_of_range_is_noop(self, llm_backend):
        data = {"items": ["x"]}
        LlmBackend._delete_at_path(data, ("items", 5))
        assert data["items"] == ["x"]


class TestApplyQuantityCoercionSkips:
    """_apply_quantity_coercion skip branches for non-quantity errors and bad paths (264, 267-268, 270)."""

    def test_skips_non_quantity_error(self, llm_backend):
        data = {"gap": 1.0}
        errors = [{"type": "float_type", "loc": ("gap",), "msg": "Input should be a valid dict"}]
        changed = llm_backend._apply_quantity_coercion(data, errors)
        assert changed is False

    def test_skips_error_with_empty_loc(self, llm_backend):
        data = {"gap": 1.0}
        errors = [
            {
                "type": "model_type",
                "loc": (),
                "msg": "QuantityWithUnit expected",
                "ctx": {"class_name": "QuantityWithUnit"},
            }
        ]
        changed = llm_backend._apply_quantity_coercion(data, errors)
        assert changed is False

    def test_skips_error_with_unreachable_path(self, llm_backend):
        data = {"gap": 1.0}
        errors = [
            {
                "type": "model_type",
                "loc": ("missing", "nested"),
                "msg": "QuantityWithUnit expected",
                "ctx": {"class_name": "QuantityWithUnit"},
            }
        ]
        changed = llm_backend._apply_quantity_coercion(data, errors)
        assert changed is False

    def test_skips_when_value_already_dict(self, llm_backend):
        data = {"gap": {"numeric_value": 1.0}}
        errors = [
            {
                "type": "model_type",
                "loc": ("gap",),
                "msg": "QuantityWithUnit expected",
                "ctx": {"class_name": "QuantityWithUnit"},
            }
        ]
        changed = llm_backend._apply_quantity_coercion(data, errors)
        assert changed is False


class TestSchemaRefResolution:
    """_resolve_schema_ref / _schema_node_properties_or_any_of / _get_field_schema_at_path (288, 299-304, 317-318, 325, 330)."""

    def test_resolve_schema_ref_definitions_style(self):
        defs = {"Foo": {"properties": {"x": {"type": "string"}}}}
        node = {"$ref": "#/definitions/Foo"}
        resolved = LlmBackend._resolve_schema_ref(node, defs)
        assert resolved == {"properties": {"x": {"type": "string"}}}

    def test_schema_node_properties_from_any_of(self):
        defs: dict = {}
        node = {"anyOf": [{"properties": {"y": {"type": "integer"}}}, {"type": "null"}]}
        props = LlmBackend._schema_node_properties_or_any_of(node, defs)
        assert props == {"y": {"type": "integer"}}

    def test_schema_node_any_of_empty_list_returns_empty(self):
        node = {"anyOf": []}
        props = LlmBackend._schema_node_properties_or_any_of(node, {})
        assert props == {}

    def test_get_field_schema_at_path_handles_exception(self, llm_backend, monkeypatch):
        class BadTemplate:
            @staticmethod
            def model_json_schema() -> NoReturn:
                raise RuntimeError("boom")

        result = LlmBackend._get_field_schema_at_path(BadTemplate, ("field",))
        assert result is None

    def test_get_field_schema_at_path_array_items(self):
        class Item(BaseModel):
            name: str

        class Root(BaseModel):
            items: list[Item]

        schema = LlmBackend._get_field_schema_at_path(Root, ("items", 0, "name"))
        assert schema is not None
        assert schema.get("type") == "string"

    def test_get_field_schema_at_path_missing_key_returns_none(self):
        class Root(BaseModel):
            name: str

        result = LlmBackend._get_field_schema_at_path(Root, ("does_not_exist",))
        assert result is None


class TestEnumDefaultFromSchema:
    """_enum_default_from_schema prefers OTHER, falls back to first enum value (340-343)."""

    def test_prefers_other_case_insensitive(self):
        schema = {"enum": ["FOO", "other", "BAR"]}
        assert LlmBackend._enum_default_from_schema(schema) == "other"

    def test_falls_back_to_first_when_no_other(self):
        schema = {"enum": ["FOO", "BAR"]}
        assert LlmBackend._enum_default_from_schema(schema) == "FOO"

    def test_no_enum_returns_none(self):
        assert LlmBackend._enum_default_from_schema({}) is None


class TestFillMissingRequiredFieldsBranches:
    """_fill_missing_required_fields dedup/skip/doc-id/enum branches (365, 368, 372-374, 382, 389, 394-406, 415)."""

    def test_duplicate_loc_only_filled_once(self, llm_backend):
        data = {"study_id": None}
        errors = [
            {"type": "missing", "loc": ("study_id",)},
            {"type": "missing", "loc": ("study_id",)},
        ]
        # Force both entries to look "already seen" isn't directly observable,
        # but calling with duplicate locs must not raise and must fill once.
        data = {}
        changed = llm_backend._fill_missing_required_fields(data, errors)
        assert changed is True
        assert "study_id" in data

    def test_skips_error_with_non_string_leaf(self, llm_backend):
        data = {"items": [1, 2]}
        errors = [{"type": "missing", "loc": ("items", 0)}]
        changed = llm_backend._fill_missing_required_fields(data, errors)
        assert changed is False

    def test_skips_when_parent_not_dict_or_key_present(self, llm_backend):
        data = {"name": "already set"}
        errors = [{"type": "missing", "loc": ("name",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors)
        assert changed is False

    def test_root_document_reference_field_uses_template_name(self, llm_backend):
        class InsuranceContract(BaseModel):
            document_reference: str

        data = {}
        errors = [{"type": "missing", "loc": ("document_reference",)}]
        changed = llm_backend._fill_missing_required_fields(
            data, errors, template=InsuranceContract
        )
        assert changed is True
        assert data["document_reference"] == "InsuranceContract"

    def test_field_ending_in_document_uses_template_name(self, llm_backend):
        class SomeTemplate(BaseModel):
            source_document: str

        data = {}
        errors = [{"type": "missing", "loc": ("source_document",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=SomeTemplate)
        assert changed is True
        assert data["source_document"] == "SomeTemplate"

    def test_enum_field_uses_schema_enum_default(self, llm_backend):
        class Category(BaseModel):
            pass

        from enum import Enum

        class Status(str, Enum):
            OTHER = "OTHER"
            ACTIVE = "ACTIVE"

        class Root(BaseModel):
            status: Status

        data = {}
        errors = [{"type": "missing", "loc": ("status",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=Root)
        assert changed is True
        assert data["status"] == "OTHER"

    def test_non_id_string_field_without_enum_defaults_to_empty_string(self, llm_backend):
        class Root(BaseModel):
            summary: str

        data = {}
        errors = [{"type": "missing", "loc": ("summary",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=Root)
        assert changed is True
        assert data["summary"] == ""

    def test_no_template_id_field_gets_generated_id(self, llm_backend):
        data = {}
        errors = [{"type": "missing", "loc": ("widget_id",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=None)
        assert changed is True
        assert data["widget_id"].startswith("WIDG-")

    def test_no_template_non_id_field_defaults_to_empty_string(self, llm_backend):
        data = {}
        errors = [{"type": "missing", "loc": ("summary",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=None)
        assert changed is True
        assert data["summary"] == ""


class TestExtractStringFromListOrDict:
    """_extract_string_from_list_or_dict type branches (446-483)."""

    def test_none_returns_none(self):
        assert LlmBackend._extract_string_from_list_or_dict(None) is None

    def test_blank_string_falls_through_to_none(self):
        assert LlmBackend._extract_string_from_list_or_dict("   ") is None

    def test_bool_scalar_stringified(self):
        assert LlmBackend._extract_string_from_list_or_dict(True) == "True"

    def test_int_scalar_stringified(self):
        assert LlmBackend._extract_string_from_list_or_dict(42) == "42"

    def test_list_of_plain_strings_returns_first_nonblank(self):
        assert LlmBackend._extract_string_from_list_or_dict(["", "  ", "first"]) == "first"

    def test_list_skips_complex_block_dict(self):
        # A dict nesting a container is a full entity block, whatever its keys
        # are named — its inner name must not become the parent's string value.
        complex_item = {"conditions": [{"texte": "clause"}], "nom": "should not use"}
        result = LlmBackend._extract_string_from_list_or_dict([complex_item])
        assert result is None

    def test_list_of_dicts_extracts_identity_key(self):
        result = LlmBackend._extract_string_from_list_or_dict([{"nom": "Alpha"}])
        assert result == "Alpha"

    def test_list_of_dicts_identity_key_scalar_value(self):
        result = LlmBackend._extract_string_from_list_or_dict([{"id": 7}])
        assert result == "7"

    def test_list_of_dicts_falls_back_to_any_string_value(self):
        result = LlmBackend._extract_string_from_list_or_dict([{"unrelated_key": "fallback text"}])
        assert result == "fallback text"

    def test_list_with_no_extractable_string_returns_none(self):
        result = LlmBackend._extract_string_from_list_or_dict([{"count": 5}])
        assert result is None

    def test_dict_complex_block_returns_none(self):
        result = LlmBackend._extract_string_from_list_or_dict(
            {"conditions": [{"texte": "t"}], "nom": "X"}
        )
        assert result is None

    def test_dict_with_long_prose_is_complex(self):
        """Long text marks a full entity block (a clause body, a description)
        regardless of key names — structural, not name-based."""
        prose = "les Dommages corporels subis par les personnes Assurées lors du sinistre " * 2
        result = LlmBackend._extract_string_from_list_or_dict(
            {"exclusion_id": "corporels", "texte": prose}
        )
        assert result is None

    def test_dict_of_short_scalars_is_a_simple_label(self):
        """Two short scalar values stay coercible: structurally this is a
        labeled reference, whatever the keys are called (the old name-list
        heuristic would have vetoed domain-specific key names here)."""
        result = LlmBackend._extract_string_from_list_or_dict({"status": "x", "nom": "X"})
        assert result == "X"

    def test_single_key_container_dict_is_not_complex(self):
        """Single-key dicts are never complex: there is no label to confuse."""
        assert LlmBackend._looks_like_complex_block({"items": [1, 2]}) is False

    def test_dict_identity_key_scalar_value(self):
        result = LlmBackend._extract_string_from_list_or_dict({"id": 3.5})
        assert result == "3.5"

    def test_dict_falls_back_to_any_string_value(self):
        result = LlmBackend._extract_string_from_list_or_dict({"other": "value here"})
        assert result == "value here"

    def test_dict_with_no_extractable_string_returns_none(self):
        result = LlmBackend._extract_string_from_list_or_dict({"count": 5})
        assert result is None

    def test_other_type_returns_none(self):
        assert LlmBackend._extract_string_from_list_or_dict(object()) is None


class TestCoerceStringTypeErrorsSkips:
    """_coerce_string_type_errors skip branches (501, 504-505, 507, 516, 521-522)."""

    def test_skips_uncoercible_error_type(self, llm_backend):
        data = {"name": "ok"}
        errors = [{"type": "missing", "loc": ("name",)}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False

    def test_skips_error_with_empty_loc(self, llm_backend):
        data = {"name": 1}
        errors = [{"type": "int_type", "loc": ()}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False

    def test_skips_unreachable_path(self, llm_backend):
        data = {"name": "ok"}
        errors = [{"type": "int_type", "loc": ("missing", "path")}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False

    def test_coerces_none_to_empty_string_for_string_type(self, llm_backend):
        # A null for a field that rejects None is recovered like a missing
        # field (2026-07-05 IBM report regression: null identity strings).
        data = {"name": None}
        errors = [{"type": "string_type", "loc": ("name",)}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is True
        assert data["name"] == ""

    def test_skips_none_for_non_string_scalar_types(self, llm_backend):
        # None cannot become a valid int/float/bool by coercion.
        data = {"count": None}
        errors = [{"type": "int_type", "loc": ("count",)}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False

    def test_set_at_path_failure_is_caught(self, llm_backend):
        # loc[:-1] resolves to a non-dict parent (a string), so _set_at_path raises
        # TypeError when indexing into it; the method should swallow and continue.
        data = {"name": "abc"}
        errors = [{"type": "int_type", "loc": ("name", 0)}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False


class TestCoerceListTypeErrorsBranches:
    """_coerce_list_type_errors skip/parse branches (538, 541-542, 544, 550-559, 561, 565, 569-570)."""

    def test_skips_wrong_error_type(self, llm_backend):
        data = {"tags": "a,b"}
        errors = [{"type": "string_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is False

    def test_skips_error_with_empty_loc(self, llm_backend):
        data = {"tags": "a,b"}
        errors = [{"type": "list_type", "loc": ()}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is False

    def test_skips_unreachable_path(self, llm_backend):
        data = {"tags": "a,b"}
        errors = [{"type": "list_type", "loc": ("missing", "path")}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is False

    def test_already_list_is_skipped(self, llm_backend):
        data = {"tags": ["already", "list"]}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is False

    def test_python_list_literal_string_is_parsed(self, llm_backend):
        data = {"tags": "['a', 'b']"}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        assert data["tags"] == ["a", "b"]

    def test_bracketed_non_list_syntax_falls_back_to_comma_split(self, llm_backend):
        """Value has bracket-delimited shape but fails ast.literal_eval (SyntaxError/ValueError
        at line 550-559), so it falls through to the comma-split/whole-string branch."""
        data = {"tags": "[a, b]"}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        # "[a, b]" is not valid Python literal syntax (a, b are undefined names),
        # so literal_eval raises and the comma-split path runs on the whole string.
        assert data["tags"] == ["[a", "b]"]

    def test_malformed_list_literal_without_closing_bracket_comma_split(self, llm_backend):
        data = {"tags": "[a, b"}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        assert data["tags"] == ["[a", "b"]

    def test_comma_separated_string_is_split(self, llm_backend):
        data = {"tags": "alpha, beta, gamma"}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        assert data["tags"] == ["alpha", "beta", "gamma"]

    def test_single_string_without_comma_wrapped_in_list(self, llm_backend):
        data = {"tags": "solo"}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        assert data["tags"] == ["solo"]

    def test_non_string_scalar_wrapped_in_list(self, llm_backend):
        data = {"tags": 5}
        errors = [{"type": "list_type", "loc": ("tags",)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is True
        assert data["tags"] == [5]

    def test_set_at_path_failure_is_caught(self, llm_backend):
        data = {"tags": "solo"}
        errors = [{"type": "list_type", "loc": ("tags", 0)}]
        changed = llm_backend._coerce_list_type_errors(data, errors)
        assert changed is False


class TestPruneInvalidFieldsDedup:
    """_prune_invalid_fields skips duplicate locs (584)."""

    def test_duplicate_loc_pruned_only_once(self, llm_backend):
        data = {"name": "ok", "bad": "value"}
        errors = [
            {"type": "string_type", "loc": ("bad",)},
            {"type": "string_type", "loc": ("bad",)},
        ]
        llm_backend._prune_invalid_fields(data, errors)
        assert "bad" not in data
        assert data["name"] == "ok"


class TestCountNonEmptyValues:
    """_count_non_empty_values branches (648, 652, 654, 657)."""

    def test_none_counts_zero(self):
        assert LlmBackend._count_non_empty_values(None) == 0

    def test_blank_string_counts_zero(self):
        assert LlmBackend._count_non_empty_values("   ") == 0

    def test_bool_counts_one(self):
        assert LlmBackend._count_non_empty_values(True) == 1

    def test_list_sums_recursively(self):
        assert LlmBackend._count_non_empty_values(["a", "", "b"]) == 2

    def test_dict_sums_recursively(self):
        assert LlmBackend._count_non_empty_values({"a": "x", "b": ""}) == 1

    def test_other_type_counts_one(self):
        assert LlmBackend._count_non_empty_values(object()) == 1


class TestCountSchemaLeafFieldsAndSparsity:
    """_count_schema_leaf_fields ref-resolution/array/depth branches and sparsity gate (668-703)."""

    def test_resolves_ref_in_defs(self, llm_backend):
        schema = {
            "properties": {"child": {"$ref": "#/$defs/Child"}},
            "$defs": {"Child": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}},
        }
        assert LlmBackend._count_schema_leaf_fields(schema) == 2

    def test_array_of_objects_walks_items(self, llm_backend):
        schema = {
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"properties": {"a": {"type": "string"}}},
                }
            }
        }
        assert LlmBackend._count_schema_leaf_fields(schema) == 1

    def test_non_dict_property_value_is_skipped(self, llm_backend):
        schema = {"properties": {"weird": "not-a-dict", "ok": {"type": "string"}}}
        assert LlmBackend._count_schema_leaf_fields(schema) == 1

    def test_depth_limit_short_circuits(self, llm_backend):
        # Build a deeply nested schema (depth > 6) to hit the depth guard.
        node: dict = {"type": "string"}
        for i in range(10):
            node = {"properties": {f"level{i}": node}}
        assert LlmBackend._count_schema_leaf_fields(node) >= 0  # does not raise / infinite loop

    def test_is_sparse_structured_result_short_markdown_not_sparse(self, llm_backend):
        assert llm_backend._is_sparse_structured_result({}, {}, "short") is False

    def test_is_sparse_structured_result_few_schema_leafs_not_sparse(self, llm_backend):
        schema = {"properties": {"a": {"type": "string"}}}
        assert llm_backend._is_sparse_structured_result({"a": "x"}, schema, "x" * 500) is False

    def test_is_sparse_structured_result_ratio_below_threshold(self, llm_backend):
        schema = {"properties": {f"f{i}": {"type": "string"} for i in range(12)}}
        parsed = {"f0": "only one value"}
        assert llm_backend._is_sparse_structured_result(parsed, schema, "x" * 500) is True


class TestGetClientMaxTokensBranches:
    """_get_client_max_tokens fallback chain (951)."""

    def test_falls_back_to_client_max_tokens_attribute(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client._generation = None
        mock_llm_client.max_tokens = 2048
        assert backend._get_client_max_tokens() == 2048

    def test_returns_none_when_nothing_available(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client._generation = None
        if hasattr(mock_llm_client, "max_tokens"):
            del mock_llm_client.max_tokens
        assert backend._get_client_max_tokens() is None


class TestRetryMaxTokensNoCurrent:
    """_retry_max_tokens_for_truncation returns None when no current budget known (1002)."""

    def test_no_current_max_tokens_returns_none(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client._generation = None
        if hasattr(mock_llm_client, "max_tokens"):
            del mock_llm_client.max_tokens
        assert backend._retry_max_tokens_for_truncation(None) is None


class TestCallPromptNoJsonAndPassthrough:
    """_call_prompt 'no JSON returned' branch and provider/model passthrough (1164-1165, 1189)."""

    def test_no_json_returned_logs_warning_and_returns_none(self, llm_backend, mock_llm_client):
        mock_llm_client.get_json_response.return_value = None
        out = llm_backend._call_prompt(
            {"system": "s", "user": "u"}, "{}", "ctx", structured_output_override=False
        )
        assert out is None

    def test_provider_and_model_passthrough_from_client_diagnostics(
        self, llm_backend, mock_llm_client
    ):
        mock_llm_client.get_json_response.return_value = {"ok": True}
        mock_llm_client.last_call_diagnostics = {"provider": "anthropic", "model": "claude"}
        llm_backend._call_prompt(
            {"system": "s", "user": "u"}, "{}", "ctx", structured_output_override=False
        )
        assert llm_backend.last_call_diagnostics["provider"] == "anthropic"
        assert llm_backend.last_call_diagnostics["model"] == "claude"


class TestExtractFromChunkBatchesBranches:
    """extract_from_chunk_batches direct-contract error and invalid-result branches (1280-1291)."""

    def test_direct_contract_batch_extraction_returns_none(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="direct")
        result = backend.extract_from_chunk_batches(
            chunks=["a"], chunk_metadata=None, template=MockTemplate
        )
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_empty_orchestrator_result_returns_none(self, mock_orch, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_orch.return_value = (None, {}, None)
        result = backend.extract_from_chunk_batches(
            chunks=["a"], chunk_metadata=None, template=MockTemplate
        )
        assert result is None

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_invalid_model_clears_provenance(self, mock_orch, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        from docling_graph.core.provenance import ProvenanceLedger

        fake_ledger = MagicMock(spec=ProvenanceLedger)
        mock_orch.return_value = ({"unrelated_field": "value"}, {"skeleton_nodes": 1}, fake_ledger)
        result = backend.extract_from_chunk_batches(
            chunks=["a"], chunk_metadata=None, template=MockTemplate
        )
        assert result is None
        assert backend.last_provenance is None


class TestExtractWithDenseContractChunking:
    """_extract_with_dense_contract builds single-chunk metadata (1302-1306) via _run_dense_orchestrator (1338)."""

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_dense_contract_builds_single_chunk_with_token_estimate(
        self, mock_orch, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_orch.return_value = ({"name": "A"}, {"skeleton_nodes": 1}, None)
        markdown = "one two three four five"
        result = backend._extract_with_dense_contract(markdown=markdown, context="doc")
        assert result == {"name": "A"}
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["chunks"] == [markdown]
        assert call_kwargs["chunk_metadata"][0]["token_count"] == 5
        assert call_kwargs["chunk_metadata"][0]["page_numbers"] == [0]

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_dense_llm_wrapper_delegates_to_call_prompt(self, mock_orch, mock_llm_client):
        """The _dense_llm closure passed to run_dense_orchestrator forces legacy mode (1338-1346)."""
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")

        def fake_orchestrator(*, llm_call_fn: Any, **kwargs: Any) -> tuple[dict, dict, None]:
            # Invoke the closure the way the real orchestrator would.
            result = llm_call_fn({"system": "s", "user": "u"}, "{}", "ctx")
            return ({"delegated": result}, {}, None)

        mock_orch.side_effect = fake_orchestrator
        mock_llm_client.get_json_response.return_value = {"ok": True}
        result = backend._extract_with_dense_contract(markdown="doc text", context="doc")
        assert result == {"delegated": {"ok": True}}
        # Confirm legacy-only mode: structured_output=False passed to client.
        assert mock_llm_client.get_json_response.call_args.kwargs["structured_output"] is False


class TestGenerateResponseBranches:
    """generate() Response class text-building branches (1477-1486)."""

    def test_generate_dict_response_serialized_to_json_text(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.get_json_response.return_value = {"key": "value"}
        response = backend.generate(system_prompt="s", user_prompt="u")
        assert json.loads(response.text) == {"key": "value"}

    def test_generate_string_response_kept_as_is(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.get_json_response.return_value = "plain text response"
        response = backend.generate(system_prompt="s", user_prompt="u")
        assert response.text == "plain text response"

    def test_generate_other_type_response_stringified(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client)
        mock_llm_client.get_json_response.return_value = 12345
        response = backend.generate(system_prompt="s", user_prompt="u")
        assert response.text == "12345"


class TestGleaningLlmCallClosure:
    """The gleaning closure (_gleaning_llm_call, line ~917) delegates through _get_json_response."""

    @patch("docling_graph.core.extractors.backends.llm_backend.merge_gleaned_direct")
    @patch("docling_graph.core.extractors.backends.llm_backend.run_gleaning_pass_direct")
    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_gleaning_closure_invokes_get_json_response(
        self, mock_get_prompt, mock_gleaning, mock_merge, mock_llm_client
    ):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            dense_config={"gleaning_enabled": True},
        )
        mock_get_prompt.return_value = {"system": "s", "user": "u"}
        mock_llm_client.get_json_response.return_value = {"name": "Pre", "age": 10}

        captured_llm_call_fn = {}

        def fake_gleaning(
            *, markdown: str, existing_result: Any, schema_json: str, llm_call_fn: Any
        ) -> None:
            captured_llm_call_fn["fn"] = llm_call_fn
            # Exercise the closure directly to hit its body.
            llm_call_fn({"system": "s2", "user": "u2"})
            return None

        mock_gleaning.side_effect = fake_gleaning
        result = backend.extract_from_markdown(
            markdown="Full doc content.", template=MockTemplate, context="doc"
        )
        assert result is not None
        assert "fn" in captured_llm_call_fn
        # get_json_response called once for primary extraction, once via the closure.
        assert mock_llm_client.get_json_response.call_count == 2


class TestFillMissingRequiredFieldsExceptionAndIdBranches:
    """Remaining _fill_missing_required_fields branches: _get_at_path exception (373-374),
    no-enum-default id-field branch with template (394-396), and empty field-schema fallback (400-406)."""

    def test_get_at_path_exception_is_caught(self, llm_backend):
        # loc[:-1] resolves through a string (not indexable by string key), raising TypeError.
        data = {"name": "abc"}
        errors = [{"type": "missing", "loc": ("name", "nested", "leaf")}]
        changed = llm_backend._fill_missing_required_fields(data, errors)
        assert changed is False

    def test_template_present_no_enum_default_id_field_gets_generated_id(self, llm_backend):
        class Root(BaseModel):
            widget_id: str
            label: str = "x"

        data = {"label": "x"}
        errors = [{"type": "missing", "loc": ("widget_id",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=Root)
        assert changed is True
        assert data["widget_id"].startswith("WIDG-")

    def test_template_present_field_schema_not_found_falls_back_to_generated_id(self, llm_backend):
        """When _get_field_schema_at_path can't resolve the field (falsy), the fallback
        branch (400-404) still generates an id value using the template path."""

        class Root(BaseModel):
            name: str

        # loc references a field not present in the schema at all, so
        # _get_field_schema_at_path returns None -> falsy -> fallback branch.
        data = {"name": "ok"}
        errors = [{"type": "missing", "loc": ("ghost_id",)}]
        # parent must be a dict missing the key for the pre-check to pass.
        changed = llm_backend._fill_missing_required_fields(data, errors, template=Root)
        assert changed is True
        assert data["ghost_id"].startswith("GHOS-")

    def test_template_present_field_schema_not_found_non_id_falls_back_to_empty_string(
        self, llm_backend
    ):
        """Same fallback branch as above (400-406), but a non-'_id' field name takes the
        empty-string leaf (line 406) instead of generating an id."""

        class Root(BaseModel):
            name: str

        data = {"name": "ok"}
        errors = [{"type": "missing", "loc": ("ghost_summary",)}]
        changed = llm_backend._fill_missing_required_fields(data, errors, template=Root)
        assert changed is True
        assert data["ghost_summary"] == ""


class TestExtractStringFromListOrDictPlainString:
    """Plain non-blank string short-circuit (line 448)."""

    def test_plain_string_returned_stripped(self):
        assert LlmBackend._extract_string_from_list_or_dict("  hello  ") == "hello"


class TestCoerceStringTypeErrorsSetAtPathFailure:
    """_set_at_path raising after a successful _get_at_path in _coerce_string_type_errors (521-522)."""

    def test_set_at_path_failure_after_successful_get(self, llm_backend):
        # data["items"] is a tuple (immutable): _get_at_path(data, ("items", 0)) reads
        # the int element 1 successfully (coercible -> coerced = "1"), but _set_at_path's
        # `parent[loc[-1]] = value` then raises TypeError because tuples don't support
        # item assignment, exercising the except at 521-522.
        data = {"items": (1, 2, 3)}
        errors = [{"type": "int_type", "loc": ("items", 0)}]
        changed = llm_backend._coerce_string_type_errors(data, errors)
        assert changed is False
        assert data["items"] == (1, 2, 3)


class TestCallLlmForExtractionDenseBranch:
    """_call_llm_for_extraction dense-contract delegation and fallback-to-direct (730-738)."""

    @patch(
        "docling_graph.core.extractors.backends.llm_backend.LlmBackend._extract_with_dense_contract"
    )
    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_dense_result_returned_directly(
        self, mock_get_prompt, mock_dense_extract, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_dense_extract.return_value = {"name": "FromDense"}
        result = backend._call_llm_for_extraction(
            markdown="doc",
            schema_json="{}",
            schema_dict={},
            is_partial=False,
            context="doc",
            template=MockTemplate,
        )
        assert result == {"name": "FromDense"}
        mock_dense_extract.assert_called_once()
        mock_get_prompt.assert_not_called()

    @patch(
        "docling_graph.core.extractors.backends.llm_backend.LlmBackend._extract_with_dense_contract"
    )
    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_dense_empty_result_falls_back_to_direct(
        self, mock_get_prompt, mock_dense_extract, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_dense_extract.return_value = None
        mock_get_prompt.return_value = {"system": "s", "user": "u"}
        mock_llm_client.get_json_response.return_value = {"name": "FromDirectFallback", "age": 1}
        result = backend._call_llm_for_extraction(
            markdown="doc",
            schema_json="{}",
            schema_dict={},
            is_partial=False,
            context="doc",
            template=MockTemplate,
        )
        assert result == {"name": "FromDirectFallback", "age": 1}
        mock_dense_extract.assert_called_once()
        mock_get_prompt.assert_called_once()


class TestCallLlmForExtractionStructuredOutputFalseReRaises:
    """When structured_output=False, a ClientError from the primary call re-raises (line 775)."""

    @patch("docling_graph.core.extractors.contracts.direct.get_extraction_prompt")
    def test_client_error_reraises_when_structured_output_disabled(
        self, mock_get_prompt, mock_llm_client
    ):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="direct",
            structured_output=False,
        )
        mock_get_prompt.return_value = {"system": "s", "user": "u"}
        mock_llm_client.get_json_response.side_effect = ClientError("hard failure")
        # The ClientError propagates out of _call_llm_for_extraction's inner try,
        # is caught by the outer except Exception, logged, and None is returned.
        result = backend._call_llm_for_extraction(
            markdown="doc",
            schema_json="{}",
            schema_dict={},
            is_partial=False,
            context="doc",
            template=MockTemplate,
        )
        assert result is None


class TestCallPromptStructuredOutputFalseReRaises:
    """_call_prompt: when structured_output=False, ClientError from the primary
    (non-legacy-only) call re-raises instead of falling back to legacy (line 1119)."""

    def test_client_error_reraises_when_structured_output_disabled(self, mock_llm_client):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            structured_output=False,
        )
        mock_llm_client.get_json_response.side_effect = ClientError("hard failure")
        # structured_output_override is None here (not False), so the "primary" branch
        # runs; ClientError re-raises because self.structured_output is False, and the
        # outer except Exception in _call_prompt logs and returns None.
        out = backend._call_prompt({"system": "s", "user": "u"}, "{}", "ctx")
        assert out is None


class TestCallPromptClientErrorEmitsTrace:
    """_call_prompt: ClientError in the primary (non-legacy-only) branch, with
    structured_output=True and trace_data set, emits before falling back to legacy (line 1121)."""

    def test_client_error_emits_trace_then_falls_back_to_legacy(self, mock_llm_client):
        backend = LlmBackend(
            llm_client=mock_llm_client,
            extraction_contract="dense",
            structured_output=True,
        )
        backend.trace_data = MagicMock()
        mock_llm_client.get_json_response.side_effect = [
            ClientError("structured failed"),
            {"ok": True},
        ]
        out = backend._call_prompt({"system": "s", "user": "u"}, "{}", "ctx")
        assert out == {"ok": True}
        backend.trace_data.emit.assert_called_once()
        emit_args = backend.trace_data.emit.call_args[0]
        assert emit_args[0] == "structured_output_fallback_triggered"
        assert emit_args[2]["context"] == "ctx"


class TestCallPromptRetryRaisesException:
    """_call_prompt: the truncation-retry call itself raising is caught and logged (1243-1251)."""

    def test_retry_call_raises_non_client_error(self, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_llm_client.context_limit = 64000
        generation = MagicMock()
        generation.max_tokens = 4000
        mock_llm_client._generation = generation

        first_error = ClientError("truncated", details={"truncated": True, "max_tokens": 4000})
        mock_llm_client.get_json_response.side_effect = [first_error, RuntimeError("retry boom")]

        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
        )
        assert out is None
        assert mock_llm_client.get_json_response.call_count == 2

    def test_retry_call_raises_client_error_with_truncated_details_marks_futile(
        self, mock_llm_client
    ):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_llm_client.context_limit = 64000
        generation = MagicMock()
        generation.max_tokens = 4000
        mock_llm_client._generation = generation

        first_error = ClientError("truncated", details={"truncated": True, "max_tokens": 4000})
        retry_error = ClientError("still truncated", details={"truncated": True})
        mock_llm_client.get_json_response.side_effect = [first_error, retry_error]

        out = backend._call_prompt(
            {"system": "s", "user": "u"},
            "{}",
            "ctx",
            structured_output_override=False,
        )
        assert out is None
        assert backend._escalation_futile is True


# --- 2026-07-05 IBM annual-report failure regressions ---


class _AutoSegment(BaseModel):
    name: str


class _AutoAcquisition(BaseModel):
    target_name: str
    assigned_segment: _AutoSegment | None = None


class _AutoOfficer(BaseModel):
    full_name: str


class _AutoReport(BaseModel):
    acquisitions: list[_AutoAcquisition] = Field(default_factory=list)
    executive_officers: list[_AutoOfficer] = Field(default_factory=list)


def _ibm_error_shape_data() -> dict:
    """The exact error shape that discarded a completed 26-minute dense run:
    two null required strings plus two missing required fields."""
    return {
        "acquisitions": [
            {"target_name": "A", "assigned_segment": {"name": "Software"}},
            {"target_name": "B", "assigned_segment": {"name": None}},
            {"assigned_segment": {}},
        ],
        "executive_officers": [
            {"full_name": "Jane Doe"},
            {"full_name": None},
        ],
    }


class TestSalvageLoopRegression:
    """fill -> stuck -> prune -> fill used to consume every salvage pass and
    exit without re-validating data that had just become valid."""

    def test_ibm_error_shape_recovers(self, llm_backend):
        model = llm_backend._validate_extraction(_ibm_error_shape_data(), _AutoReport, "test")
        assert model is not None
        assert len(model.acquisitions) == 3
        assert model.acquisitions[1].assigned_segment.name == ""
        assert model.acquisitions[2].target_name == ""
        assert model.executive_officers[1].full_name == ""
        assert llm_backend.last_validation_errors == []

    def test_final_validation_attempt_after_last_mutation_pass(self, llm_backend, monkeypatch):
        # Disable None->"" coercion to force the pre-fix fill -> stuck ->
        # prune -> fill sequence; only the final validation attempt (added
        # after the last mutation round) makes this recoverable.
        monkeypatch.setattr(llm_backend, "_coerce_string_type_errors", lambda data, errors: False)
        model = llm_backend._validate_extraction(_ibm_error_shape_data(), _AutoReport, "test")
        assert model is not None
        assert model.executive_officers[1].full_name == ""

    def test_terminal_errors_recorded_on_failure(self, llm_backend):
        class Strict(BaseModel):
            model_config = ConfigDict(extra="forbid")
            count: int

        model = llm_backend._validate_extraction({"count": {"no": "way"}}, Strict, "test")
        assert model is None
        assert llm_backend.last_validation_errors
        assert llm_backend.last_validation_errors[0]["loc"] == ["count"]


class TestPreflightContextGuard:
    """Full-document calls that cannot fit the context window are refused
    before any provider round-trip."""

    def _backend_with_limits(
        self, mock_llm_client, context_limit: int, max_tokens: int
    ) -> LlmBackend:
        mock_llm_client.context_limit = context_limit
        generation = MagicMock()
        generation.max_tokens = max_tokens
        mock_llm_client._generation = generation
        return LlmBackend(llm_client=mock_llm_client)

    def test_oversized_document_is_refused_without_llm_call(self, mock_llm_client):
        backend = self._backend_with_limits(mock_llm_client, context_limit=1000, max_tokens=200)
        result = backend.extract_from_markdown(markdown="x" * 10_000, template=MockTemplate)
        assert result is None
        mock_llm_client.get_json_response.assert_not_called()

    def test_fitting_document_proceeds(self, mock_llm_client):
        backend = self._backend_with_limits(mock_llm_client, context_limit=1000, max_tokens=200)
        mock_llm_client.get_json_response.return_value = {"name": "n", "age": 3}
        result = backend.extract_from_markdown(markdown="x" * 400, template=MockTemplate)
        assert result is not None

    def test_partial_calls_are_not_guarded(self, mock_llm_client):
        # Chunk-level calls are sized by the chunker; the guard is only for
        # full-document calls.
        backend = self._backend_with_limits(mock_llm_client, context_limit=1000, max_tokens=200)
        mock_llm_client.get_json_response.return_value = {"name": "n", "age": 3}
        result = backend.extract_from_markdown(
            markdown="x" * 10_000, template=MockTemplate, is_partial=True
        )
        assert result is not None

    def test_unknown_context_limit_is_not_guarded(self, mock_llm_client):
        del mock_llm_client.context_limit
        mock_llm_client.get_json_response.return_value = {"name": "n", "age": 3}
        backend = LlmBackend(llm_client=mock_llm_client)
        result = backend.extract_from_markdown(markdown="x" * 100_000, template=MockTemplate)
        assert result is not None


class TestAllowDenseFlag:
    """allow_dense=False must keep a dense-contract backend on the direct
    single-call path (no dense re-entry after a chunked dense run failed)."""

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_allow_dense_false_skips_dense_reentry(self, mock_orchestrator, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_llm_client.get_json_response.return_value = {"name": "n", "age": 3}
        result = backend.extract_from_markdown(
            markdown="Some content", template=MockTemplate, allow_dense=False
        )
        assert result is not None
        mock_orchestrator.assert_not_called()

    @patch("docling_graph.core.extractors.backends.llm_backend.run_dense_orchestrator")
    def test_allow_dense_default_still_routes_dense(self, mock_orchestrator, mock_llm_client):
        backend = LlmBackend(llm_client=mock_llm_client, extraction_contract="dense")
        mock_orchestrator.return_value = ({"name": "n", "age": 3}, {}, None)
        result = backend.extract_from_markdown(markdown="Some content", template=MockTemplate)
        assert result is not None
        mock_orchestrator.assert_called_once()


class TestModelTypeStringCoercion:
    """Bare-string references where the schema expects a nested model are
    coerced to identity-only instances instead of falling through to field
    pruning (observed: one drifted fill response cost ~50 membership edges)."""

    def test_bare_string_list_items_become_identity_instances(self, llm_backend):
        class Tag(BaseModel):
            name: str
            detail: str | None = None
            model_config = ConfigDict(graph_id_fields=["name"])

        class Offer(BaseModel):
            nom: str
            included: list[Tag] = Field(default_factory=list)
            model_config = ConfigDict(graph_id_fields=["nom"])

        class RootDoc(BaseModel):
            title: str = ""
            offers: list[Offer] = Field(default_factory=list)
            model_config = ConfigDict(graph_id_fields=["title"])

        data = {
            "title": "doc",
            "offers": [
                {"nom": "A", "included": [{"name": "ok"}, "Jardin", "Piscine"]},
                {"nom": "B", "included": ["Assurance scolaire"]},
            ],
        }
        model = llm_backend._validate_extraction(data, RootDoc, "test")
        assert model is not None
        assert [t.name for t in model.offers[0].included] == ["ok", "Jardin", "Piscine"]
        assert model.offers[1].included[0].name == "Assurance scolaire"

    def test_bare_string_singular_model_field(self, llm_backend):
        class Setup(BaseModel):
            label: str
            model_config = ConfigDict(graph_id_fields=["label"])

        class RootDoc(BaseModel):
            title: str = ""
            setup: Setup | None = None
            model_config = ConfigDict(graph_id_fields=["title"])

        model = llm_backend._validate_extraction(
            {"title": "d", "setup": "MCR 300"}, RootDoc, "test"
        )
        assert model is not None
        assert model.setup is not None and model.setup.label == "MCR 300"

    def test_model_without_identity_fields_is_left_to_pruner(self, llm_backend):
        class NoId(BaseModel):
            text: str | None = None

        class RootDoc(BaseModel):
            title: str = ""
            notes: list[NoId] = Field(default_factory=list)
            model_config = ConfigDict(graph_id_fields=["title"])

        # No identity field to hang the string on: the pruner removes the item,
        # the document still validates.
        model = llm_backend._validate_extraction(
            {"title": "d", "notes": ["stray"]}, RootDoc, "test"
        )
        assert model is not None
        assert model.notes == []
