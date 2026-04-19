"""
Tests for protocol definitions and type checking utilities.
"""

from collections.abc import Iterator, Mapping
from typing import Any, List, Type
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from docling_graph.protocols import (
    Backend,
    DocumentProcessor,
    DocumentProcessorProtocol,
    ExtractionBackendProtocol,
    Extractor,
    ExtractorProtocol,
    LLMClient,
    LLMClientProtocol,
    TextExtractionBackendProtocol,
    get_backend_type,
    is_llm_backend,
    is_vlm_backend,
)


# Test Models
class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


class ValidExtractionBackend:
    def extract_from_document(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        return [template(name=source, value=1)]

    def cleanup(self) -> None:
        return None


class ValidTextExtractionBackend:
    client: Any = object()
    extraction_contract: str = "direct"

    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False,
    ) -> BaseModel | None:
        if is_partial:
            return None
        return template(name=f"{context}:{markdown}", value=1)

    def cleanup(self) -> None:
        return None


class JsonResponse(dict[str, Any]):
    """Concrete dict subtype to keep static typing precise in tests."""


class ValidLLMClient:
    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str,
        structured_output: bool = True,
        response_top_level: str = "object",
        response_schema_name: str = "extraction_result",
    ) -> dict[str, Any] | list[Any]:
        return JsonResponse(
            {
                "prompt": prompt,
                "schema_json": schema_json,
                "structured_output": structured_output,
                "response_top_level": response_top_level,
                "response_schema_name": response_schema_name,
            }
        )

    def get_json_response_stream(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str,
        structured_output: bool = True,
        response_top_level: str = "object",
        response_schema_name: str = "extraction_result",
    ) -> Iterator[dict[str, Any] | list[Any]]:
        yield self.get_json_response(
            prompt,
            schema_json,
            structured_output=structured_output,
            response_top_level=response_top_level,
            response_schema_name=response_schema_name,
        )


class ValidExtractor:
    backend: Any

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def extract(self, source: str, template: Type[BaseModel]) -> List[BaseModel]:
        return [template(name=source, value=1)]


class ValidDocumentProcessor:
    def convert_to_docling_doc(self, source: str) -> Any:
        return {"source": source}

    def extract_full_markdown(self, document: Any) -> str:
        return f"# {document['source']}"

    def extract_page_markdowns(self, document: Any) -> List[str]:
        return [f"page:{document['source']}"]


class TestExtractionBackendProtocol:
    """Test VLM backend protocol compliance."""

    def test_is_vlm_backend_with_valid_backend(self):
        """Should return True for valid VLM backend."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock(return_value=[SampleModel(name="test", value=1)])

        result = is_vlm_backend(backend)
        assert result is True

    def test_is_vlm_backend_missing_method(self):
        """Should return False if method is missing."""
        backend = MagicMock(spec=[])  # Empty spec, no methods
        result = is_vlm_backend(backend)
        assert result is False

    def test_is_vlm_backend_with_none(self):
        """Should handle None gracefully."""
        result = is_vlm_backend(None)
        assert result is False

    def test_runtime_checkable_extraction_backend_protocol(self):
        """Concrete implementations should satisfy runtime protocol checks."""
        backend = ValidExtractionBackend()

        assert isinstance(backend, ExtractionBackendProtocol)

    def test_extraction_backend_method_contract(self):
        """Concrete extraction backend should return validated models."""
        backend = ValidExtractionBackend()

        result = backend.extract_from_document("source.pdf", SampleModel)

        assert len(result) == 1
        assert result[0].name == "source.pdf"
        assert result[0].value == 1


class TestTextExtractionBackendProtocol:
    """Test LLM backend protocol compliance."""

    def test_is_llm_backend_with_valid_backend(self):
        """Should return True for valid LLM backend."""
        backend = MagicMock()
        backend.extract_from_markdown = MagicMock(return_value=SampleModel(name="test", value=1))

        result = is_llm_backend(backend)
        assert result is True

    def test_is_llm_backend_missing_method(self):
        """Should return False if method is missing."""
        backend = MagicMock(spec=[])
        result = is_llm_backend(backend)
        assert result is False

    def test_is_llm_backend_with_none(self):
        """Should handle None gracefully."""
        result = is_llm_backend(None)
        assert result is False

    def test_runtime_checkable_text_extraction_backend_protocol(self):
        """Concrete implementations should satisfy runtime protocol checks."""
        backend = ValidTextExtractionBackend()

        assert isinstance(backend, TextExtractionBackendProtocol)

    @pytest.mark.parametrize(
        ("context", "is_partial", "expected_name"),
        [
            ("document", False, "document:# Title"),
            ("page 1", False, "page 1:# Title"),
            ("chunk 1", True, None),
        ],
    )
    def test_text_extraction_backend_method_contract(self, context, is_partial, expected_name):
        """Text extraction backend should honor context and partial extraction behavior."""
        backend = ValidTextExtractionBackend()

        result = backend.extract_from_markdown(
            "# Title", SampleModel, context=context, is_partial=is_partial
        )

        if expected_name is None:
            assert result is None
        else:
            assert result is not None
            assert result.name == expected_name
            assert result.value == 1


class TestLLMClientProtocol:
    """Test LLM client protocol."""

    def test_llm_client_has_context_limit(self):
        """Client should have context_limit property."""
        client = MagicMock()
        client.context_limit = 4096
        assert client.context_limit == 4096

    def test_llm_client_has_get_json_response(self):
        """Client should have get_json_response method."""
        client = MagicMock()
        response = {"key": "value"}
        client.get_json_response = MagicMock(return_value=response)

        result = client.get_json_response("prompt", '{"schema": "json"}')
        assert result == response

    def test_llm_client_has_get_json_response_stream(self):
        """Client should have get_json_response_stream method that returns an iterator."""
        client = MagicMock()
        # Mock streaming responses as an iterator
        stream_responses = [{"partial": "data1"}, {"partial": "data2"}, {"complete": "result"}]
        client.get_json_response_stream = MagicMock(return_value=iter(stream_responses))

        # Call the streaming method
        result_iterator = client.get_json_response_stream("prompt", '{"schema": "json"}')

        # Verify it returns an iterator and yields expected responses
        results = list(result_iterator)
        assert len(results) == 3
        assert results[0] == {"partial": "data1"}
        assert results[1] == {"partial": "data2"}
        assert results[2] == {"complete": "result"}

    def test_runtime_checkable_llm_client_protocol(self):
        """Concrete implementations should satisfy runtime protocol checks."""
        client = ValidLLMClient()

        assert isinstance(client, LLMClientProtocol)

    def test_llm_client_get_json_response_accepts_mapping_prompt_and_defaults(self):
        """Concrete client should support mapping prompts and default protocol arguments."""
        client = ValidLLMClient()

        result = client.get_json_response(
            {"system": "sys", "user": "usr"},
            '{"type": "object"}',
        )
        assert isinstance(result, dict)

        assert result["prompt"] == {"system": "sys", "user": "usr"}
        assert result["schema_json"] == '{"type": "object"}'
        assert result["structured_output"] is True
        assert result["response_top_level"] == "object"
        assert result["response_schema_name"] == "extraction_result"

    def test_llm_client_stream_supports_non_default_arguments(self):
        """Streaming client should propagate explicit protocol arguments."""
        client = ValidLLMClient()

        results = list(
            client.get_json_response_stream(
                "prompt",
                '{"type": "array"}',
                structured_output=False,
                response_top_level="array",
                response_schema_name="custom_schema",
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["structured_output"] is False
        assert results[0]["response_top_level"] == "array"
        assert results[0]["response_schema_name"] == "custom_schema"


class TestExtractorProtocol:
    """Test extractor strategy protocol."""

    def test_extractor_has_backend(self):
        """Extractor should have backend attribute."""
        extractor = MagicMock()
        extractor.backend = MagicMock()
        assert extractor.backend is not None

    def test_extractor_has_extract_method(self):
        """Extractor should have extract method."""
        extractor = MagicMock()
        extractor.extract = MagicMock(return_value=([SampleModel(name="test", value=1)], None))

        result = extractor.extract("source.pdf", SampleModel)
        assert isinstance(result, tuple)
        models, _document = result
        assert isinstance(models, list)

    def test_runtime_checkable_extractor_protocol(self):
        """Concrete implementations should satisfy runtime protocol checks."""
        extractor = ValidExtractor(ValidExtractionBackend())

        assert isinstance(extractor, ExtractorProtocol)

    def test_extractor_method_contract_returns_models(self):
        """Concrete extractor should return a list of models."""
        extractor = ValidExtractor(ValidExtractionBackend())

        result = extractor.extract("source.pdf", SampleModel)

        assert isinstance(result, list)
        assert result[0].name == "source.pdf"
        assert result[0].value == 1


class TestDocumentProcessorProtocol:
    """Test document processor protocol."""

    def test_document_processor_has_convert_to_docling_doc(self):
        """Processor should have convert_to_docling_doc method."""
        processor = MagicMock()
        processor.convert_to_docling_doc = MagicMock(return_value=MagicMock())

        result = processor.convert_to_docling_doc("doc.pdf")
        assert result is not None

    def test_document_processor_has_extract_full_markdown(self):
        """Processor should have extract_full_markdown method."""
        processor = MagicMock()
        doc = MagicMock()
        processor.extract_full_markdown = MagicMock(return_value="# Markdown\ncontent")

        result = processor.extract_full_markdown(doc)
        assert "Markdown" in result

    def test_document_processor_has_extract_page_markdowns(self):
        """Processor should have extract_page_markdowns method."""
        processor = MagicMock()
        doc = MagicMock()
        processor.extract_page_markdowns = MagicMock(return_value=["Page 1", "Page 2"])

        result = processor.extract_page_markdowns(doc)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_runtime_checkable_document_processor_protocol(self):
        """Concrete implementations should satisfy runtime protocol checks."""
        processor = ValidDocumentProcessor()

        assert isinstance(processor, DocumentProcessorProtocol)

    def test_document_processor_method_contract_round_trip(self):
        """Concrete document processor should convert and extract markdown consistently."""
        processor = ValidDocumentProcessor()

        document = processor.convert_to_docling_doc("doc.pdf")

        assert document == {"source": "doc.pdf"}
        assert processor.extract_full_markdown(document) == "# doc.pdf"
        assert processor.extract_page_markdowns(document) == ["page:doc.pdf"]


class TestTypeAliases:
    """Test type aliases exported by the protocols module."""

    def test_protocol_type_aliases_reference_expected_protocols(self):
        """Aliases should point to the intended protocol classes."""
        assert Extractor is ExtractorProtocol
        assert LLMClient is LLMClientProtocol
        assert DocumentProcessor is DocumentProcessorProtocol

    def test_backend_alias_accepts_vlm_and_llm_runtime_instances(self):
        """Backend alias should correspond to the supported backend protocol family."""
        vlm_backend = ValidExtractionBackend()
        llm_backend = ValidTextExtractionBackend()

        assert isinstance(vlm_backend, ExtractionBackendProtocol)
        assert isinstance(llm_backend, TextExtractionBackendProtocol)
        assert is_vlm_backend(vlm_backend) is True
        assert is_llm_backend(llm_backend) is True
        assert Backend is not None


class TestGetBackendType:
    """Test backend type detection."""

    def test_get_backend_type_vlm(self):
        """Should return 'vlm' for VLM backend."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock()

        result = get_backend_type(backend)
        assert result == "vlm"

    def test_get_backend_type_llm(self):
        """Should return 'llm' for LLM backend."""
        backend = MagicMock()
        backend.extract_from_markdown = MagicMock()
        del backend.extract_from_document  # Remove VLM method

        result = get_backend_type(backend)
        assert result == "llm"

    def test_get_backend_type_unknown(self):
        """Should return 'unknown' for unknown backend."""
        backend = MagicMock(spec=[])
        result = get_backend_type(backend)
        assert result == "unknown"

    def test_get_backend_type_prefers_vlm(self):
        """Should prefer VLM if both methods present."""
        backend = MagicMock()
        backend.extract_from_document = MagicMock()
        backend.extract_from_markdown = MagicMock()

        result = get_backend_type(backend)
        assert result == "vlm"  # VLM check comes first

    def test_get_backend_type_with_non_callable_attributes_is_unknown(self):
        """Non-callable marker attributes should not qualify as backends."""
        backend = MagicMock(spec=[])
        backend.extract_from_document = "not callable"
        backend.extract_from_markdown = "also not callable"

        result = get_backend_type(backend)

        assert result == "unknown"
