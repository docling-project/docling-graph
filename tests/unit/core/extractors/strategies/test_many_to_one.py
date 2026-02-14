"""
Unit tests for ManyToOneStrategy.

Tests the many-to-one strategy with direct full-document extraction:
- Direct extraction (single LLM call)
- VLM backend support
"""

from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
from docling_graph.protocols import ExtractionBackendProtocol, TextExtractionBackendProtocol


class MockTemplate(BaseModel):
    """Simple test template."""

    name: str
    value: int = 0


@pytest.fixture
def mock_llm_backend():
    """Create a mock LLM backend."""
    backend = MagicMock(spec=TextExtractionBackendProtocol)
    backend.client = MagicMock()
    backend.__class__.__name__ = "MockLlmBackend"

    def mock_extract(markdown, template, context, is_partial) -> MockTemplate | None:
        if "fail" in markdown:
            return None
        return template(name=context, value=len(markdown))

    backend.extract_from_markdown.side_effect = mock_extract

    return backend


@pytest.fixture
def mock_vlm_backend():
    """Create a mock VLM backend."""
    backend = MagicMock(spec=ExtractionBackendProtocol)
    backend.__class__.__name__ = "MockVlmBackend"

    def mock_extract(source, template) -> List[MockTemplate]:
        if "single" in source:
            return [template(name="Page 1", value=10)]
        if "multi" in source:
            return [
                template(name="Page 1", value=10),
                template(name="Page 2", value=20),
            ]
        return []

    backend.extract_from_document.side_effect = mock_extract

    return backend


@pytest.fixture(autouse=True)
def patch_deps():
    """Patch common dependencies."""
    with (
        patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor") as mock_dp,
        patch(
            "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
        ) as mock_merge,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend") as mock_is_llm,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend") as mock_is_vlm,
    ):
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.convert_to_docling_doc.return_value = "MockDoc"
        mock_doc_processor.extract_full_markdown.return_value = "full_doc_md"

        mock_merge.return_value = MockTemplate(name="Merged", value=123)

        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        yield mock_dp, mock_merge, mock_is_llm, mock_is_vlm


class TestInitialization:
    """Test strategy initialization."""

    def test_init_with_llm_backend(self, mock_llm_backend, patch_deps):
        """Test initialization with LLM backend."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)

        assert strategy.backend == mock_llm_backend

    def test_init_with_docling_config(self, mock_llm_backend, patch_deps):
        """Test initialization with custom docling config."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(
            backend=mock_llm_backend,
            docling_config="vision",
        )

        assert strategy.doc_processor is not None

    def test_delta_initializes_chunker_with_default_tokens(self, mock_llm_backend, patch_deps):
        """Delta mode should always pass non-empty chunker_config."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
            use_chunking=True,
            chunk_max_tokens=None,
        )

        kwargs = mock_dp.call_args.kwargs
        assert kwargs["chunker_config"] == {"chunk_max_tokens": 512}

    def test_delta_initializes_chunker_with_custom_tokens(self, mock_llm_backend, patch_deps):
        """Delta mode should propagate configured chunk_max_tokens."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="delta",
            use_chunking=True,
            chunk_max_tokens=1024,
        )

        kwargs = mock_dp.call_args.kwargs
        assert kwargs["chunker_config"] == {"chunk_max_tokens": 1024}

    def test_delta_requires_chunking_enabled(self, mock_llm_backend, patch_deps):
        """Delta mode should reject disabled chunking."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        with pytest.raises(ValueError, match="requires use_chunking=True"):
            ManyToOneStrategy(
                backend=mock_llm_backend,
                extraction_contract="delta",
                use_chunking=False,
            )


class TestVLMExtraction:
    """Test VLM backend extraction."""

    def test_extract_single_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction for single-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("single_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Page 1"
        mock_merge.assert_not_called()

    def test_extract_multi_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction and merge for multi-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Merged"
        mock_merge.assert_called_once()

    def test_merge_failure_returns_all_pages(self, mock_vlm_backend, patch_deps):
        """Test that VLM merge failure returns all page models (zero data loss)."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        mock_merge.return_value = None

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _ = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 2
        assert results[0].name == "Page 1"
        assert results[1].name == "Page 2"


class TestDirectExtraction:
    """Test direct extraction (single LLM call)."""

    def test_direct_full_document_extraction(self, mock_llm_backend, patch_deps):
        """Test direct full-document extraction."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "test content"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert mock_llm_backend.extract_from_markdown.called
        assert len(results) >= 0

    def test_direct_failure_returns_empty(self, mock_llm_backend, patch_deps):
        """Test that direct extraction returns empty list on failure."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "fail"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 0

    def test_delta_falls_back_to_direct_when_no_model(self, mock_llm_backend, patch_deps):
        """Delta mode should retry once with direct extraction when delta returns no model."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "invoice markdown"
        mock_doc_processor.extract_chunks_with_metadata.return_value = (
            ["chunk1"],
            [{"chunk_id": 0, "token_count": 15, "page_numbers": [1]}],
        )

        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_chunk_batches = Mock(return_value=None)
        mock_llm_backend.extract_from_markdown = Mock(
            return_value=MockTemplate(name="fallback", value=123)
        )

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "fallback"
        mock_llm_backend.extract_from_chunk_batches.assert_called_once()
        mock_llm_backend.extract_from_markdown.assert_called_once()

    def test_delta_fallback_emits_trace_event(self, mock_llm_backend, patch_deps):
        """Delta fallback should emit explicit trace diagnostics."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "invoice markdown"
        mock_doc_processor.extract_chunks_with_metadata.return_value = (
            ["chunk1"],
            [{"chunk_id": 0, "token_count": 15, "page_numbers": [1]}],
        )

        mock_llm_backend.extraction_contract = "delta"
        mock_llm_backend.extract_from_chunk_batches = Mock(return_value=None)
        mock_llm_backend.extract_from_markdown = Mock(
            return_value=MockTemplate(name="fallback", value=123)
        )

        strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="delta")
        strategy.trace_data = MagicMock()
        strategy.trace_data.latest_payload.return_value = {
            "quality_gate": {"ok": False, "reasons": ["missing_root_instance"]},
            "merge_stats": {"parent_lookup_miss": 2},
            "normalizer_stats": {"unknown_path_dropped": 1},
        }

        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 1
        emit_calls = strategy.trace_data.emit.call_args_list
        assert any(call.args[0] == "delta_failed_then_direct_fallback" for call in emit_calls)
