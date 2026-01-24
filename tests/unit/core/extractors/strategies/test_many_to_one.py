from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
from docling_graph.protocols import ExtractionBackendProtocol, TextExtractionBackendProtocol


class MockTemplate(BaseModel):
    name: str
    value: int = 0


@pytest.fixture
def mock_llm_backend():
    backend = MagicMock(spec=TextExtractionBackendProtocol)
    backend.client = MagicMock(context_limit=8000, content_ratio=0.8)
    backend.__class__.__name__ = "MockLlmBackend"

    def mock_extract(markdown, template, context, is_partial) -> MockTemplate | None:
        if "fail" in markdown:
            return None
        return template(name=context, value=len(markdown))

    backend.extract_from_markdown.side_effect = mock_extract

    def mock_consolidate(raw_models, programmatic_model, template) -> MockTemplate:
        return template(name="Consolidated", value=999)

    backend.consolidate_from_pydantic_models.side_effect = mock_consolidate

    return backend


@pytest.fixture
def mock_vlm_backend():
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
    with (
        patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor") as mock_dp,
        patch("docling_graph.core.extractors.strategies.many_to_one.ChunkBatcher") as mock_cb,
        patch(
            "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
        ) as mock_merge,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend") as mock_is_llm,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend") as mock_is_vlm,
    ):
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.convert_to_docling_doc.return_value = "MockDoc"
        mock_doc_processor.extract_chunks.return_value = ["chunk1", "chunk2"]
        mock_doc_processor.extract_page_markdowns.return_value = ["page1_md", "page2_md"]
        mock_doc_processor.extract_full_markdown.return_value = "full_doc_md"

        mock_batcher = mock_cb.return_value
        mock_batcher.batch_chunks.return_value = [
            MagicMock(batch_id=0, chunk_count=2, combined_text="chunk1chunk2")
        ]

        mock_merge.return_value = MockTemplate(name="Merged", value=123)

        # Default: not LLM, not VLM (will be overridden in tests)
        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        yield mock_dp, mock_cb, mock_merge, mock_is_llm, mock_is_vlm


def test_init_llm_chunking(mock_llm_backend, patch_deps):
    """Test init with LLM backend and chunking enabled."""
    _, _, _, _, _ = patch_deps

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)

    assert strategy.use_chunking is True
    assert strategy.llm_consolidation is False


def test_extract_with_vlm_single_page(mock_vlm_backend, patch_deps):
    """Test VLM extraction for a single-page document."""
    _, _, mock_merge, _, mock_is_vlm = patch_deps
    mock_is_vlm.return_value = True

    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
    results, _document = strategy.extract("single_page_doc.pdf", MockTemplate)

    assert len(results) == 1
    assert results[0].name == "Page 1"
    mock_merge.assert_not_called()


def test_extract_with_vlm_multi_page(mock_vlm_backend, patch_deps):
    """Test VLM extraction and merge for a multi-page document."""
    _, _, mock_merge, _, mock_is_vlm = patch_deps
    mock_is_vlm.return_value = True

    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
    results, _document = strategy.extract("multi_page_doc.pdf", MockTemplate)

    assert len(results) == 1
    assert results[0].name == "Merged"
    mock_merge.assert_called_once()


# --- Tests for Phase 1 Fix 6: Error Recovery and Tokenizer Passing ---


def test_tokenizer_passing_to_batcher(mock_llm_backend, patch_deps):
    """Test that tokenizer is extracted from DocumentChunker and passed to ChunkBatcher."""
    mock_dp, mock_cb, _, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock chunker with tokenizer that has count_tokens method
    mock_chunker = MagicMock()
    mock_tokenizer = MagicMock()
    mock_count_tokens = MagicMock(return_value=100)
    mock_tokenizer.count_tokens = mock_count_tokens
    mock_chunker.tokenizer = mock_tokenizer
    mock_doc_processor = mock_dp.return_value
    mock_doc_processor.chunker = mock_chunker

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)
    strategy.extract("test.pdf", MockTemplate)

    # Verify tokenizer's count_tokens method was passed to batch_chunks
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.assert_called_once()
    call_kwargs = mock_batcher.batch_chunks.call_args[1]
    assert "tokenizer_fn" in call_kwargs
    assert call_kwargs["tokenizer_fn"] == mock_count_tokens


def test_tokenizer_passing_when_no_chunker(mock_llm_backend, patch_deps):
    """Test graceful handling when chunker doesn't exist."""
    mock_dp, mock_cb, _, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock processor without chunker
    mock_doc_processor = mock_dp.return_value
    mock_doc_processor.chunker = None

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)
    strategy.extract("test.pdf", MockTemplate)

    # Should still work, just without tokenizer
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.assert_called_once()
    call_kwargs = mock_batcher.batch_chunks.call_args[1]
    assert call_kwargs.get("tokenizer_fn") is None


def test_error_recovery_merge_failure_returns_all_models(mock_llm_backend, patch_deps):
    """Test that merge failure returns all extracted models (zero data loss)."""
    _mock_dp, mock_cb, mock_merge, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock merge to fail
    mock_merge.return_value = None

    # Mock multiple batch extractions
    model1 = MockTemplate(name="Batch1", value=10)
    model2 = MockTemplate(name="Batch2", value=20)
    mock_llm_backend.extract_from_markdown.side_effect = [model1, model2]

    # Mock 2 batches
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.return_value = [
        MagicMock(batch_id=0, chunk_count=1, combined_text="chunk1"),
        MagicMock(batch_id=1, chunk_count=1, combined_text="chunk2"),
    ]

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)
    results, _ = strategy.extract("test.pdf", MockTemplate)

    # Should return all extracted models, not empty list
    assert len(results) == 2
    assert results[0].name == "Batch1"
    assert results[1].name == "Batch2"


def test_error_recovery_consolidation_failure_returns_programmatic(mock_llm_backend, patch_deps):
    """Test that consolidation failure returns programmatic merge (zero data loss)."""
    _mock_dp, mock_cb, mock_merge, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock 2 batches to trigger merge logic
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.return_value = [
        MagicMock(batch_id=0, chunk_count=1, combined_text="chunk1"),
        MagicMock(batch_id=1, chunk_count=1, combined_text="chunk2"),
    ]

    # Mock successful extractions
    model1 = MockTemplate(name="Batch1", value=10)
    model2 = MockTemplate(name="Batch2", value=20)
    mock_llm_backend.extract_from_markdown.side_effect = [model1, model2]

    # Mock successful merge
    merged_model = MockTemplate(name="Merged", value=100)
    mock_merge.return_value = merged_model

    # Mock consolidation to fail - must override side_effect, not return_value
    mock_llm_backend.consolidate_from_pydantic_models.side_effect = None
    mock_llm_backend.consolidate_from_pydantic_models.return_value = None

    strategy = ManyToOneStrategy(
        backend=mock_llm_backend, use_chunking=True, llm_consolidation=True
    )
    results, _ = strategy.extract("test.pdf", MockTemplate)

    # Should return programmatic merge, not empty list
    assert len(results) == 1
    assert results[0].name == "Merged"
    assert results[0].value == 100


def test_error_recovery_vlm_merge_failure_returns_all_pages(mock_vlm_backend, patch_deps):
    """Test that VLM merge failure returns all page models (zero data loss)."""
    _, _, mock_merge, _, mock_is_vlm = patch_deps
    mock_is_vlm.return_value = True

    # Mock merge to fail
    mock_merge.return_value = None

    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
    results, _ = strategy.extract("multi_page_doc.pdf", MockTemplate)

    # Should return all page models, not just first
    assert len(results) == 2
    assert results[0].name == "Page 1"
    assert results[1].name == "Page 2"


def test_error_recovery_page_by_page_merge_failure(mock_llm_backend, patch_deps):
    """Test error recovery for page-by-page extraction merge failure."""
    mock_dp, _, mock_merge, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock merge to fail
    mock_merge.return_value = None

    # Mock page extractions
    model1 = MockTemplate(name="Page1", value=10)
    model2 = MockTemplate(name="Page2", value=20)
    mock_llm_backend.extract_from_markdown.side_effect = [model1, model2]

    # Force page-by-page mode by making document too large
    mock_llm_backend.client.context_limit = 1000
    mock_doc_processor = mock_dp.return_value
    mock_doc_processor.extract_full_markdown.return_value = "x" * 10000  # Large doc

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=False)
    results, _ = strategy.extract("test.pdf", MockTemplate)

    # Should return all page models
    assert len(results) == 2
    assert results[0].name == "Page1"
    assert results[1].name == "Page2"


def test_zero_data_loss_no_models_extracted(mock_llm_backend, patch_deps):
    """Test that empty list is returned when no models extracted (no data to preserve)."""
    _mock_dp, mock_cb, _, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock 2 batches but all extractions fail
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.return_value = [
        MagicMock(batch_id=0, chunk_count=1, combined_text="chunk1"),
        MagicMock(batch_id=1, chunk_count=1, combined_text="chunk2"),
    ]

    # Mock all extractions to fail (return None for both batches)
    mock_llm_backend.extract_from_markdown.side_effect = [None, None]

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)
    results, _ = strategy.extract("test.pdf", MockTemplate)

    # Should return empty list (no data to preserve)
    assert len(results) == 0


def test_error_recovery_with_consolidation_enabled(mock_llm_backend, patch_deps):
    """Test error recovery path with LLM consolidation enabled."""
    _mock_dp, mock_cb, mock_merge, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock 2 batches
    mock_batcher = mock_cb.return_value
    mock_batcher.batch_chunks.return_value = [
        MagicMock(batch_id=0, chunk_count=1, combined_text="chunk1"),
        MagicMock(batch_id=1, chunk_count=1, combined_text="chunk2"),
    ]

    # Mock successful extractions and merge
    model1 = MockTemplate(name="Batch1", value=10)
    model2 = MockTemplate(name="Batch2", value=20)
    merged = MockTemplate(name="Merged", value=30)
    consolidated = MockTemplate(name="Consolidated", value=40)

    mock_llm_backend.extract_from_markdown.side_effect = [model1, model2]
    mock_merge.return_value = merged
    # Override the default side_effect from fixture
    mock_llm_backend.consolidate_from_pydantic_models.side_effect = None
    mock_llm_backend.consolidate_from_pydantic_models.return_value = consolidated

    strategy = ManyToOneStrategy(
        backend=mock_llm_backend, use_chunking=True, llm_consolidation=True
    )
    results, _ = strategy.extract("test.pdf", MockTemplate)

    # Should return consolidated result
    assert len(results) == 1
    assert results[0].name == "Consolidated"
    assert results[0].value == 40


def test_catastrophic_failure_returns_empty_with_traceback(mock_llm_backend, patch_deps):
    """Test that catastrophic failures return empty list with traceback logging."""
    mock_dp, _, _, mock_is_llm, _ = patch_deps
    mock_is_llm.return_value = True

    # Mock document processor to raise exception
    mock_dp.return_value.convert_to_docling_doc.side_effect = Exception("Catastrophic error")

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)
    results, doc = strategy.extract("test.pdf", MockTemplate)

    # Should return empty list and None document
    assert len(results) == 0
    assert doc is None


# ============================================================================
# Fix 9: Cached Protocol Checks Tests
# ============================================================================


class TestCachedProtocolChecks:
    """Tests for Fix 9: Cached protocol checks optimization."""

    def test_protocol_checks_cached_at_init(self):
        """Test that protocol checks are cached during initialization."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "LlmBackend"
        mock_backend.client = Mock()
        mock_backend.client.__class__.__name__ = "OpenAIClient"

        with (
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend"
            ) as mock_is_llm,
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend"
            ) as mock_is_vlm,
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.get_backend_type"
            ) as mock_get_type,
        ):
            mock_is_llm.return_value = True
            mock_is_vlm.return_value = False
            mock_get_type.return_value = "llm"

            strategy = ManyToOneStrategy(
                backend=mock_backend,
                use_chunking=False,
            )

            # Protocol checks should be called during init
            mock_is_llm.assert_called_once()
            mock_is_vlm.assert_called_once()
            mock_get_type.assert_called_once()

            # Cached values should be set
            assert strategy._is_llm is True
            assert strategy._is_vlm is False
            assert strategy._backend_type == "llm"

    def test_cached_checks_used_in_extract(self):
        """Test that cached checks are used instead of repeated calls."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "LlmBackend"
        mock_backend.client = Mock()
        mock_backend.client.__class__.__name__ = "OpenAIClient"

        with (
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend"
            ) as mock_is_llm,
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend"
            ) as mock_is_vlm,
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.get_backend_type"
            ) as mock_get_type,
        ):
            mock_is_llm.return_value = True
            mock_is_vlm.return_value = False
            mock_get_type.return_value = "llm"

            strategy = ManyToOneStrategy(
                backend=mock_backend,
                use_chunking=False,
            )

            # Reset call counts after init
            mock_is_llm.reset_mock()
            mock_is_vlm.reset_mock()

            # Mock the extraction method to avoid actual extraction
            strategy._extract_with_llm = Mock(return_value=([], None))

            # Call extract
            strategy.extract(source="test.pdf", template=MockTemplate)

            # Protocol check functions should NOT be called again
            mock_is_llm.assert_not_called()
            mock_is_vlm.assert_not_called()

    def test_cached_checks_consistency(self):
        """Test that cached checks remain consistent."""
        mock_backend = Mock()
        mock_backend.__class__.__name__ = "VlmBackend"

        with (
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend",
                return_value=False,
            ),
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend",
                return_value=True,
            ),
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.get_backend_type",
                return_value="vlm",
            ),
        ):
            strategy = ManyToOneStrategy(
                backend=mock_backend,
                use_chunking=False,
            )

            # Cached values should match protocol checks
            assert strategy._is_llm is False
            assert strategy._is_vlm is True
            assert strategy._backend_type == "vlm"
