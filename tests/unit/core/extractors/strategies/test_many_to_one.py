from typing import List, Optional
from unittest.mock import MagicMock, patch

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

    def mock_extract(markdown, template, context, is_partial) -> Optional[MockTemplate]:
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
    mock_dp, _, _, _, _ = patch_deps

    strategy = ManyToOneStrategy(backend=mock_llm_backend, use_chunking=True)

    assert strategy.use_chunking is True
    assert strategy.llm_consolidation is False


def test_extract_with_vlm_single_page(mock_vlm_backend, patch_deps):
    """Test VLM extraction for a single-page document."""
    _, _, mock_merge, mock_is_llm, mock_is_vlm = patch_deps
    mock_is_vlm.return_value = True

    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
    results = strategy.extract("single_page_doc.pdf", MockTemplate)

    assert len(results) == 1
    assert results[0].name == "Page 1"
    mock_merge.assert_not_called()


def test_extract_with_vlm_multi_page(mock_vlm_backend, patch_deps):
    """Test VLM extraction and merge for a multi-page document."""
    _, _, mock_merge, mock_is_llm, mock_is_vlm = patch_deps
    mock_is_vlm.return_value = True

    strategy = ManyToOneStrategy(backend=mock_vlm_backend)
    results = strategy.extract("multi_page_doc.pdf", MockTemplate)

    assert len(results) == 1
    assert results[0].name == "Merged"
    mock_merge.assert_called_once()
