"""
Integration tests for Phase 2 & 3 fixes.

Tests the integration of multiple features:
- Dynamic chunker configuration with batching
- Provider-specific optimization with cached protocol checks
"""

import json
from typing import List
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.extractors.chunk_batcher import ChunkBatcher
from docling_graph.core.extractors.document_chunker import DocumentChunker


class ComplexSchema(BaseModel):
    """Complex test schema with nested structures."""

    title: str = Field(description="Document title")
    authors: List[str] = Field(description="List of authors")
    sections: List[dict] = Field(description="Document sections")
    metadata: dict = Field(description="Additional metadata")
    tags: List[str] = Field(description="Tags")


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(side_effect=lambda text: len(text) // 4)
    return tokenizer


@pytest.fixture
def mock_chunker():
    """Mock DocumentChunker for testing."""
    chunker = Mock(spec=DocumentChunker)
    chunker.max_tokens = 8000
    chunker.tokenizer = Mock()
    chunker.tokenizer.count_tokens = Mock(side_effect=lambda text: len(text) // 4)
    return chunker


class TestPhase2Phase3Integration:
    """Integration tests combining multiple Phase 2 & 3 features."""

    def test_chunker_update_with_batching(self, mock_chunker, mock_tokenizer):
        """Test dynamic chunker update integrated with batching."""
        # Create chunker
        with patch("docling_graph.core.extractors.document_chunker.AutoTokenizer"), \
             patch("docling_graph.core.extractors.document_chunker.HuggingFaceTokenizer"), \
             patch("docling_graph.core.extractors.document_chunker.HybridChunker"):

            doc_chunker = DocumentChunker(max_tokens=8000)
            doc_chunker.chunker = mock_chunker
            doc_chunker.original_max_tokens = 8000

            # Update with schema
            schema_size = len(json.dumps(ComplexSchema.model_json_schema()))
            doc_chunker.update_schema_config(schema_size)

            # Create batcher with provider config
            batcher = ChunkBatcher(
                context_limit=doc_chunker.chunker.max_tokens,
                provider="openai",
            )

            chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
            batches = batcher.batch_chunks(
                chunks=chunks,
                tokenizer_fn=mock_tokenizer.count_tokens,
            )

            # Should create batches
            assert len(batches) > 0

    def test_provider_optimization_with_cached_checks(self):
        """Test provider-specific optimization with cached protocol checks."""
        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        mock_backend = Mock()
        mock_backend.__class__.__name__ = "LlmBackend"
        mock_backend.client = Mock()
        mock_backend.client.__class__.__name__ = "OpenAIClient"
        mock_backend.client.context_limit = 128000

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend",
            return_value=True,
        ), patch(
            "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend",
            return_value=False,
        ), patch(
            "docling_graph.core.extractors.strategies.many_to_one.get_backend_type",
            return_value="llm",
        ):

            strategy = ManyToOneStrategy(
                backend=mock_backend,
                use_chunking=True,
            )

            # Should have cached checks
            assert strategy._is_llm is True
            assert strategy._is_vlm is False

            # Should have provider-aware chunker config
            assert strategy.doc_processor.chunker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
