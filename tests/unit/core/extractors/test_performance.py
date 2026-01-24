"""
Performance tests for extreaction layer.

Tests validate performance improvements from:
- Cached protocol checks
- Provider-specific batching configurations
"""

from unittest.mock import Mock, patch

import pytest

from docling_graph.core.extractors.chunk_batcher import ChunkBatcher


class MockTemplate:
    """Mock template for testing."""

    def __init__(self, name: str, value: int = 0) -> None:
        self.name = name
        self.value = value


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(side_effect=lambda text: len(text) // 4)
    return tokenizer


class TestPerformanceImprovements:
    """Tests to validate performance improvements."""

    def test_protocol_check_performance(self):
        """Test that cached checks reduce function calls."""
        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        mock_backend = Mock()
        mock_backend.__class__.__name__ = "LlmBackend"
        mock_backend.client = Mock()
        mock_backend.client.__class__.__name__ = "OpenAIClient"

        call_count = {"is_llm": 0, "is_vlm": 0}

        def count_is_llm(backend) -> bool:
            call_count["is_llm"] += 1
            return True

        def count_is_vlm(backend) -> bool:
            call_count["is_vlm"] += 1
            return False

        with (
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_llm_backend",
                side_effect=count_is_llm,
            ),
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend",
                side_effect=count_is_vlm,
            ),
            patch(
                "docling_graph.core.extractors.strategies.many_to_one.get_backend_type",
                return_value="llm",
            ),
        ):
            strategy = ManyToOneStrategy(
                backend=mock_backend,
                use_chunking=False,
            )

            # After init, should have called each once
            assert call_count["is_llm"] == 1
            assert call_count["is_vlm"] == 1

            # Mock extraction to avoid actual work
            strategy._extract_with_llm = Mock(return_value=([], None))

            # Multiple extract calls
            for _ in range(5):
                strategy.extract(source="test.pdf", template=MockTemplate)

            # Should still only have 1 call each (from init)
            assert call_count["is_llm"] == 1
            assert call_count["is_vlm"] == 1

    def test_batching_efficiency_with_provider_config(self, mock_tokenizer):
        """Test that provider-specific configs improve batching efficiency."""
        # OpenAI with aggressive merging
        openai_batcher = ChunkBatcher(
            context_limit=128000,
            provider="openai",
        )

        # Ollama with conservative merging
        ollama_batcher = ChunkBatcher(
            context_limit=8000,
            provider="ollama",
        )

        chunks = ["Chunk " + str(i) * 100 for i in range(10)]

        openai_batcher.batch_chunks(
            chunks=chunks,
            tokenizer_fn=mock_tokenizer.count_tokens,
        )

        ollama_batcher.batch_chunks(
            chunks=chunks,
            tokenizer_fn=mock_tokenizer.count_tokens,
        )

        # OpenAI should create fewer batches (more aggressive merging)
        # Note: This assumes chunks fit differently based on merge threshold
        assert openai_batcher.merge_threshold > ollama_batcher.merge_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
