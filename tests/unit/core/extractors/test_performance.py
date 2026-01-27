"""
Performance tests for extraction layer.

Tests validate performance improvements from:
- Cached protocol checks

Note: ChunkBatcher tests removed as ChunkBatcher was deleted.
Provider-specific batching is now handled by DocumentChunker.
"""

from unittest.mock import Mock, patch

import pytest


class MockTemplate:
    """Mock template for testing."""

    def __init__(self, name: str, value: int = 0) -> None:
        self.name = name
        self.value = value


class TestPerformanceImprovements:
    """Tests to validate performance improvements."""

    def test_protocol_check_performance(self):
        """Test that cached checks reduce function calls."""
        from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy

        mock_backend = Mock()
        mock_backend.__class__.__name__ = "LlmBackend"
        mock_backend.client = Mock()
        mock_backend.client.__class__.__name__ = "LiteLLMClient"

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
            patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor"),
        ):
            strategy = ManyToOneStrategy(backend=mock_backend)

            # After init, should have called each once
            assert call_count["is_llm"] == 1
            assert call_count["is_vlm"] == 1

            # Cached values should be set
            assert strategy._is_llm is True
            assert strategy._is_vlm is False
            assert strategy._backend_type == "llm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
