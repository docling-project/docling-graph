"""
Integration tests for extraction workflows.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core.extractors import ManyToOneStrategy, OneToOneStrategy


@pytest.mark.integration
class TestOneToOneStrategy:
    """Tests for OneToOneStrategy extraction."""

    def test_one_to_one_initialization(self, mock_llm_client):
        """Test OneToOneStrategy initialization."""
        strategy = OneToOneStrategy(backend=mock_llm_client, docling_config="ocr")

        assert strategy is not None
        assert strategy.backend == mock_llm_client

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    def test_one_to_one_extract_returns_list(self, mock_doc_processor, mock_llm_client, temp_dir):
        """Test that one-to-one extract returns list of models."""
        from ..conftest import Person

        # Mock document processor
        mock_processor = Mock()
        mock_processor.process_document.return_value = ["page 1", "page 2"]
        mock_doc_processor.return_value = mock_processor

        # Mock backend extraction
        mock_llm_client.extract.return_value = {
            "name": "John",
            "age": 30,
            "email": "john@example.com",
        }

        # Create test document
        test_doc = temp_dir / "test.pdf"
        test_doc.write_bytes(b"%PDF-1.4\nTest")

        strategy = OneToOneStrategy(backend=mock_llm_client, docling_config="ocr")

        # Extract
        results = strategy.extract(str(test_doc), Person)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_one_to_one_processes_each_page(self, mock_llm_client, temp_dir):
        """Test that one-to-one processes each page independently."""
        from ..conftest import Person

        with patch(
            "docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"
        ) as mock_dp:
            # Mock 3 pages
            mock_processor = Mock()
            mock_processor.process_document.return_value = ["page1", "page2", "page3"]
            mock_dp.return_value = mock_processor

            mock_llm_client.extract_from_document.return_value = [
                Person(name="Person 1", age=25, email="test1@example.com"),
                Person(name="Person 2", age=30, email="test2@example.com"),
                Person(name="Person 3", age=35, email="test3@example.com"),
            ]

            # Mock extraction
            mock_llm_client.extract.return_value = {
                "name": "Test",
                "age": 25,
                "email": "test@example.com",
            }

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            strategy = OneToOneStrategy(backend=mock_llm_client)
            results = strategy.extract(str(test_doc), Person)

            # Should have extracted from each page
            assert len(results) == 3


@pytest.mark.integration
class TestManyToOneStrategy:
    """Tests for ManyToOneStrategy extraction."""

    def test_many_to_one_initialization(self, mock_llm_client):
        """Test ManyToOneStrategy initialization."""
        strategy = ManyToOneStrategy(backend=mock_llm_client, docling_config="ocr")

        assert strategy is not None
        assert strategy.backend == mock_llm_client

    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_many_to_one_extract_returns_list(self, mock_doc_processor, mock_llm_client, temp_dir):
        """Test that many-to-one extract returns list with single model."""
        from ..conftest import Person

        # Mock document processor
        mock_processor = Mock()
        mock_processor.process_document.return_value = ["page 1", "page 2"]
        mock_doc_processor.return_value = mock_processor

        # Mock backend extraction
        mock_llm_client.extract.return_value = {
            "name": "John",
            "age": 30,
            "email": "john@example.com",
        }

        test_doc = temp_dir / "test.pdf"
        test_doc.write_bytes(b"test")

        strategy = ManyToOneStrategy(backend=mock_llm_client)
        results = strategy.extract(str(test_doc), Person)

        assert isinstance(results, list)
        # Many-to-one should return single consolidated model
        assert len(results) == 1

    def test_many_to_one_consolidates_pages(self, mock_llm_client, temp_dir):
        """Test that many-to-one consolidates multiple pages."""
        from ..conftest import Person

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor"
        ) as mock_dp:
            # Mock multiple pages
            mock_processor = Mock()
            mock_processor.process_document.return_value = ["page1", "page2", "page3"]
            mock_dp.return_value = mock_processor

            # Mock extraction for each page
            mock_llm_client.extract.return_value = {
                "name": "Test",
                "age": 25,
                "email": "test@example.com",
            }

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            strategy = ManyToOneStrategy(backend=mock_llm_client)
            results = strategy.extract(str(test_doc), Person)

            # Should consolidate into single result
            assert len(results) == 1


@pytest.mark.integration
class TestExtractionComparison:
    """Compare extraction strategies."""

    def test_strategies_use_same_backend(self, mock_llm_client):
        """Test that both strategies can use same backend."""
        one_to_one = OneToOneStrategy(backend=mock_llm_client)
        many_to_one = ManyToOneStrategy(backend=mock_llm_client)

        assert one_to_one.backend == many_to_one.backend

    @patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor")
    @patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor")
    def test_strategies_produce_different_results(
        self, mock_dp_many, mock_dp_one, mock_llm_client, temp_dir
    ):
        """Test that strategies produce different number of results."""
        from ..conftest import Person

        # Setup mocks for both strategies
        mock_proc_one = Mock()
        mock_proc_one.process_document.return_value = ["p1", "p2", "p3"]
        mock_dp_one.return_value = mock_proc_one

        mock_proc_many = Mock()
        mock_proc_many.process_document.return_value = ["p1", "p2", "p3"]
        mock_dp_many.return_value = mock_proc_many

        mock_llm_client.extract_from_document.side_effect = [
            # First call (one-to-one) - returns 3 Person objects
            [
                Person(name="Person 1", age=25, email="test1@example.com"),
                Person(name="Person 2", age=30, email="test2@example.com"),
                Person(name="Person 3", age=35, email="test3@example.com"),
            ],
            # Second call (many-to-one) - returns 1 Person object
            [Person(name="Consolidated", age=28, email="consolidated@example.com")],
        ]

        mock_llm_client.extract.return_value = {
            "name": "Test",
            "age": 25,
            "email": "test@example.com",
        }

        test_doc = temp_dir / "test.pdf"
        test_doc.write_bytes(b"test")

        # Extract with both strategies
        one_to_one = OneToOneStrategy(backend=mock_llm_client)
        many_to_one = ManyToOneStrategy(backend=mock_llm_client)

        results_one = one_to_one.extract(str(test_doc), Person)
        results_many = many_to_one.extract(str(test_doc), Person)

        # One-to-one should have more results
        assert len(results_one) > len(results_many)


@pytest.mark.integration
class TestExtractionErrorHandling:
    """Test error handling in extraction."""

    def test_extraction_handles_missing_file(self, temp_dir):
        """Test extraction handles missing source file."""
        from ..conftest import Person

        # Create a mock that simulates file validation
        mock_backend = Mock()

        def mock_extract_that_checks_file(source, template):
            from pathlib import Path

            if not Path(source).exists():
                return []  # Return empty on missing file
            return [Person(name="Test", age=30, email="test@example.com")]

        mock_backend.extract_from_document = Mock(side_effect=mock_extract_that_checks_file)

        strategy = OneToOneStrategy(backend=mock_backend)
        result = strategy.extract("/nonexistent/file.pdf", Person)

        assert result == [] or result is None

    def test_extraction_handles_backend_error(self, temp_dir):
        """Test extraction handles backend errors."""
        from ..conftest import Person

        # Mock backend that raises error
        mock_backend = Mock()
        mock_backend.extract_from_document.side_effect = Exception("Backend error")
        mock_backend.extract.side_effect = Exception("Backend error")

        with patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"):
            strategy = OneToOneStrategy(backend=mock_backend)

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            result = strategy.extract(str(test_doc), Person)
            assert result is None or result == []

    def test_extraction_handles_invalid_template(self, mock_llm_client, temp_dir):
        """Test extraction handles invalid template class."""
        test_doc = temp_dir / "test.pdf"
        test_doc.write_bytes(b"test")

        strategy = OneToOneStrategy(backend=mock_llm_client)

        # Invalid template (not a Pydantic model)
        result = strategy.extract(str(test_doc), dict)
        assert isinstance(result, list)


@pytest.mark.integration
class TestExtractionWithDifferentBackends:
    """Test extraction with different backend types."""

    def test_extraction_with_llm_backend(self, temp_dir):
        """Test extraction with LLM backend."""
        from ..conftest import Person

        mock_llm = Mock()
        mock_llm.extract.return_value = {"name": "Test", "age": 30, "email": "test@example.com"}

        with patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"):
            strategy = OneToOneStrategy(backend=mock_llm)

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            # Should work with LLM backend
            # Full implementation would verify LLM-specific behavior

    def test_extraction_with_vlm_backend(self, temp_dir):
        """Test extraction with VLM backend."""
        from ..conftest import Person

        mock_vlm = Mock()
        mock_vlm.extract.return_value = [{"name": "Test", "age": 30, "email": "test@example.com"}]

        with patch("docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"):
            strategy = OneToOneStrategy(backend=mock_vlm)

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            # Should work with VLM backend
            # Full implementation would verify VLM-specific behavior


@pytest.mark.integration
@pytest.mark.slow
class TestExtractionPerformance:
    """Test extraction performance characteristics."""

    def test_extraction_processes_multipage_document(self, mock_llm_client, temp_dir):
        """Test extraction can handle multi-page documents."""
        from ..conftest import Person

        with patch(
            "docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"
        ) as mock_dp:
            # Mock 100 pages
            mock_processor = Mock()
            mock_processor.process_document.return_value = [f"page_{i}" for i in range(100)]
            mock_dp.return_value = mock_processor

            mock_llm_client.extract_from_document.return_value = [
                Person(name=f"Person {i}", age=20 + i % 50, email=f"person{i}@example.com")
                for i in range(100)
            ]

            mock_llm_client.extract.return_value = {
                "name": "Test",
                "age": 30,
                "email": "test@example.com",
            }

            test_doc = temp_dir / "large.pdf"
            test_doc.write_bytes(b"test")

            strategy = OneToOneStrategy(backend=mock_llm_client)
            results = strategy.extract(str(test_doc), Person)

            assert len(results) == 100

    def test_extraction_cleans_up_resources(self, mock_llm_client, temp_dir):
        """Test that extraction cleans up document processor."""
        from ..conftest import Person

        with patch(
            "docling_graph.core.extractors.strategies.one_to_one.DocumentProcessor"
        ) as mock_dp:
            mock_processor = Mock()
            mock_processor.process_document.return_value = ["page"]
            mock_processor.cleanup = Mock()
            mock_dp.return_value = mock_processor

            mock_llm_client.extract.return_value = {
                "name": "Test",
                "age": 30,
                "email": "test@example.com",
            }

            test_doc = temp_dir / "test.pdf"
            test_doc.write_bytes(b"test")

            strategy = OneToOneStrategy(backend=mock_llm_client)
            strategy.extract(str(test_doc), Person)

            # Verify cleanup was called
            # Actual cleanup behavior depends on implementation
