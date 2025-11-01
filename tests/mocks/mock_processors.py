"""
Mock document processors for testing without Docling dependency.
"""

from pathlib import Path
from typing import List, Optional


class MockDocumentProcessor:
    """Mock document processor for testing."""

    def __init__(self, docling_config: str = "ocr"):
        """Initialize mock document processor.

        Args:
            docling_config: Pipeline configuration ("ocr" or "vision").
        """
        self.docling_config = docling_config
        self.call_count = 0
        self.cleaned_up = False
        self.processed_files = []

    def process_document(self, source: str) -> List[str]:
        """Mock document processing.

        Args:
            source: Path to document file.

        Returns:
            List of page contents (markdown strings).
        """
        self.call_count += 1
        self.processed_files.append(source)

        # Simulate processing by returning mock page content
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        # Return different number of pages based on file size
        file_size = source_path.stat().st_size
        num_pages = max(1, file_size // 1000)  # Rough estimation

        pages = []
        for i in range(min(num_pages, 10)):  # Max 10 pages
            pages.append(
                f"# Page {i + 1}\n\n"
                f"This is mock content for page {i + 1} of {source_path.name}\n\n"
                f"Lorem ipsum dolor sit amet, consectetur adipiscing elit."
            )

        return pages

    def cleanup(self):
        """Mock cleanup of resources."""
        self.cleaned_up = True

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.cleaned_up = False
        self.processed_files = []


class ConfigurableMockProcessor:
    """Configurable mock processor for advanced testing."""

    def __init__(self, page_contents: Optional[List[str]] = None, should_fail: bool = False):
        """Initialize configurable mock processor.

        Args:
            page_contents: Custom page contents to return.
            should_fail: Whether to simulate processing failure.
        """
        self.page_contents = page_contents
        self.should_fail = should_fail
        self.call_count = 0
        self.cleaned_up = False

    def process_document(self, source: str) -> List[str]:
        """Mock document processing with configuration.

        Args:
            source: Path to document.

        Returns:
            Configured page contents.

        Raises:
            Exception: If should_fail is True.
        """
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock processing failure")

        if self.page_contents:
            return self.page_contents

        # Default behavior
        return ["# Mock Page\n\nMock content"]

    def cleanup(self):
        """Mock cleanup."""
        self.cleaned_up = True


class SlowMockProcessor:
    """Mock processor that simulates slow processing."""

    def __init__(self, delay_per_page: float = 0.1):
        """Initialize slow mock processor.

        Args:
            delay_per_page: Delay in seconds per page.
        """
        self.delay_per_page = delay_per_page
        self.call_count = 0

    def process_document(self, source: str) -> List[str]:
        """Mock slow document processing.

        Args:
            source: Path to document.

        Returns:
            Mock page contents.
        """
        import time

        self.call_count += 1

        num_pages = 5
        pages = []

        for i in range(num_pages):
            time.sleep(self.delay_per_page)
            pages.append(f"Page {i + 1} content")

        return pages

    def cleanup(self):
        """Mock cleanup."""


class EmptyMockProcessor:
    """Mock processor that returns empty results."""

    def __init__(self):
        """Initialize empty mock processor."""
        self.call_count = 0

    def process_document(self, source: str) -> List[str]:
        """Return empty page list.

        Args:
            source: Path to document.

        Returns:
            Empty list.
        """
        self.call_count += 1
        return []

    def cleanup(self):
        """Mock cleanup."""


class MultiPageMockProcessor:
    """Mock processor that returns specific number of pages."""

    def __init__(self, num_pages: int = 3):
        """Initialize multi-page mock processor.

        Args:
            num_pages: Number of pages to return.
        """
        self.num_pages = num_pages
        self.call_count = 0

    def process_document(self, source: str) -> List[str]:
        """Return specific number of pages.

        Args:
            source: Path to document.

        Returns:
            List of mock pages.
        """
        self.call_count += 1

        pages = []
        for i in range(self.num_pages):
            pages.append(
                f"# Page {i + 1}\n\nContent for page {i + 1}\n\n- Item 1\n- Item 2\n- Item 3"
            )

        return pages

    def cleanup(self):
        """Mock cleanup."""


# Convenience functions


def create_mock_processor(docling_config: str = "ocr") -> MockDocumentProcessor:
    """Create a standard mock document processor.

    Args:
        docling_config: Pipeline configuration.

    Returns:
        MockDocumentProcessor instance.
    """
    return MockDocumentProcessor(docling_config=docling_config)


def create_failing_processor() -> ConfigurableMockProcessor:
    """Create a mock processor that fails.

    Returns:
        ConfigurableMockProcessor that will fail.
    """
    return ConfigurableMockProcessor(should_fail=True)


def create_multipage_processor(num_pages: int = 5) -> MultiPageMockProcessor:
    """Create a mock processor with specific page count.

    Args:
        num_pages: Number of pages to return.

    Returns:
        MultiPageMockProcessor instance.
    """
    return MultiPageMockProcessor(num_pages=num_pages)
