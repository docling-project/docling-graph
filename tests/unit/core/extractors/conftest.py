import json
from typing import List
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase  # Important import


@pytest.fixture
def mock_hf_tokenizer():
    """
    Provides a mock HuggingFace tokenizer that passes Pydantic's
    isinstance(..., PreTrainedTokenizerBase) check.
    """
    # Create a MagicMock
    mock_tokenizer = MagicMock()

    # --- THIS IS THE FIX ---
    # Set the mock's __class__ to the base class it needs to be.
    # This will satisfy pydantic's validation.
    mock_tokenizer.__class__ = PreTrainedTokenizerBase
    # --- END FIX ---

    # Mock return values for any methods it might call
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.tokenize.return_value = ["token1", "token2"]

    return mock_tokenizer


"""
Shared fixtures for extractor tests.
"""

# ============================================================================
# Shared Test Schemas
# ============================================================================


class SimpleSchema(BaseModel):
    """Simple test schema."""

    name: str = Field(description="Name")
    value: int = Field(description="Value")


class ComplexSchema(BaseModel):
    """Complex test schema with nested structures."""

    title: str = Field(description="Document title")
    authors: List[str] = Field(description="List of authors")
    sections: List[dict] = Field(description="Document sections")
    metadata: dict = Field(description="Additional metadata")
    tags: List[str] = Field(description="Tags")


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(side_effect=lambda text: len(text) // 4)
    return tokenizer


@pytest.fixture
def mock_chunker():
    """Mock DocumentChunker for testing."""
    from docling_graph.core.extractors.document_chunker import DocumentChunker

    chunker = Mock(spec=DocumentChunker)
    chunker.max_tokens = 8000
    chunker.tokenizer = Mock()
    chunker.tokenizer.count_tokens = Mock(side_effect=lambda text: len(text) // 4)
    return chunker
