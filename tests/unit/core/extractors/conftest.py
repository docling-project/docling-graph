from unittest.mock import MagicMock

import pytest
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
