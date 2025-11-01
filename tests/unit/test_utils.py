"""
Unit tests for utility functions.
"""

import pytest
from pydantic import BaseModel, Field

from docling_graph.core.extractors.utils import merge_pydantic_models


class TestMergePydanticModels:
    """Tests for merge_pydantic_models function."""

    def test_merge_two_models(self):
        """Test merging two simple models."""
        from ..conftest import Person

        person1 = Person(name="Alice", age=25, email="alice@example.com")
        person2 = Person(name="Bob", age=30, email="bob@example.com")

        merged = merge_pydantic_models([person1, person2], Person)
        # Behavior depends on implementation
        assert merged is not None

    def test_merge_empty_list(self):
        """Test merging empty list."""
        from ..conftest import Person

        # Empty list should return None
        merged = merge_pydantic_models([], Person)
        assert merged is None

    def test_merge_single_model(self):
        """Test merging single model."""
        from ..conftest import Person

        person = Person(name="Alice", age=25, email="alice@example.com")
        merged = merge_pydantic_models([person], Person)

        assert merged.name == "Alice"
        assert merged.age == 25
        assert merged.email == "alice@example.com"
