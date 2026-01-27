from typing import List, Optional

import pytest
from pydantic import BaseModel

from docling_graph.core.utils.dict_merger import merge_pydantic_models

# --- Test Pydantic Models ---


class SimpleItem(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    id: str
    items: List[SimpleItem] = []


class DocumentModel(BaseModel):
    title: str | None = None
    page_count: int | None = None
    content: List[NestedModel] = []


class EntityModel(BaseModel):
    id: str
    age: int | None = None
    email: str | None = None


class EntityDocument(BaseModel):
    entities: List[EntityModel] = []


# --- Tests ---


def test_merge_simple_fields():
    """Test that simple fields (str, int) are merged correctly."""
    model1 = DocumentModel(title="Page 1", page_count=1)
    model2 = DocumentModel(title="Page 2", page_count=2)

    # 'last-one-wins' strategy for simple fields
    merged = merge_pydantic_models([model1, model2], DocumentModel)

    assert merged.title == "Page 2"
    assert merged.page_count == 2


def test_merge_simple_fields_with_none():
    """Test that None values do not overwrite existing values."""
    model1 = DocumentModel(title="Real Title", page_count=10)
    model2 = DocumentModel(title=None, page_count=None)

    merged = merge_pydantic_models([model1, model2], DocumentModel)

    assert merged.title == "Real Title"
    assert merged.page_count == 10

    # Test the other way
    model3 = DocumentModel(title=None, page_count=None)
    model4 = DocumentModel(title="Final Title", page_count=5)

    merged2 = merge_pydantic_models([model3, model4], DocumentModel)
    assert merged2.title == "Final Title"
    assert merged2.page_count == 5


def test_merge_list_of_models():
    """Test that lists of nested models are concatenated."""
    item1 = SimpleItem(name="A", value=1)
    item2 = SimpleItem(name="B", value=2)
    item3 = SimpleItem(name="C", value=3)

    model1 = DocumentModel(content=[NestedModel(id="doc1", items=[item1])])
    model2 = DocumentModel(content=[NestedModel(id="doc2", items=[item2, item3])])

    merged = merge_pydantic_models([model1, model2], DocumentModel)

    assert len(merged.content) == 2
    assert merged.content[0].id == "doc1"
    assert merged.content[1].id == "doc2"
    assert len(merged.content[0].items) == 1
    assert len(merged.content[1].items) == 2


def test_merge_list_with_deduplication():
    """Test that identical list items are de-duplicated."""
    item_a = SimpleItem(name="A", value=1)
    item_b = SimpleItem(name="B", value=2)
    item_c = SimpleItem(name="C", value=3)

    model1 = DocumentModel(content=[NestedModel(id="doc1", items=[item_a, item_b])])
    model2 = DocumentModel(content=[NestedModel(id="doc1", items=[item_b, item_c])])

    merged = merge_pydantic_models([model1, model2], DocumentModel)

    assert len(merged.content) == 1
    assert merged.content[0].id == "doc1"
    assert merged.content[0].items == [item_a, item_b, item_c]


def test_merge_by_id_deep_merge():
    """Test that entities with same id are deep-merged."""
    model1 = EntityDocument(entities=[EntityModel(id="John", age=30)])
    model2 = EntityDocument(entities=[EntityModel(id="John", email="j@b.com")])

    merged = merge_pydantic_models([model1, model2], EntityDocument)

    assert len(merged.entities) == 1
    assert merged.entities[0].id == "John"
    assert merged.entities[0].age == 30
    assert merged.entities[0].email == "j@b.com"


def test_merge_empty_list():
    """Test merging with empty lists."""
    model1 = DocumentModel(content=[])
    model2 = DocumentModel(content=[NestedModel(id="doc1", items=[])])

    merged = merge_pydantic_models([model1, model2], DocumentModel)

    assert len(merged.content) == 1
    assert merged.content[0].id == "doc1"


def test_merge_no_models():
    """Test that merging an empty list returns a default model instance."""
    merged = merge_pydantic_models([], DocumentModel)

    assert isinstance(merged, DocumentModel)
    assert merged.title is None
    assert merged.content == []
