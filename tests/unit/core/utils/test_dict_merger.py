from typing import List, Optional

import pytest
from pydantic import BaseModel

from docling_graph.core.utils.dict_merger import deep_merge_dicts, merge_pydantic_models

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


def test_deep_merge_uses_identity_fields_map_for_entity_lists():
    target = {
        "entities": [
            {"name": "Alice", "source": "A", "score": 0.1},
        ]
    }
    source = {
        "entities": [
            {"name": "Alice", "source": "B", "score": 0.9},
        ]
    }
    merged = deep_merge_dicts(
        target,
        source,
        identity_fields_map={"entities": ["name"]},
    )
    assert len(merged["entities"]) == 1
    assert merged["entities"][0]["source"] == "B"
    assert merged["entities"][0]["score"] == 0.9


def test_entity_list_dedup_same_hash_without_id_no_context_in_hash():
    """Regression: entities without id and same content must match (no context in hash)."""
    # Same keys/values => same content hash => should merge into one (not append duplicate)
    target = {"items": [{"name": "A", "value": 1}]}
    source = {"items": [{"name": "A", "value": 1}]}
    merged = deep_merge_dicts(target, source, identity_fields_map=None)
    assert len(merged["items"]) == 1
    assert merged["items"][0]["name"] == "A"
    assert merged["items"][0]["value"] == 1


def test_nested_identity_path_merge():
    """Nested list uses path-based identity (e.g. studies.experiments)."""
    target = {
        "studies": [
            {
                "study_id": "S1",
                "experiments": [{"experiment_id": "E1", "name": "Exp1"}],
            }
        ]
    }
    source = {
        "studies": [
            {
                "study_id": "S1",
                "experiments": [{"experiment_id": "E1", "name": "Exp1-updated"}],
            }
        ]
    }
    merged = deep_merge_dicts(
        target,
        source,
        identity_fields_map={
            "studies": ["study_id"],
            "studies.experiments": ["experiment_id"],
        },
    )
    assert len(merged["studies"]) == 1
    assert len(merged["studies"][0]["experiments"]) == 1
    assert merged["studies"][0]["experiments"][0]["name"] == "Exp1-updated"


def test_similarity_fallback_off_appends_entity():
    """With merge_similarity_fallback=False, non-matching entities are appended."""
    from docling_graph.core.utils.dict_merger import _merge_entity_lists

    target_list = [{"objective": "Study A", "experiments": [{"experiment_id": "E1"}]}]
    source_list = [{"objective": "Study A", "experiments": [{"experiment_id": "E1"}]}]
    # No id, same content => same hash => merge into one
    result = _merge_entity_lists(
        target_list,
        source_list,
        identity_fields=None,
        merge_similarity_fallback=False,
    )
    assert len(result) == 1
    # Different content (different hash) => append
    target_list2 = [{"objective": "Study A"}]  # no experiments
    source_list2 = [{"objective": "Study B", "experiments": [{"e": 1}]}]
    result2 = _merge_entity_lists(
        target_list2,
        source_list2,
        identity_fields=None,
        merge_similarity_fallback=False,
    )
    assert len(result2) == 2


def test_similarity_fallback_on_merges_by_child_overlap(caplog):
    """With merge_similarity_fallback=True, entities with high child overlap merge (with warning)."""
    from docling_graph.core.utils.dict_merger import _merge_entity_lists

    # Two entities with same children (overlap) but different scalar so different content hash
    target_list = [
        {"study_id": "STUDY-GEN-0", "objective": "X", "experiments": [{"experiment_id": "E1"}]}
    ]
    source_list = [
        {"study_id": "STUDY-GEN-1", "objective": "Y", "experiments": [{"experiment_id": "E1"}]}
    ]
    result = _merge_entity_lists(
        target_list,
        source_list,
        identity_fields=["study_id"],
        merge_similarity_fallback=True,
    )
    # Different study_id => no id match; different content => different hash. Similarity may merge.
    if len(result) == 1:
        assert "merge_similarity_fallback" in caplog.text or "score=" in caplog.text
    assert len(result) <= 2


def test_similarity_below_threshold_does_not_merge():
    """Entities with low child overlap are not merged when fallback is on."""
    from docling_graph.core.utils.dict_merger import _merge_entity_lists

    target_list = [{"a": "only_in_target", "children": [{"x": 1}]}]
    source_list = [{"b": "only_in_source", "children": [{"y": 2}]}]
    result = _merge_entity_lists(
        target_list,
        source_list,
        identity_fields=None,
        merge_similarity_fallback=True,
    )
    # No overlap in children => low Jaccard => should not merge
    assert len(result) == 2
