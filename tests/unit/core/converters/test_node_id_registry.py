from typing import List

import pytest
from pydantic import BaseModel

from docling_graph.core.converters.node_id_registry import NodeIDRegistry


class PersonModel(BaseModel):
    name: str
    age: int

    model_config = {"graph_id_fields": ["name"]}


class TaggedItemModel(BaseModel):
    tags: List[str]

    model_config = {"graph_id_fields": ["tags"]}


class CompanyModel(BaseModel):
    name: str
    location: str

    model_config = {"graph_id_fields": ["name", "location"]}


@pytest.fixture
def registry():
    """Returns a clean NodeIDRegistry instance for each test."""
    return NodeIDRegistry()


def test_registry_init(registry):
    """Test that the registry initializes with empty structures."""
    assert registry.fingerprint_to_id == {}
    assert registry.id_to_fingerprint == {}
    assert registry.seen_classes == {}


def test_get_node_id_new_item(registry):
    """Test registering a new item."""
    person = PersonModel(name="Alice", age=30)
    node_id = registry.get_node_id(person)

    assert node_id.startswith("PersonModel_")
    assert len(node_id) > len("PersonModel_")


def test_get_node_id_existing_item(registry):
    """Test that registering the same item returns the same ID."""
    person1 = PersonModel(name="Alice", age=30)
    node_id_1 = registry.get_node_id(person1)

    # Same name (identity field), different age
    person2 = PersonModel(name="Alice", age=35)
    node_id_2 = registry.get_node_id(person2)

    # Should return the same ID since identity is based on 'name' only
    assert node_id_1 == node_id_2


def test_get_node_id_different_items(registry):
    """Test that different items get different IDs."""
    person1 = PersonModel(name="Alice", age=30)
    person2 = PersonModel(name="Bob", age=30)

    node_id_1 = registry.get_node_id(person1)
    node_id_2 = registry.get_node_id(person2)

    assert node_id_1 != node_id_2


def test_get_node_id_multiple_identity_fields(registry):
    """Test with multiple identity fields."""
    company1 = CompanyModel(name="Acme Inc.", location="NY")
    company2 = CompanyModel(name="Acme Inc.", location="LA")

    node_id_1 = registry.get_node_id(company1)
    node_id_2 = registry.get_node_id(company2)

    # Different locations should result in different IDs
    assert node_id_1 != node_id_2


def test_register_batch(registry):
    """Test batch registration."""
    person1 = PersonModel(name="Alice", age=30)
    person2 = PersonModel(name="Bob", age=25)

    registry.register_batch([person1, person2])

    stats = registry.get_stats()
    assert stats["total_entities"] == 2
    assert "PersonModel" in stats["classes"]


def test_get_stats(registry):
    """Test getting registry statistics."""
    person = PersonModel(name="Alice", age=30)
    company = CompanyModel(name="Acme Inc.", location="NY")

    registry.get_node_id(person)
    registry.get_node_id(company)

    stats = registry.get_stats()

    assert stats["total_entities"] == 2
    assert len(stats["classes"]) == 2
    assert "PersonModel" in stats["classes"]
    assert "CompanyModel" in stats["classes"]


def test_deterministic_ids(registry):
    """Test that IDs are deterministic across registry instances."""
    person1 = PersonModel(name="Alice", age=30)

    registry1 = NodeIDRegistry()
    node_id_1 = registry1.get_node_id(person1)

    registry2 = NodeIDRegistry()
    node_id_2 = registry2.get_node_id(person1)

    # Same model should produce same ID across different registries
    assert node_id_1 == node_id_2


class BienModel(BaseModel):
    nom: str

    model_config = {"graph_id_fields": ["nom"]}


class ArticleModel(BaseModel):
    title: str

    model_config = {"graph_id_fields": ["title"]}


def test_case_and_diacritic_variants_share_id(registry):
    """Q1: registry canonicalizes identity, so case/diacritic-only variants merge.

    Regression for the split-identity bug: the registry used to fingerprint raw
    values, so "électroménager"/"Électroménager" produced two node ids that
    dense dedup (which canonicalizes) could never reconcile.
    """
    lower = BienModel(nom="électroménager")
    upper = BienModel(nom="Électroménager")
    assert registry.get_node_id(lower) == registry.get_node_id(upper)


def test_digit_bearing_identities_stay_distinct(registry):
    """Q1 guard: canonicalization keeps digits, so numbered siblings never merge."""
    a5 = ArticleModel(title="Article 5")
    a6 = ArticleModel(title="Article 6")
    assert registry.get_node_id(a5) != registry.get_node_id(a6)


def test_list_valued_identity_canonicalizes_each_element(registry):
    """A list-valued id field is canonicalized element-wise, so case-only
    variants of the same tag set fingerprint identically."""
    lower = TaggedItemModel(tags=["Alpha", "beta"])
    upper = TaggedItemModel(tags=["alpha", "BETA"])
    assert registry.get_node_id(lower) == registry.get_node_id(upper)


def test_list_valued_identity_dedupes_and_ignores_order(registry):
    """Duplicate and reordered tags fingerprint the same once canonicalized."""
    a = TaggedItemModel(tags=["x", "y", "x"])
    b = TaggedItemModel(tags=["y", "x"])
    assert registry.get_node_id(a) == registry.get_node_id(b)


def test_list_valued_identity_differs_by_content(registry):
    a = TaggedItemModel(tags=["x", "y"])
    b = TaggedItemModel(tags=["x", "z"])
    assert registry.get_node_id(a) != registry.get_node_id(b)
