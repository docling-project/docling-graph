"""
Unit tests for dense extraction prompts (Phase 1 skeleton and Phase 2 fill).
"""

import pytest

from docling_graph.core.extractors.contracts.dense.catalog import (
    NodeCatalog,
    NodeSpec,
    build_node_catalog,
    build_skeleton_semantic_guide,
)
from docling_graph.core.extractors.contracts.dense.prompts import (
    build_skeleton_catalog_block,
    format_batch_markdown,
    get_skeleton_batch_prompt,
)
from tests.fixtures.sample_templates.test_template import SampleInvoice


def test_skeleton_prompt_includes_localized_and_global_entity_instructions():
    """Phase 1 system prompt must instruct extraction of both localized and global entities."""
    result = get_skeleton_batch_prompt(
        batch_markdown="--- CHUNK 1 ---\nSome document text.",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = result["system"]
    assert "Global / shared entities" in system
    assert "Do not ignore an entity just because it lacks a distinct sub-label" in system
    assert "localized" in system.lower()
    assert "if the schema defines it and the text describes it, it must be included in the skeleton" in system


def test_skeleton_prompt_includes_global_singleton_id_guidance():
    """Rule 2 must allow global/singleton entities to use descriptive id or ids={}."""
    result = get_skeleton_batch_prompt(
        batch_markdown="Chunk content",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = result["system"]
    assert "global/singleton" in system or "singleton" in system.lower()
    assert "Materials and Methods" in system or "General Protocol" in system


def test_skeleton_batch_prompt_structure():
    """Skeleton prompt returns system and user with expected content."""
    catalog_block = '- "" (Root) ids=[]\n- "studies[]" (Study) ids=[study_id]'
    result = get_skeleton_batch_prompt(
        batch_markdown="--- CHUNK 1 ---\nDoc text.",
        catalog_block=catalog_block,
        batch_index=0,
        total_batches=2,
        allowed_paths=["", "studies[]"],
    )
    assert "system" in result
    assert "user" in result
    assert "Batch 1/2" in result["user"]
    assert "CATALOG" in result["user"]
    assert catalog_block in result["user"]
    assert "nodes" in result["system"].lower()
    assert "path" in result["system"].lower()
    assert "parent" in result["system"].lower()


def test_format_batch_markdown():
    """format_batch_markdown produces chunk-labeled blocks."""
    chunks = [(0, "First chunk text", 100), (1, "Second chunk", 50)]
    out = format_batch_markdown(chunks)
    assert "CHUNK 1" in out
    assert "CHUNK 2" in out
    assert "First chunk text" in out
    assert "Second chunk" in out


def test_build_skeleton_catalog_block():
    """build_skeleton_catalog_block formats path and id_fields per node."""
    catalog = NodeCatalog(
        nodes=[
            NodeSpec(path="", node_type="Root", id_fields=[]),
            NodeSpec(path="items[]", node_type="Item", id_fields=["item_id"]),
        ]
    )
    block = build_skeleton_catalog_block(catalog)
    assert '""' in block
    assert "Root" in block
    assert "items[]" in block
    assert "item_id" in block


def test_skeleton_prompt_requires_ancestry_for_non_root():
    """Phase 1 system prompt must require ancestry array for every non-root node."""
    result = get_skeleton_batch_prompt(
        batch_markdown="Chunk content",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = result["system"]
    assert "MUST" in system and "ancestry" in system
    assert "prove the lineage via ancestry" in system


def test_skeleton_prompt_includes_scope_boundary():
    """Phase 1 system prompt must include scope boundary (primary subject, no external references)."""
    result = get_skeleton_batch_prompt(
        batch_markdown="Chunk content",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = result["system"]
    assert "Scope boundary" in system or "primary subject" in system
    assert "external references" in system or "cited works" in system


def test_skeleton_semantic_guide_omits_non_identity_properties():
    """Phase 1 skeleton semantic guide must not contain non-identity property names from the template."""
    catalog = build_node_catalog(SampleInvoice)
    guide = build_skeleton_semantic_guide(catalog)
    assert "total_amount" not in guide
    assert "vendor_name" not in guide
    assert "invoice_number" in guide
