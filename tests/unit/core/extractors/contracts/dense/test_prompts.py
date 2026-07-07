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
    assert (
        "if the schema defines it and the text describes it, it must be included in the skeleton"
        in system
    )


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


def test_skeleton_prompt_uses_integer_handles_not_ancestry():
    """Phase 1 prompt requires the compact handle contract (i/p), never ancestry arrays."""
    result = get_skeleton_batch_prompt(
        batch_markdown="Chunk content",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = result["system"]
    assert '"i"' in system and '"p"' in system
    assert "handle" in system
    assert "ancestry" not in system
    # Token discipline: ids must be short verbatim labels, never sentences.
    assert "copied verbatim" in system
    assert "never a sentence" in system


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


def test_fill_prompt_lists_every_instance_id():
    """Phase 2 fill prompt must list all instance ids, since the response is matched by position."""
    from docling_graph.core.extractors.contracts.dense.prompts import get_fill_batch_prompt

    spec = NodeSpec(
        path="items[]",
        node_type="Item",
        id_fields=["item_id"],
        parent_path="",
        field_name="items",
        is_list=True,
    )
    descriptors = [{"path": "items[]", "ids": {"item_id": f"IT-{i}"}} for i in range(7)]
    prompt = get_fill_batch_prompt(
        markdown="doc",
        path="items[]",
        spec=spec,
        descriptors=descriptors,
        projected_schema_json="{}",
    )
    user = prompt["user"]
    for i in range(7):
        assert f"IT-{i}" in user
    assert "... and" not in user
    assert "(7 instances)" in user


def test_prompts_contain_no_internal_fix_labels():
    """Internal change-tracking labels must never leak into LLM prompts."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_fill_batch_prompt,
        get_skeleton_batch_prompt,
    )

    skeleton = get_skeleton_batch_prompt(
        batch_markdown="text",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    spec = NodeSpec(path="", node_type="Root")
    fill = get_fill_batch_prompt(
        markdown="doc", path="", spec=spec, descriptors=[{"ids": {}}], projected_schema_json="{}"
    )
    for text in (skeleton["system"], skeleton["user"], fill["system"], fill["user"]):
        assert "Fix 1." not in text and "Fix 2." not in text


def test_fill_prompt_includes_precision_primitives():
    """Fill prompt carries the generic numeric- and summary-row precision rules."""
    from docling_graph.core.extractors.contracts.dense.prompts import get_fill_batch_prompt

    spec = NodeSpec(path="", node_type="Root")
    prompt = get_fill_batch_prompt(
        markdown="doc", path="", spec=spec, descriptors=[{"ids": {}}], projected_schema_json="{}"
    )
    system = prompt["system"]
    assert "digit-for-digit" in system
    assert "never compute" in system
    assert "summary rows" in system


def test_reconciliation_prompt_is_conservative_and_id_space_only():
    """Reconciliation prompt lists ids per path and forbids merging parameter variants."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_reconciliation_prompt,
    )

    prompt = get_skeleton_reconciliation_prompt(
        {"batches[]": [{"batch_id": "LFP slurry"}, {"batch_id": "LFP_20vol"}]}
    )
    system = prompt["system"]
    user = prompt["user"]
    assert "NEVER group instances that differ by any parameter" in system
    assert "When in doubt, do not merge" in system
    assert "=== PATH batches[] ===" in user
    assert "LFP slurry" in user and "LFP_20vol" in user


def test_reconciliation_prompt_carries_tier_guard_and_renders_candidates():
    """Containment proposals are shown as candidates to verify, never as done
    deals, and the system prompt forbids merging tier/variant names."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_reconciliation_prompt,
    )

    prompt = get_skeleton_reconciliation_prompt(
        {"offres[]": [{"nom": "CONFORT"}, {"nom": "CONFORT PLUS"}]},
        candidate_groups=[{"path": "offres[]", "keep": 0, "merge": [1]}],
    )
    system = prompt["system"]
    assert "CONFORT PLUS" in system
    assert "never merge them" in system
    user = prompt["user"]
    assert "CONTAINMENT CANDIDATES" in user
    assert "may alias" in user
    assert "reject tier/variant pairs" in user


def test_reconciliation_prompt_without_candidates_has_no_candidates_block():
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_reconciliation_prompt,
    )

    prompt = get_skeleton_reconciliation_prompt({"items[]": [{"name": "a"}, {"name": "b"}]})
    assert "CONTAINMENT CANDIDATES" not in prompt["user"]


def test_semantic_guide_renders_docstrings_and_identity_examples():
    """Phase 1 must see the template author's classification guidance: the class
    docstring (truncated) and identity-field examples reach the guide."""
    from pydantic import BaseModel, ConfigDict, Field

    class Formula(BaseModel):
        """A subscribable plan tier; add-on options are NOT formulas."""

        model_config = ConfigDict(graph_id_fields=["name"])
        name: str = Field(..., examples=["ESSENTIAL", "COMFORT"])
        summary: str | None = None

    class Root(BaseModel):
        """Root document."""

        model_config = ConfigDict(graph_id_fields=[])
        formulas: list[Formula] = Field(default_factory=list)

    guide = build_skeleton_semantic_guide(build_node_catalog(Root))
    assert "A subscribable plan tier" in guide
    assert "ESSENTIAL" in guide
    assert "summary" not in guide  # non-identity property names stay out


def test_skeleton_prompt_includes_path_choice_rule():
    """Rule 9 tells the model to pick paths using the per-path descriptions."""
    prompt = get_skeleton_batch_prompt(
        batch_markdown="doc",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = prompt["system"]
    assert "SEMANTIC FIELD GUIDANCE" in system
    assert "never place an instance at a path whose description does not fit" in system


def test_skeleton_prompt_prefers_specific_names_over_generic_labels():
    """Rule 3a steers the model to proper names over generic/positional labels."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_batch_prompt,
    )

    prompt = get_skeleton_batch_prompt(
        batch_markdown="doc",
        catalog_block='- "" (Root) ids=[]',
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    system = prompt["system"]
    assert "most specific proper name" in system
    assert "ESSENTIELLE" in system
    # It must explicitly reject the generic/positional fallback labels.
    assert "Offer 1" in system or "Offre" in system


class TestNegativeHandleClauseConditional:
    """The cross-batch negative-handle clause appears ONLY when an already-found
    list exists: on batch 0 it has nothing to refer to and was observed to
    dilute the core p-emission instruction on small models (invoice regression:
    all parents omitted)."""

    def test_first_batch_prompt_has_no_negative_handle_clause(self):
        from docling_graph.core.extractors.contracts.dense.prompts import (
            get_skeleton_batch_prompt,
        )

        prompt = get_skeleton_batch_prompt(
            batch_markdown="doc",
            catalog_block="",
            batch_index=0,
            total_batches=3,
            allowed_paths=[""],
            already_found=None,
        )
        assert "NEGATIVE handle" not in prompt["system"]
        assert '"p" must reference the handle of a node in this same response' in prompt["system"]

    def test_later_batch_prompt_carries_negative_handle_clause(self):
        from docling_graph.core.extractors.contracts.dense.prompts import (
            get_skeleton_batch_prompt,
        )

        prompt = get_skeleton_batch_prompt(
            batch_markdown="doc",
            catalog_block="",
            batch_index=1,
            total_batches=3,
            allowed_paths=[""],
            already_found='{"i": -1, "path": "x[]", "ids": {"nom": "A"}}',
        )
        assert "NEGATIVE handle" in prompt["system"]
        assert "ALREADY EXTRACTED" in prompt["user"]


def test_skeleton_prompt_forbids_pair_combination_ids():
    """Rule 10: table row-column pairings must not become combined-id instances."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_batch_prompt,
    )

    prompt = get_skeleton_batch_prompt(
        batch_markdown="doc",
        catalog_block="",
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    assert "NEVER join two entity names into a single id" in prompt["system"]


class TestNestedParentRuleConditional:
    """Rule 11 (nested entities attach to the section-named parent, never the
    root) exists ONLY for multi-level catalogs: on flat catalogs every node
    hangs off the root and the sentence would dilute the core instructions."""

    def test_nested_catalog_carries_rule(self):
        from docling_graph.core.extractors.contracts.dense.prompts import (
            get_skeleton_batch_prompt,
        )

        prompt = get_skeleton_batch_prompt(
            batch_markdown="doc",
            catalog_block="",
            batch_index=0,
            total_batches=1,
            allowed_paths=["", "garanties[]", "garanties[].exclusions_specifiques[]"],
        )
        assert "NEVER to the root" in prompt["system"]
        assert "section or article names it" in prompt["system"]

    def test_flat_catalog_has_no_rule(self):
        from docling_graph.core.extractors.contracts.dense.prompts import (
            get_skeleton_batch_prompt,
        )

        prompt = get_skeleton_batch_prompt(
            batch_markdown="doc",
            catalog_block="",
            batch_index=0,
            total_batches=1,
            allowed_paths=["", "seller", "line_items[]"],
        )
        assert "NEVER to the root" not in prompt["system"]


def test_skeleton_prompt_mentions_chunk_attribution():
    """The optional per-node "c" chunk attribution is described in the rules."""
    from docling_graph.core.extractors.contracts.dense.prompts import (
        get_skeleton_batch_prompt,
    )

    prompt = get_skeleton_batch_prompt(
        batch_markdown="doc",
        catalog_block="",
        batch_index=0,
        total_batches=1,
        allowed_paths=[""],
    )
    assert '"c"' in prompt["system"]
    assert "--- CHUNK N ---" in prompt["system"]


def test_fill_prompt_membership_rule_only_for_reference_paths():
    from docling_graph.core.extractors.contracts.dense.catalog import (
        NodeSpec,
        build_node_catalog,
    )
    from docling_graph.core.extractors.contracts.dense.prompts import get_fill_batch_prompt

    spec = NodeSpec(path="offres[]", node_type="entity", id_fields=["nom"], parent_path="")
    with_refs = get_fill_batch_prompt(
        markdown="doc",
        path="offres[]",
        spec=spec,
        descriptors=[{"ids": {"nom": "A"}}],
        projected_schema_json="{}",
        has_reference_fields=True,
    )
    without = get_fill_batch_prompt(
        markdown="doc",
        path="offres[]",
        spec=spec,
        descriptors=[{"ids": {"nom": "A"}}],
        projected_schema_json="{}",
    )
    assert "THIS specific instance" in with_refs["system"]
    assert "THIS specific instance" not in without["system"]
