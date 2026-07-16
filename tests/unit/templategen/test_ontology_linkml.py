"""Unit tests for the LinkML -> SPEC-draft compiler (design §5.2 mapping table)."""

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("linkml_runtime")

from docling_graph.templategen.linter import repair_draft
from docling_graph.templategen.ontology import spec_draft_from_ontology
from docling_graph.templategen.ontology.linkml import spec_draft_from_linkml
from docling_graph.templategen.spec import TemplateSpec

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "templategen" / "ontologies"
LIBRARY = FIXTURES / "library.yaml"
CONTAINER = FIXTURES / "container.yaml"


@pytest.fixture(scope="module")
def compiled() -> tuple[dict, list]:
    return spec_draft_from_linkml(LIBRARY)


@pytest.fixture(scope="module")
def draft(compiled) -> dict:
    return compiled[0]


@pytest.fixture(scope="module")
def gaps(compiled) -> list:
    return compiled[1]


def models_by_name(draft: dict) -> dict[str, dict]:
    return {m["name"]: m for m in draft["models"]}


def field(model: dict, name: str) -> dict:
    return next(f for f in model["fields"] if f["name"] == name)


def compile_fixture(**kwargs: Any):
    return spec_draft_from_linkml(LIBRARY, **kwargs)


# ---------------------------------------------------------------------------
# Root + class harvesting
# ---------------------------------------------------------------------------


class TestRootAndClasses:
    def test_tree_root_becomes_root(self, draft):
        assert draft["root"] == "Library"
        assert models_by_name(draft)["Library"]["kind"] == "root"

    def test_root_override(self):
        draft, _ = compile_fixture(root="Book")
        assert draft["root"] == "Book"
        assert "Library" not in models_by_name(draft)

    def test_unknown_root_lists_available(self):
        with pytest.raises(ValueError, match="Available classes"):
            compile_fixture(root="Ghost")

    def test_abstract_class_not_emitted(self, draft):
        assert "NamedThing" not in models_by_name(draft)

    def test_class_description_becomes_docstring(self, draft):
        assert models_by_name(draft)["Book"]["docstring"].startswith(
            "A published book in the catalog."
        )

    def test_aliases_become_also_called(self, draft):
        assert "Also called: Volume." in models_by_name(draft)["Book"]["docstring"]

    def test_provenance_is_ontology(self, draft):
        assert all(m["provenance"] == "ontology" for m in draft["models"])


# ---------------------------------------------------------------------------
# Induced slots: inheritance, slot_usage, types, examples
# ---------------------------------------------------------------------------


class TestInducedSlots:
    def test_inherited_slot_flattened_into_child(self, draft):
        # common_note comes from the abstract NamedThing parent.
        assert field(models_by_name(draft)["Book"], "common_note")["type"] == "str"

    def test_slot_usage_overrides_description(self, draft):
        assert (
            field(models_by_name(draft)["Book"], "page_count")["description"]
            == "Number of pages as printed on the last page."
        )

    @pytest.mark.parametrize(
        ("model", "field_name", "expected_type"),
        [
            ("Book", "page_count", "int"),
            ("Author", "birth_date", "date"),
            ("Author", "author_name", "str"),  # default_range
        ],
    )
    def test_scalar_type_mapping(self, draft, model, field_name, expected_type):
        assert field(models_by_name(draft)[model], field_name)["type"] == expected_type

    def test_slot_examples_become_field_examples(self, draft):
        assert field(models_by_name(draft)["Library"], "library_name")["examples"] == [
            "Central City Library",
            "Northside Branch Library",
        ]

    def test_minimum_cardinality_noted_never_required(self, draft):
        description = field(models_by_name(draft)["Author"], "birth_date")["description"]
        assert "Always present in conforming documents." in description


# ---------------------------------------------------------------------------
# Identity: identifier / key / ladder / demotion
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_identifier_slot_becomes_identity(self, draft):
        assert models_by_name(draft)["Library"]["identity_fields"] == ["library_name"]

    def test_key_slot_becomes_identity(self, draft):
        assert models_by_name(draft)["Book"]["identity_fields"] == ["isbn"]

    def test_heuristic_ladder_fallback(self, draft):
        # Author has no identifier/key; author_name matches the ladder.
        assert models_by_name(draft)["Author"]["identity_fields"] == ["author_name"]

    def test_no_identity_demotes_to_component_with_gap(self, draft, gaps):
        publisher = models_by_name(draft)["Publisher"]
        assert publisher["kind"] == "component"
        assert any(g.model == "Publisher" and g.kind == "missing_identity" for g in gaps)

    def test_identity_with_examples_has_no_gap(self, gaps):
        assert not any(
            g.field in ("library_name", "isbn") and g.kind == "missing_examples" for g in gaps
        )

    def test_identity_without_examples_raises_gap(self, gaps):
        assert any(
            g.model == "Author" and g.field == "author_name" and g.kind == "missing_examples"
            for g in gaps
        )


# ---------------------------------------------------------------------------
# Edges, references, cardinality
# ---------------------------------------------------------------------------


class TestEdges:
    def test_class_ranged_slot_becomes_edge(self, draft):
        books = field(models_by_name(draft)["Library"], "books")
        assert books["role"] == "edge"
        assert books["type"] == "Book"
        assert books["is_list"] is True

    def test_inlined_false_becomes_reference(self, draft):
        # LinkML's not-inlined IS docling-graph's reference edge.
        assert field(models_by_name(draft)["Library"], "books")["reference"] is True

    def test_inlined_unset_stays_full_edge(self, draft):
        authors = field(models_by_name(draft)["Book"], "authors")
        assert authors.get("reference") is not True

    def test_edge_labels_normalized(self, draft):
        by_name = models_by_name(draft)
        assert field(by_name["Library"], "books")["edge_label"] == "HAS_BOOKS"
        assert field(by_name["Book"], "published_by")["edge_label"] == "PUBLISHED_BY"

    def test_maximum_cardinality_one_forces_single(self, draft):
        # primary_author is multivalued but maximum_cardinality: 1.
        assert field(models_by_name(draft)["Book"], "primary_author")["is_list"] is False

    def test_maximum_cardinality_keeps_documented_max(self, draft):
        # Drafts carry the DOCUMENTED maximum; repair_draft doubles exactly once.
        branch = models_by_name(draft)["Branch"]
        assert branch["max_instances"] == 4
        assert "At most 4 per document." in branch["docstring"]

    def test_repair_draft_doubles_documented_max_exactly_once(self):
        # End-to-end R13 contract: graph_max_instances == 2 x the documented n
        # (a doubled draft would end up at 4x — a live runtime bound).
        draft, _ = spec_draft_from_ontology(LIBRARY)
        spec, _ = repair_draft(draft)
        branch = next(m for m in spec.models if m.name == "Branch")
        assert branch.max_instances == 8  # 2 x documented max of 4

    def test_required_non_identity_dropped_to_optional_with_note(self, draft):
        notes = draft["generator"]["notes"]
        assert any("status" in note and "Optionality Law" in note for note in notes)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_permissible_values_become_enum(self, draft):
        assert draft["enums"] == [
            {
                "name": "LibraryStatus",
                "members": ["OPEN", "CLOSED"],
                "synonyms": {"OPEN": ["Open"], "CLOSED": ["Closed", "lib:closed"]},
                "include_other": True,
            }
        ]

    def test_enum_ranged_slot_becomes_enum_field(self, draft):
        status = field(models_by_name(draft)["Library"], "status")
        assert status["type"] == "LibraryStatus"
        assert status["role"] == "property"
        assert status["normalizer"] == "enum"


# ---------------------------------------------------------------------------
# Abstract/mixin slot ranges fan out per concrete descendant
# ---------------------------------------------------------------------------


class TestAbstractRangeFanOut:
    """A slot ranged on an abstract class must not silently become a str field
    (which would drop the whole subtree): it fans out to one edge per concrete
    descendant, mirroring the OWL dropped-abstract convention."""

    @pytest.fixture(scope="class")
    def container_draft(self) -> dict:
        # Container.holds -> NamedThing(abstract){Person, Organization}
        draft, _ = spec_draft_from_linkml(CONTAINER)
        return draft

    def test_concrete_descendants_become_models(self, container_draft):
        assert {"Container", "Person", "Organization"} == set(models_by_name(container_draft))

    def test_edge_fans_out_per_concrete_descendant(self, container_draft):
        container = models_by_name(container_draft)["Container"]
        person = field(container, "holds_person")
        organization = field(container, "holds_organization")
        assert person["type"] == "Person"
        assert organization["type"] == "Organization"
        assert person["role"] == organization["role"] == "edge"
        assert person["is_list"] is organization["is_list"] is True

    def test_no_str_typed_leftover_for_the_abstract_slot(self, container_draft):
        container = models_by_name(container_draft)["Container"]
        assert not any(f["name"] == "holds" for f in container["fields"])

    def test_fan_out_edges_share_the_slot_label(self, container_draft):
        container = models_by_name(container_draft)["Container"]
        labels = {
            field(container, name)["edge_label"] for name in ("holds_person", "holds_organization")
        }
        assert len(labels) == 1

    def test_fan_out_noted_in_generator_notes(self, container_draft):
        notes = container_draft["generator"]["notes"]
        assert any("NamedThing" in note and "fanned out" in note for note in notes)

    def test_abstract_slot_of_library_schema_unaffected(self, draft):
        # library.yaml's NamedThing is abstract but never a slot range: the
        # fan-out path must not disturb plain inheritance flattening.
        assert "NamedThing" not in models_by_name(draft)

    def test_fan_out_draft_validates(self, container_draft):
        assert TemplateSpec.model_validate(container_draft).root == "Container"

    def test_abstract_range_without_concrete_descendants_degrades_to_str(self, tmp_path):
        schema = (
            CONTAINER.read_text(encoding="utf-8")
            .replace("  Person:\n    is_a: NamedThing\n", "  Person:\n")
            .replace("  Organization:\n    is_a: NamedThing\n", "  Organization:\n")
        )
        source = tmp_path / "container_orphan.yaml"
        source.write_text(schema, encoding="utf-8")
        draft, _ = spec_draft_from_linkml(source)
        holds = field(models_by_name(draft)["Container"], "holds")
        assert holds["type"] == "str"
        assert holds["role"] == "property"
        assert any("no concrete descendant" in note for note in draft["generator"]["notes"])


# ---------------------------------------------------------------------------
# Draft validity + dispatcher + dependency guard
# ---------------------------------------------------------------------------


class TestDraftValidity:
    def test_draft_validates_directly(self, draft):
        spec = TemplateSpec.model_validate(draft)
        assert spec.root == "Library"

    def test_root_override_draft_validates(self):
        draft, _ = compile_fixture(root="Book")
        assert TemplateSpec.model_validate(draft).root == "Book"


class TestDispatcher:
    def test_auto_format_dispatches_yaml_to_linkml(self):
        draft, _ = spec_draft_from_ontology(LIBRARY)
        assert draft["generator"]["format"] == "linkml"
        assert draft["root"] == "Library"


class TestMissingDependency:
    def test_import_error_names_the_extra(self, monkeypatch):
        for module in list(sys.modules):
            if module == "linkml_runtime" or module.startswith("linkml_runtime."):
                monkeypatch.delitem(sys.modules, module)
        monkeypatch.setitem(sys.modules, "linkml_runtime", None)
        with pytest.raises(ImportError, match=r"docling-graph\[templategen\]"):
            spec_draft_from_linkml(LIBRARY)
