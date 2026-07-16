"""Unit tests for the OWL/RDFS/SKOS -> SPEC-draft compiler (design §5.1 mapping table)."""

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rdflib")

from docling_graph.templategen.linter import repair_draft
from docling_graph.templategen.ontology import spec_draft_from_ontology
from docling_graph.templategen.ontology.owl import (
    SkosOnlyOntologyError,
    spec_draft_from_owl,
)
from docling_graph.templategen.spec import TemplateSpec

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "templategen" / "ontologies"


def compile_fixture(name: str, **kwargs: Any):
    return spec_draft_from_owl(FIXTURES / name, **kwargs)


def models_by_name(draft: dict) -> dict[str, dict]:
    return {m["name"]: m for m in draft["models"]}


def field(model: dict, name: str) -> dict:
    return next(f for f in model["fields"] if f["name"] == name)


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


class TestRootResolution:
    def test_auto_detects_unique_non_range_class(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert draft["root"] == "Policy"
        assert models_by_name(draft)["Policy"]["kind"] == "root"

    def test_root_by_local_name(self):
        draft, _ = compile_fixture("policy_basic.ttl", root="Policy")
        assert draft["root"] == "Policy"

    def test_root_by_curie(self):
        draft, _ = compile_fixture("policy_basic.ttl", root="ex:Policy")
        assert draft["root"] == "Policy"

    def test_root_by_full_iri(self):
        draft, _ = compile_fixture("policy_basic.ttl", root="http://example.org/insurance#Policy")
        assert draft["root"] == "Policy"

    def test_unknown_local_name_lists_available(self):
        with pytest.raises(ValueError, match="Available classes"):
            compile_fixture("policy_basic.ttl", root="Ghost")

    def test_ambiguous_auto_root_lists_candidates(self):
        # Fleet, FireCoverage, TheftCoverage, and Car are all no property's range.
        with pytest.raises(ValueError, match="candidates"):
            compile_fixture("subclass_flatten.ttl")

    def test_enum_class_rejected_as_root(self):
        with pytest.raises(ValueError, match="enumerated class"):
            compile_fixture("oneof_enum.ttl", root="Severity")

    def test_unknown_curie_prefix_fails(self):
        with pytest.raises(ValueError, match="CURIE"):
            compile_fixture("policy_basic.ttl", root="ghost:Policy")


# ---------------------------------------------------------------------------
# Class mapping: docstrings, labels, provenance
# ---------------------------------------------------------------------------


class TestClassMapping:
    def test_comment_becomes_docstring(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        docstring = models_by_name(draft)["Policy"]["docstring"]
        assert docstring.startswith("An insurance contract between an insurer and a policyholder.")

    def test_label_variant_becomes_also_called(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert "Also called: Insurance Policy." in models_by_name(draft)["Policy"]["docstring"]

    def test_module_docstring_from_ontology_header(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert draft["module_docstring"] == "Insurance policy extraction ontology."

    def test_source_ref_keeps_class_iri(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert (
            models_by_name(draft)["Policy"]["source_ref"] == "http://example.org/insurance#Policy"
        )

    def test_provenance_is_ontology(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert all(m["provenance"] == "ontology" for m in draft["models"])

    def test_missing_comment_gets_placeholder_and_gap(self):
        draft, gaps = compile_fixture("no_identity.ttl")
        remark = models_by_name(draft)["Remark"]
        assert remark["docstring"]  # non-empty placeholder keeps the draft valid
        assert any(g.model == "Remark" and g.kind == "missing_docstring" for g in gaps)


# ---------------------------------------------------------------------------
# Datatype properties (XSD_MAP, functional, open-world lists, examples)
# ---------------------------------------------------------------------------


class TestDatatypeProperties:
    @pytest.fixture()
    def policy(self) -> dict:
        draft, _ = compile_fixture("policy_basic.ttl")
        return models_by_name(draft)["Policy"]

    @pytest.mark.parametrize(
        ("field_name", "expected_type"),
        [
            ("premium", "float"),  # xsd:decimal
            ("active", "bool"),
            ("start_date", "date"),
            ("created_at", "datetime"),
            ("renewal_year", "str"),  # xsd:gYear is unmapped -> str
            ("note", "str"),
        ],
    )
    def test_xsd_map(self, policy, field_name, expected_type):
        assert field(policy, field_name)["type"] == expected_type

    def test_functional_property_is_single_valued(self, policy):
        assert field(policy, "premium")["is_list"] is False

    def test_non_functional_property_defaults_to_list(self, policy):
        # Open-world: repeatable unless restricted.
        assert field(policy, "note")["is_list"] is True

    def test_max_cardinality_one_forces_single(self, policy):
        assert field(policy, "footnote")["is_list"] is False

    def test_skos_examples_harvested(self, policy):
        assert field(policy, "policy_number")["examples"] == [
            "POL-2024-001",
            "POL-2024-002",
        ]

    def test_min_cardinality_noted_never_required(self, policy):
        description = field(policy, "policy_number")["description"]
        assert "Always present in conforming documents." in description

    def test_property_comment_becomes_description(self, policy):
        assert field(policy, "policy_number")["description"].startswith(
            "Unique policy number printed on the schedule."
        )


# ---------------------------------------------------------------------------
# Object properties (edges) + cardinality restrictions
# ---------------------------------------------------------------------------


class TestObjectProperties:
    def test_object_property_becomes_edge(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        covers = field(models_by_name(draft)["Policy"], "covers")
        assert covers["role"] == "edge"
        assert covers["type"] == "Guarantee"
        assert covers["edge_label"] == "COVERS"
        assert covers["is_list"] is True

    def test_max_qualified_cardinality_keeps_documented_max(self):
        # Drafts carry the DOCUMENTED maximum; repair_draft doubles exactly once.
        draft, _ = compile_fixture("policy_basic.ttl")
        guarantee = models_by_name(draft)["Guarantee"]
        assert guarantee["max_instances"] == 6

    def test_repair_draft_doubles_documented_max_exactly_once(self):
        # End-to-end R13 contract: graph_max_instances == 2 x the documented n
        # (a doubled draft would end up at 4x — a live runtime bound).
        draft, _ = spec_draft_from_ontology(FIXTURES / "policy_basic.ttl")
        spec, _ = repair_draft(draft)
        guarantee = next(m for m in spec.models if m.name == "Guarantee")
        assert guarantee.max_instances == 12  # 2 x documented max of 6

    def test_cardinality_sentence_lands_in_target_docstring(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert "At most 6 per document." in models_by_name(draft)["Guarantee"]["docstring"]

    def test_restriction_blank_nodes_never_emitted(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert {m["name"] for m in draft["models"]} == {"Policy", "Guarantee"}

    def test_domainless_property_skipped_with_note(self):
        draft, _ = compile_fixture("no_identity.ttl")
        assert any("orphanProp" in note for note in draft["generator"]["notes"])

    def test_union_domain_attaches_to_each_member(self):
        draft, _ = compile_fixture("cycles.ttl")
        by_name = models_by_name(draft)
        assert field(by_name["PartA"], "part_code")
        assert field(by_name["PartB"], "part_code")


# ---------------------------------------------------------------------------
# Identity: hasKey > InverseFunctional > heuristic ladder > demotion
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_has_key_becomes_identity(self):
        draft, _ = compile_fixture("haskey.ttl")
        contract = models_by_name(draft)["Contract"]
        assert contract["identity_fields"] == ["contract_number", "contract_series"]

    def test_has_key_beyond_two_truncated_with_gap(self):
        draft, gaps = compile_fixture("haskey.ttl")
        amendment = models_by_name(draft)["Amendment"]
        assert amendment["identity_fields"] == ["amendment_id", "amendment_date"]
        assert any(
            g.model == "Amendment" and g.field == "amendment_seq" and g.kind == "missing_identity"
            for g in gaps
        )

    def test_inverse_functional_datatype_becomes_identity(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert models_by_name(draft)["Policy"]["identity_fields"] == ["policy_number"]

    def test_heuristic_ladder_matches_name_suffix(self):
        draft, _ = compile_fixture("policy_basic.ttl")
        assert models_by_name(draft)["Guarantee"]["identity_fields"] == ["guarantee_name"]

    def test_identity_fields_are_scalar_even_when_open_world(self):
        # guarantee_name is not functional (would default to a list).
        draft, _ = compile_fixture("policy_basic.ttl")
        guarantee_name = field(models_by_name(draft)["Guarantee"], "guarantee_name")
        assert guarantee_name["is_list"] is False
        assert guarantee_name["role"] == "identity"

    def test_no_identity_demotes_to_component_with_gap(self):
        draft, gaps = compile_fixture("no_identity.ttl")
        remark = models_by_name(draft)["Remark"]
        assert remark["kind"] == "component"
        assert remark["identity_fields"] == []
        assert any(g.model == "Remark" and g.kind == "missing_identity" for g in gaps)

    def test_edge_to_demoted_component_is_kept(self):
        draft, _ = compile_fixture("no_identity.ttl")
        has_remark = field(models_by_name(draft)["Report"], "has_remark")
        assert has_remark["role"] == "edge"
        assert has_remark["type"] == "Remark"

    def test_identity_less_root_synthesizes_document_reference(self):
        draft, gaps = compile_fixture("rootless.ttl")
        cover_note = models_by_name(draft)["CoverNote"]
        assert cover_note["kind"] == "root"
        assert cover_note["identity_fields"] == ["document_reference"]
        reference = field(cover_note, "document_reference")
        assert reference["description"] == (
            "Identifier printed on the document, e.g. reference number or title."
        )
        assert any(
            g.model == "CoverNote"
            and g.field == "document_reference"
            and g.kind == "missing_identity"
            for g in gaps
        )

    def test_identity_without_examples_raises_gap(self):
        _, gaps = compile_fixture("policy_basic.ttl")
        assert any(
            g.model == "Guarantee" and g.field == "guarantee_name" and g.kind == "missing_examples"
            for g in gaps
        )

    def test_identity_with_harvested_examples_has_no_gap(self):
        _, gaps = compile_fixture("policy_basic.ttl")
        assert not any(g.field == "policy_number" and g.kind == "missing_examples" for g in gaps)


# ---------------------------------------------------------------------------
# Subclass flattening
# ---------------------------------------------------------------------------


class TestSubclassFlattening:
    @pytest.fixture()
    def draft(self) -> dict:
        draft, _ = compile_fixture("subclass_flatten.ttl", root="Fleet")
        return draft

    def test_abstract_parent_dropped(self, draft):
        assert "Coverage" not in models_by_name(draft)

    def test_edge_to_abstract_parent_fans_out_per_child(self, draft):
        fleet = models_by_name(draft)["Fleet"]
        fire = field(fleet, "includes_coverage_fire_coverage")
        theft = field(fleet, "includes_coverage_theft_coverage")
        assert fire["type"] == "FireCoverage"
        assert theft["type"] == "TheftCoverage"
        assert fire["edge_label"] == theft["edge_label"] == "INCLUDES_COVERAGE"

    def test_siblings_get_is_not_clauses(self, draft):
        by_name = models_by_name(draft)
        assert "NOT a TheftCoverage." in by_name["FireCoverage"]["docstring"]
        assert "NOT a FireCoverage." in by_name["TheftCoverage"]["docstring"]

    def test_parent_properties_pushed_down_to_children(self, draft):
        car = models_by_name(draft)["Car"]
        assert field(car, "vin")["type"] == "str"  # inherited from Vehicle
        assert field(car, "seat_count")["type"] == "int"  # its own

    def test_concrete_parent_with_one_child_is_kept(self, draft):
        by_name = models_by_name(draft)
        assert "Vehicle" in by_name
        assert "Car" in by_name

    def test_union_domain_property_lands_on_each_child(self, draft):
        by_name = models_by_name(draft)
        assert field(by_name["FireCoverage"], "coverage_name")
        assert field(by_name["TheftCoverage"], "coverage_name")


class TestNestedAbstractFanOut:
    """An edge to a dropped abstract expands TRANSITIVELY through nested
    dropped abstracts to the emitted leaves (never orphaning grandchildren)."""

    @pytest.fixture()
    def draft(self) -> dict:
        # Fleet-[owns]->Vehicle(abstract){Car(abstract){Sedan, Suv}, Boat}
        draft, _ = compile_fixture("fleet_deep.ttl", root="Fleet")
        return draft

    def test_nested_abstracts_not_emitted(self, draft):
        assert set(models_by_name(draft)) == {"Fleet", "Boat", "Sedan", "Suv"}

    def test_fan_out_reaches_grandchildren_through_nested_abstract(self, draft):
        fleet = models_by_name(draft)["Fleet"]
        sedan = field(fleet, "owns_sedan")
        suv = field(fleet, "owns_suv")
        boat = field(fleet, "owns_boat")
        assert sedan["type"] == "Sedan"
        assert suv["type"] == "Suv"
        assert boat["type"] == "Boat"
        assert sedan["role"] == suv["role"] == boat["role"] == "edge"

    def test_fan_out_edges_share_the_property_label(self, draft):
        fleet = models_by_name(draft)["Fleet"]
        labels = {
            field(fleet, name)["edge_label"] for name in ("owns_sedan", "owns_suv", "owns_boat")
        }
        assert len(labels) == 1

    def test_no_leaf_model_is_orphaned(self, draft):
        # Every emitted non-root model has at least one inbound edge.
        targets = {
            f["type"] for m in draft["models"] for f in m["fields"] if f.get("role") == "edge"
        }
        assert {"Sedan", "Suv", "Boat"} <= targets


# ---------------------------------------------------------------------------
# Enums: owl:oneOf and SKOS
# ---------------------------------------------------------------------------


class TestEnums:
    def test_one_of_becomes_enum_with_other_fallback(self):
        draft, _ = compile_fixture("oneof_enum.ttl")
        assert draft["enums"] == [
            {
                "name": "Severity",
                "members": ["Low", "Medium", "High"],  # RDF list order preserved
                "synonyms": {"Low": ["minor"]},  # skos:altLabel
                # Even a provably total owl:oneOf keeps OTHER: the generated
                # normalizer must coerce-and-warn, never raise (R17) — the
                # ontology's own altLabel synonyms would otherwise crash it.
                "include_other": True,
            }
        ]

    def test_enum_class_not_emitted_as_model(self):
        draft, _ = compile_fixture("oneof_enum.ttl")
        assert "Severity" not in models_by_name(draft)

    def test_enum_ranged_property_becomes_enum_field(self):
        draft, _ = compile_fixture("oneof_enum.ttl")
        severity = field(models_by_name(draft)["Doc"], "severity")
        assert severity["type"] == "Severity"
        assert severity["role"] == "property"
        assert severity["normalizer"] == "enum"
        assert severity["is_list"] is False  # functional

    def test_skos_scheme_backed_class_becomes_enum(self):
        draft, _ = compile_fixture("skos_scheme.ttl")
        assert draft["enums"] == [
            {
                "name": "CoverageType",
                "members": ["Fire", "Theft"],  # prefLabels
                "synonyms": {"Fire": ["Fire damage", "Incendie"]},
                "include_other": True,  # every enum source keeps OTHER (R17)
            }
        ]

    def test_skos_only_vocabulary_raises_with_enums_attached(self):
        with pytest.raises(SkosOnlyOntologyError, match="from-docs") as exc_info:
            compile_fixture("skos_only.ttl")
        assert exc_info.value.enums == [
            {
                "name": "Countries",
                "members": ["France", "Germany"],
                "synonyms": {"France": ["FR"]},
                "include_other": True,
            }
        ]


# ---------------------------------------------------------------------------
# Cycles + multi-path references (canonical home)
# ---------------------------------------------------------------------------


class TestCyclesAndReferences:
    def test_cyclic_ontology_compiles(self):
        draft, _ = compile_fixture("cycles.ttl")
        assert {m["name"] for m in draft["models"]} == {"Doc", "PartA", "PartB"}

    def test_canonical_home_is_shallowest_path(self):
        draft, _ = compile_fixture("cycles.ttl")
        assert models_by_name(draft)["PartA"]["canonical_home"] == "Doc.has_part"

    def test_cycle_back_edges_marked_reference(self):
        draft, _ = compile_fixture("cycles.ttl")
        by_name = models_by_name(draft)
        assert field(by_name["PartB"], "belongs_to")["reference"] is True
        assert field(by_name["PartA"], "extends_part")["reference"] is True
        # The canonical path stays a full edge.
        assert field(by_name["Doc"], "has_part").get("reference") is not True

    def test_single_path_target_not_referenced(self):
        draft, _ = compile_fixture("cycles.ttl")
        references = field(models_by_name(draft)["PartA"], "references_part")
        assert references.get("reference") is not True
        assert models_by_name(draft)["PartB"]["canonical_home"] is None

    def test_canonical_home_prefers_richest_source(self):
        # Order declares 3 datatype properties, Shipment only 1.
        draft, _ = compile_fixture("multipath_reference.ttl")
        by_name = models_by_name(draft)
        assert by_name["Item"]["canonical_home"] == "Order.contains_item"
        assert field(by_name["Shipment"], "ships_item")["reference"] is True
        assert field(by_name["Order"], "contains_item").get("reference") is not True


class TestSameParentMultiRole:
    """Multi-role exception: single-valued edges from ONE parent to the same
    target are distinct roles (insurer/reinsurer), not duplicate nesting —
    flipping one to a reference would lose extractable data (R10's linter
    exception keys on the same shape, so the flip would be unrecoverable)."""

    @pytest.fixture()
    def draft(self) -> dict:
        draft, _ = compile_fixture("policy_multirole.ttl")
        return draft

    def test_both_roles_stay_full_edges(self, draft):
        policy = models_by_name(draft)["Policy"]
        assert field(policy, "insurer").get("reference") is not True
        assert field(policy, "reinsurer").get("reference") is not True

    def test_canonical_home_still_recorded(self, draft):
        assert models_by_name(draft)["Company"]["canonical_home"] == "Policy.insurer"

    def test_same_parent_with_list_edge_still_flips(self, draft):
        # lead_broker (single) + panel_broker (list) from the same parent is
        # NOT the multi-role shape: the non-canonical path becomes a reference.
        policy = models_by_name(draft)["Policy"]
        assert field(policy, "panel_broker")["reference"] is True
        assert field(policy, "lead_broker").get("reference") is not True
        assert models_by_name(draft)["Broker"]["canonical_home"] == "Policy.lead_broker"

    def test_multi_parent_case_still_flips(self):
        # Regression guard: the exception must not weaken genuine multi-parent
        # flipping (Order and Shipment both nest Item).
        draft, _ = compile_fixture("multipath_reference.ttl")
        assert field(models_by_name(draft)["Shipment"], "ships_item")["reference"] is True


# ---------------------------------------------------------------------------
# owl:equivalentClass collapse
# ---------------------------------------------------------------------------


class TestEquivalentClasses:
    def test_equivalents_collapse_to_richest_representative(self):
        draft, _ = compile_fixture("equivalent.ttl")
        by_name = models_by_name(draft)
        assert "Firm" in by_name
        assert "Enterprise" not in by_name

    def test_equivalent_properties_merged(self):
        draft, _ = compile_fixture("equivalent.ttl")
        firm = models_by_name(draft)["Firm"]
        names = {f["name"] for f in firm["fields"]}
        assert {"firm_name", "vat_code", "employs_person"} <= names

    def test_collapse_reported(self):
        draft, _ = compile_fixture("equivalent.ttl")
        assert any("equivalentClass" in note for note in draft["generator"]["notes"])


# ---------------------------------------------------------------------------
# Pruning: depth / include / exclude
# ---------------------------------------------------------------------------


class TestPruning:
    def test_depth_bounds_the_closure(self):
        draft, _ = compile_fixture("chain.ttl", depth=1)
        assert {m["name"] for m in draft["models"]} == {"Ledger", "Account"}
        assert any("contains_entry" in note for note in draft["generator"]["notes"])

    def test_default_depth_reaches_the_full_chain(self):
        draft, _ = compile_fixture("chain.ttl")
        assert {m["name"] for m in draft["models"]} == {
            "Ledger",
            "Account",
            "Entry",
            "Detail",
        }

    def test_exclude_prunes_class_and_its_edges(self):
        draft, _ = compile_fixture("policy_basic.ttl", exclude=["Guarantee"])
        assert {m["name"] for m in draft["models"]} == {"Policy"}
        assert not any(f["name"] == "covers" for f in draft["models"][0]["fields"])

    def test_include_keeps_matching_classes_and_root(self):
        draft, _ = compile_fixture("policy_basic.ttl", include=["Guar*"])
        assert {m["name"] for m in draft["models"]} == {"Policy", "Guarantee"}

    def test_include_drops_non_matching_classes(self):
        draft, _ = compile_fixture("policy_basic.ttl", include=["Nothing*"])
        assert {m["name"] for m in draft["models"]} == {"Policy"}


# ---------------------------------------------------------------------------
# Draft validity + dispatcher + dependency guard
# ---------------------------------------------------------------------------

OWL_FIXTURES = [
    ("policy_basic.ttl", None),
    ("haskey.ttl", None),
    ("oneof_enum.ttl", None),
    ("subclass_flatten.ttl", "Fleet"),
    ("no_identity.ttl", None),
    ("rootless.ttl", None),
    ("cycles.ttl", None),
    ("multipath_reference.ttl", None),
    ("policy_multirole.ttl", None),
    ("fleet_deep.ttl", "Fleet"),
    ("skos_scheme.ttl", None),
    ("equivalent.ttl", None),
    ("chain.ttl", None),
]


class TestDraftValidity:
    @pytest.mark.parametrize(("fixture", "root"), OWL_FIXTURES)
    def test_draft_validates_directly(self, fixture, root):
        # Constructibility: repair_draft is a safety net, not a crutch.
        draft, _ = compile_fixture(fixture, root=root)
        spec = TemplateSpec.model_validate(draft)
        assert spec.root == draft["root"]


class TestDispatcher:
    def test_auto_format_dispatches_ttl_to_owl(self):
        draft, gaps = spec_draft_from_ontology(FIXTURES / "policy_basic.ttl")
        assert draft["generator"]["format"] == "owl"
        assert draft["root"] == "Policy"
        assert all(g.kind for g in gaps)

    def test_explicit_owl_format(self):
        draft, _ = spec_draft_from_ontology(FIXTURES / "policy_basic.ttl", fmt="owl")
        assert draft["root"] == "Policy"


class TestMissingDependency:
    def test_import_error_names_the_extra(self, monkeypatch):
        for module in list(sys.modules):
            if module == "rdflib" or module.startswith("rdflib."):
                monkeypatch.delitem(sys.modules, module)
        monkeypatch.setitem(sys.modules, "rdflib", None)
        with pytest.raises(ImportError, match=r"docling-graph\[templategen\]"):
            spec_draft_from_owl(FIXTURES / "policy_basic.ttl")
