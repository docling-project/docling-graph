"""Unit tests for the TemplateSpec IR intrinsic validators and YAML round-trip."""

import pytest
from pydantic import ValidationError

from docling_graph.templategen.spec import (
    EnumSpec,
    FieldSpec,
    ModelSpec,
    SpecGap,
    TemplateSpec,
)

# ---------------------------------------------------------------------------
# Builders: a minimal compliant spec each test mutates in exactly one way.
# ---------------------------------------------------------------------------


def make_identity_field(name: str = "name") -> FieldSpec:
    return FieldSpec(
        name=name,
        type="str",
        role="identity",
        description="Name as printed in the document.",
        examples=["Acme Corp", "Beta SARL"],
    )


def make_party() -> ModelSpec:
    return ModelSpec(
        name="Party",
        kind="entity",
        docstring="An organization or person involved in the document.",
        identity_fields=["name"],
        fields=[
            make_identity_field(),
            FieldSpec(name="tax_id", type="str", description="VAT number if printed."),
        ],
    )


def make_amount() -> ModelSpec:
    return ModelSpec(
        name="Amount",
        kind="component",
        docstring="Monetary value component, deduplicated by content.",
        fields=[
            FieldSpec(name="value", type="float", normalizer="numeric"),
            FieldSpec(name="currency", type="str", normalizer="currency"),
        ],
    )


def make_root() -> ModelSpec:
    return ModelSpec(
        name="Invoice",
        kind="root",
        docstring="The invoice document, identified by its printed number.",
        identity_fields=["document_number"],
        fields=[
            make_identity_field("document_number"),
            FieldSpec(
                name="document_type",
                type="DocumentType",
                normalizer="enum",
                description="Document title text.",
            ),
            FieldSpec(
                name="seller",
                type="Party",
                role="edge",
                edge_label="ISSUED_BY",
                description="The party that issued this document.",
            ),
            FieldSpec(
                name="mentioned_parties",
                type="Party",
                role="edge",
                edge_label="MENTIONS",
                is_list=True,
                reference=True,
                description="Parties referenced by name only.",
            ),
            FieldSpec(name="keywords", type="str", is_list=True),
            FieldSpec(name="total", type="Amount", description="Grand total block."),
        ],
    )


def make_spec(**overrides: object) -> TemplateSpec:
    payload: dict = {
        "module_docstring": "Invoice extraction template.",
        "root": "Invoice",
        "enums": [EnumSpec(name="DocumentType", members=["Invoice", "Credit Note"])],
        "models": [make_amount(), make_party(), make_root()],
        "needs_root_list_dedup": ["keywords"],
        "generator": {"version": "0.1.0"},
    }
    payload.update(overrides)
    return TemplateSpec(**payload)


# ---------------------------------------------------------------------------
# EnumSpec
# ---------------------------------------------------------------------------


class TestEnumSpec:
    def test_valid_enum_passes(self):
        spec = EnumSpec(
            name="Status",
            members=["Active", "Closed"],
            synonyms={"Active": ["in force", "current"]},
        )
        assert spec.include_other is True

    def test_empty_members_rejected(self):
        with pytest.raises(ValidationError):
            EnumSpec(name="Status", members=[])

    def test_synonyms_for_unknown_member_rejected(self):
        with pytest.raises(ValidationError, match="unknown members"):
            EnumSpec(name="Status", members=["Active"], synonyms={"Gone": ["x"]})


# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------


class TestFieldSpec:
    def test_valid_property_passes(self):
        field = FieldSpec(name="city", type="str")
        assert field.role == "property"

    def test_valid_edge_passes(self):
        field = FieldSpec(name="seller", type="Party", role="edge", edge_label="ISSUED_BY")
        assert field.edge_label == "ISSUED_BY"

    def test_closed_catalog_requires_reference(self):
        with pytest.raises(ValidationError, match="closed_catalog"):
            FieldSpec(
                name="items",
                type="Item",
                role="edge",
                edge_label="EXCLUDES",
                closed_catalog=True,
            )

    def test_closed_catalog_with_reference_passes(self):
        field = FieldSpec(
            name="items",
            type="Item",
            role="edge",
            edge_label="EXCLUDES",
            is_list=True,
            reference=True,
            closed_catalog=True,
        )
        assert field.closed_catalog is True

    def test_edge_without_label_rejected(self):
        with pytest.raises(ValidationError, match="edge_label"):
            FieldSpec(name="seller", type="Party", role="edge")

    def test_label_on_non_edge_rejected(self):
        with pytest.raises(ValidationError, match="edge_label"):
            FieldSpec(name="city", type="str", edge_label="LOCATED_AT")

    def test_reference_on_property_rejected(self):
        with pytest.raises(ValidationError, match="edge-only"):
            FieldSpec(name="city", type="str", reference=True)

    def test_identity_list_rejected(self):
        with pytest.raises(ValidationError, match="scalar"):
            FieldSpec(name="names", type="str", role="identity", is_list=True)

    def test_identity_non_scalar_type_rejected(self):
        # Any non-scalar type string names an enum or model — both forbidden.
        with pytest.raises(ValidationError, match="scalar-typed"):
            FieldSpec(name="status", type="StatusEnum", role="identity")

    def test_identity_with_edge_label_rejected(self):
        with pytest.raises(ValidationError, match="edge_label"):
            FieldSpec(name="name", type="str", role="identity", edge_label="HAS_NAME")

    def test_identity_with_normalizer_rejected(self):
        with pytest.raises(ValidationError, match="verbatim"):
            FieldSpec(name="name", type="str", role="identity", normalizer="string_list")

    def test_normalizer_on_edge_rejected(self):
        with pytest.raises(ValidationError, match="property-only"):
            FieldSpec(
                name="seller",
                type="Party",
                role="edge",
                edge_label="ISSUED_BY",
                normalizer="enum",
            )

    def test_more_than_five_examples_rejected(self):
        with pytest.raises(ValidationError):
            FieldSpec(name="name", type="str", examples=["a", "b", "c", "d", "e", "f"])

    def test_scalar_examples_coerced_to_strings(self):
        # Hand-edited YAML often carries unquoted numbers ('examples: [1, 2]').
        field = FieldSpec(name="line_number", type="str", examples=[1, 2.5, "A1"])
        assert field.examples == ["1", "2.5", "A1"]


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_valid_entity_passes(self):
        assert make_party().kind == "entity"

    def test_valid_component_passes(self):
        assert make_amount().identity_fields == []

    def test_component_with_identity_fields_rejected(self):
        with pytest.raises(ValidationError, match="identity_fields"):
            ModelSpec(
                name="Amount",
                kind="component",
                docstring="A value block.",
                identity_fields=["value"],
                fields=[make_identity_field("value")],
            )

    def test_component_with_max_instances_rejected(self):
        with pytest.raises(ValidationError, match="max_instances"):
            ModelSpec(
                name="Amount",
                kind="component",
                docstring="A value block.",
                max_instances=4,
                fields=[FieldSpec(name="value", type="float")],
            )

    def test_entity_without_identity_rejected(self):
        with pytest.raises(ValidationError, match="1-2 identity"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                fields=[FieldSpec(name="name", type="str")],
            )

    def test_entity_with_three_identity_fields_rejected(self):
        with pytest.raises(ValidationError, match="1-2 identity"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                identity_fields=["a", "b", "c"],
                fields=[
                    make_identity_field("a"),
                    make_identity_field("b"),
                    make_identity_field("c"),
                ],
            )

    def test_identity_field_missing_from_fields_rejected(self):
        with pytest.raises(ValidationError, match="not declared"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                identity_fields=["name"],
                fields=[FieldSpec(name="tax_id", type="str")],
            )

    def test_identity_field_with_wrong_role_rejected(self):
        with pytest.raises(ValidationError, match="role='identity'"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                identity_fields=["name"],
                fields=[FieldSpec(name="name", type="str", role="property")],
            )

    def test_unlisted_identity_role_field_rejected(self):
        with pytest.raises(ValidationError, match=r"not\s+listed in identity_fields"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                identity_fields=["name"],
                fields=[make_identity_field("name"), make_identity_field("tax_id")],
            )

    def test_empty_docstring_rejected(self):
        with pytest.raises(ValidationError, match="docstring"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="   ",
                identity_fields=["name"],
                fields=[make_identity_field()],
            )

    def test_duplicate_field_names_rejected(self):
        with pytest.raises(ValidationError, match="duplicate field names"):
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party.",
                identity_fields=["name"],
                fields=[
                    make_identity_field("name"),
                    FieldSpec(name="tax_id", type="str"),
                    FieldSpec(name="tax_id", type="str"),
                ],
            )

    def test_root_requires_identity_too(self):
        with pytest.raises(ValidationError, match="1-2 identity"):
            ModelSpec(
                name="Invoice",
                kind="root",
                docstring="The document.",
                fields=[FieldSpec(name="notes", type="str")],
            )


# ---------------------------------------------------------------------------
# TemplateSpec
# ---------------------------------------------------------------------------


class TestTemplateSpec:
    def test_valid_spec_passes(self):
        spec = make_spec()
        assert spec.root == "Invoice"
        assert spec.generator == {"version": "0.1.0"}

    def test_no_root_model_rejected(self):
        with pytest.raises(ValidationError, match="exactly one model"):
            make_spec(models=[make_amount(), make_party()], root="Party")

    def test_two_root_models_rejected(self):
        second_root = make_root()
        second_root.name = "Invoice2"
        with pytest.raises(ValidationError, match="exactly one model"):
            make_spec(models=[make_amount(), make_party(), make_root(), second_root])

    def test_root_name_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="does not name the root model"):
            make_spec(root="Party")

    def test_unresolved_field_type_rejected(self):
        root = make_root()
        root.fields[1].type = "GhostEnum"
        with pytest.raises(ValidationError, match="unresolved"):
            make_spec(models=[make_amount(), make_party(), root])

    def test_edge_targeting_enum_rejected(self):
        root = make_root()
        root.fields[2].type = "DocumentType"
        with pytest.raises(ValidationError, match="must target a"):
            make_spec(models=[make_amount(), make_party(), root])

    def test_duplicate_model_names_rejected(self):
        with pytest.raises(ValidationError, match="duplicate model/enum names"):
            make_spec(models=[make_amount(), make_party(), make_party(), make_root()])

    def test_model_and_enum_name_collision_rejected(self):
        with pytest.raises(ValidationError, match="duplicate model/enum names"):
            make_spec(enums=[EnumSpec(name="Party", members=["A"])])

    def test_canonical_home_resolves(self):
        party = make_party()
        party.canonical_home = "Invoice.seller"
        spec = make_spec(models=[make_amount(), party, make_root()])
        assert spec.models[1].canonical_home == "Invoice.seller"

    def test_canonical_home_unknown_parent_rejected(self):
        party = make_party()
        party.canonical_home = "Ghost.seller"
        with pytest.raises(ValidationError, match=r"not a\s+declared model"):
            make_spec(models=[make_amount(), party, make_root()])

    def test_canonical_home_unknown_field_rejected(self):
        party = make_party()
        party.canonical_home = "Invoice.ghost_field"
        with pytest.raises(ValidationError, match="does not exist"):
            make_spec(models=[make_amount(), party, make_root()])

    def test_canonical_home_without_dot_rejected(self):
        party = make_party()
        party.canonical_home = "Invoice"
        with pytest.raises(ValidationError, match=r"ParentModel\.field_name"):
            make_spec(models=[make_amount(), party, make_root()])

    def test_dedup_field_must_exist_on_root(self):
        with pytest.raises(ValidationError, match="unknown root field"):
            make_spec(needs_root_list_dedup=["ghosts"])

    def test_dedup_field_must_be_list(self):
        with pytest.raises(ValidationError, match="is not a list"):
            make_spec(needs_root_list_dedup=["document_type"])


# ---------------------------------------------------------------------------
# SpecGap
# ---------------------------------------------------------------------------


class TestSpecGap:
    def test_valid_gap(self):
        gap = SpecGap(model="Party", field="name", kind="missing_examples", note="no evidence")
        assert gap.kind == "missing_examples"

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            SpecGap(model="Party", kind="missing_everything")


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_round_trip_equality(self):
        spec = make_spec()
        restored = TemplateSpec.from_yaml(spec.to_yaml())
        assert restored == spec

    def test_round_trip_preserves_generator_audit_block(self):
        spec = make_spec(generator={"version": "0.1.0", "sources": ["a.pdf", "b.pdf"]})
        restored = TemplateSpec.from_yaml(spec.to_yaml())
        assert restored.generator == {"version": "0.1.0", "sources": ["a.pdf", "b.pdf"]}

    def test_yaml_is_stable(self):
        spec = make_spec()
        assert spec.to_yaml() == TemplateSpec.from_yaml(spec.to_yaml()).to_yaml()

    def test_from_yaml_rejects_non_mapping(self):
        with pytest.raises(ValueError, match="mapping"):
            TemplateSpec.from_yaml("- just\n- a\n- list\n")

    def test_from_yaml_validates(self):
        spec = make_spec()
        broken = spec.to_yaml().replace("root: Invoice", "root: Party")
        with pytest.raises(ValidationError):
            TemplateSpec.from_yaml(broken)
