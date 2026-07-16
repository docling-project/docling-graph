"""Unit tests for the rulebook linter (templategen.linter).

One test class per implemented rule: a violating spec/draft goes in, the
repaired spec plus the expected report entry come out; strict mode raises for
repairing rules; ``repair=False`` reports without mutating.
"""

from typing import Any

import pytest

from docling_graph.templategen.linter import (
    MAX_LINT_PASSES,
    LintReport,
    TemplateLintError,
    lint_spec,
    repair_draft,
)
from docling_graph.templategen.spec import EnumSpec, FieldSpec, ModelSpec, TemplateSpec

# ---------------------------------------------------------------------------
# Compact factories (valid-spec path). provenance="user" keeps R22 quiet.
# ---------------------------------------------------------------------------


def fld(name, **kw: Any) -> FieldSpec:
    kw.setdefault("type", "str")
    return FieldSpec(name=name, **kw)


def ident(name="name", examples=("Acme Corp", "Beta SARL"), **kw: Any) -> FieldSpec:
    return fld(name, role="identity", examples=list(examples), **kw)


def edge_fld(name, target, label, **kw: Any) -> FieldSpec:
    return fld(name, type=target, role="edge", edge_label=label, **kw)


def mdl(
    name, kind="entity", *, fields, identity=None, docstring=None, provenance="user", **kw: Any
) -> ModelSpec:
    if identity is None:
        identity = [f.name for f in fields if f.role == "identity"]
    if kind == "component":
        identity = []
    return ModelSpec(
        name=name,
        kind=kind,
        docstring=docstring or f"A {name} record from the document.",
        identity_fields=identity,
        provenance=provenance,
        fields=list(fields),
        **kw,
    )


def make_root(*extra_fields: FieldSpec, **kw: Any) -> ModelSpec:
    return mdl(
        "Doc",
        "root",
        fields=[ident("document_number", ("INV-001", "INV-002")), *extra_fields],
        **kw,
    )


def party_model(*extra_fields: FieldSpec, **kw: Any) -> ModelSpec:
    return mdl(
        "Party",
        fields=[ident("name", ("Acme Corp", "Beta SARL")), fld("tax_id"), *extra_fields],
        **kw,
    )


def build_spec(*models: ModelSpec, enums=(), **kw: Any) -> TemplateSpec:
    root = next(m.name for m in models if m.kind == "root")
    payload = {
        "module_docstring": "Test template.",
        "root": root,
        "enums": list(enums),
        "models": list(models),
    }
    payload.update(kw)
    return TemplateSpec(**payload)


def get_model(spec: TemplateSpec, name: str) -> ModelSpec:
    return next(m for m in spec.models if m.name == name)


def get_field(spec: TemplateSpec, model: str, field: str) -> FieldSpec:
    return next(f for f in get_model(spec, model).fields if f.name == field)


def repairs(report: LintReport, rule_id: str) -> list:
    return [e for e in report.by_rule(rule_id) if e.repaired]


# ---------------------------------------------------------------------------
# Draft factories (repair_draft path — loose dicts in TemplateSpec shape).
# ---------------------------------------------------------------------------


def dfld(name, **kw: Any) -> dict:
    payload = {"name": name, "type": "str"}
    payload.update(kw)
    return payload


def dmdl(name, kind, fields, **kw: Any) -> dict:
    payload = {"name": name, "kind": kind, "docstring": f"A {name}.", "fields": list(fields)}
    payload.update(kw)
    return payload


def droot(*fields: dict, **kw: Any) -> dict:
    return dmdl(
        "Doc",
        "root",
        [dfld("document_number", role="identity", examples=["INV-001", "INV-002"]), *fields],
        identity_fields=["document_number"],
        **kw,
    )


def ddraft(*models: dict, root="Doc", **kw: Any) -> dict:
    payload = {"root": root, "models": list(models)}
    payload.update(kw)
    return payload


# ---------------------------------------------------------------------------
# R1 — identity shape (pre-validation half in repair_draft)
# ---------------------------------------------------------------------------


class TestR1IdentityShape:
    def test_entity_without_identity_demoted_to_component(self):
        draft = ddraft(droot(), dmdl("Note", "entity", [dfld("text")]))
        spec, report = repair_draft(draft)
        assert get_model(spec, "Note").kind == "component"
        assert any(e.model == "Note" for e in repairs(report, "R1"))

    def test_three_identity_fields_trimmed_to_two_best(self):
        fields = [
            dfld("long_name", role="identity", examples=["Acme Corporation Limited"]),
            dfld("code", role="identity", examples=["INV-001", "INV-002"]),
            dfld("short", role="identity", examples=["XY"]),
        ]
        model = dmdl("Party", "entity", fields, identity_fields=["long_name", "code", "short"])
        spec, report = repair_draft(ddraft(droot(), model))
        party = get_model(spec, "Party")
        # digit-bearing examples first (code), then shortest example (short)
        assert party.identity_fields == ["code", "short"]
        assert get_field(spec, "Party", "long_name").role == "property"
        assert any(e.field == "long_name" for e in repairs(report, "R1"))

    def test_identity_less_root_gets_synthesized_document_reference(self):
        spec, report = repair_draft(ddraft(dmdl("Doc", "root", [dfld("notes")])))
        root = get_model(spec, "Doc")
        assert root.identity_fields == ["document_reference"]
        assert root.fields[0].name == "document_reference"
        assert root.fields[0].role == "identity"
        assert any(g.kind == "missing_identity" and g.model == "Doc" for g in report.gaps)

    def test_component_identity_and_max_instances_stripped(self):
        model = dmdl(
            "Amount",
            "component",
            [dfld("value", role="identity", examples=["1", "2"])],
            identity_fields=["value"],
            max_instances=4,
        )
        spec, report = repair_draft(ddraft(droot(), model))
        amount = get_model(spec, "Amount")
        assert amount.identity_fields == []
        assert amount.max_instances is None
        assert get_field(spec, "Amount", "value").role == "property"
        assert repairs(report, "R1")

    def test_strict_raises_listing_r1(self):
        draft = ddraft(droot(), dmdl("Note", "entity", [dfld("text")]))
        with pytest.raises(TemplateLintError, match="R1"):
            repair_draft(draft, strict=True)


# ---------------------------------------------------------------------------
# R2 — identity typing (pre-validation half in repair_draft)
# ---------------------------------------------------------------------------


class TestR2IdentityTyping:
    def test_identity_retyped_descaled_and_denormalized(self):
        enum = {"name": "Status", "members": ["Open", "Closed"]}
        field = dfld(
            "status",
            type="Status",
            role="identity",
            is_list=True,
            normalizer="numeric",
            examples=["Open", "Closed"],
        )
        model = dmdl("Ticket", "entity", [field], identity_fields=["status"])
        spec, report = repair_draft(ddraft(droot(), model, enums=[enum]))
        status = get_field(spec, "Ticket", "status")
        assert status.type == "str"
        assert status.is_list is False
        assert status.normalizer == "none"
        assert len(repairs(report, "R2")) == 3

    def test_strict_raises(self):
        field = dfld("status", role="identity", is_list=True, examples=["a", "b"])
        model = dmdl("Ticket", "entity", [field], identity_fields=["status"])
        with pytest.raises(TemplateLintError, match="R2"):
            repair_draft(ddraft(droot(), model), strict=True)


# ---------------------------------------------------------------------------
# R3 — identity examples 2-5
# ---------------------------------------------------------------------------


class TestR3IdentityExamples:
    def test_single_example_raises_gap(self):
        spec = build_spec(make_root(), mdl("Party", fields=[ident("name", ("Acme Corp",))]))
        _, report = lint_spec(spec)
        entries = report.by_rule("R3")
        assert entries and not entries[0].repaired
        assert any(g.kind == "missing_examples" and g.model == "Party" for g in report.gaps)

    def test_two_examples_pass(self):
        spec = build_spec(make_root(), party_model())
        _, report = lint_spec(spec)
        assert not report.by_rule("R3")


# ---------------------------------------------------------------------------
# R4 — 240-char docstring budget
# ---------------------------------------------------------------------------

_IS = "A reportable business segment as named by the segment note of the annual report filing."
_ISNOT = (
    "It is not a geographic region and not a revenue sub-line, which are excluded here entirely."
)
_CARD = (
    "There are at most 6 reportable segments in scope for this template across every filing year."
)


class TestR4DocstringBudget:
    def _spec(self, docstring) -> TemplateSpec:
        return build_spec(make_root(), mdl("Segment", fields=[ident()], docstring=docstring))

    def test_overlong_docstring_reordered_is_first(self):
        spec = self._spec(f"{_CARD} {_ISNOT} {_IS}")
        result, report = lint_spec(spec)
        assert get_model(result, "Segment").docstring == f"{_IS} {_ISNOT} {_CARD}"
        assert repairs(report, "R4")

    def test_exact_window_reported(self):
        spec = self._spec(f"{_CARD} {_ISNOT} {_IS}")
        _, report = lint_spec(spec)
        window = f"{_IS} {_ISNOT} {_CARD}"[:240]
        infos = [e for e in report.by_rule("R4") if e.severity == "info"]
        assert infos and window in infos[0].message

    def test_short_docstring_untouched_even_if_unordered(self):
        docstring = f"{_ISNOT} {_IS}"[:230]
        spec = self._spec(docstring)
        result, report = lint_spec(spec)
        assert get_model(result, "Segment").docstring == docstring
        assert not report.by_rule("R4")

    def test_strict_raises(self):
        with pytest.raises(TemplateLintError, match="R4"):
            lint_spec(self._spec(f"{_CARD} {_ISNOT} {_IS}"), strict=True)


# ---------------------------------------------------------------------------
# R5 — digit-honest identity names
# ---------------------------------------------------------------------------


class TestR5DigitHonesty:
    def test_number_named_identity_with_prose_renamed_to_name(self):
        party = mdl("Party", fields=[ident("party_number", ("Acme Corp", "Beta SARL"))])
        result, report = lint_spec(build_spec(make_root(), party))
        renamed = get_model(result, "Party")
        assert renamed.identity_fields == ["name"]
        assert renamed.fields[0].name == "name"
        assert repairs(report, "R5")

    def test_rename_prefers_title_when_name_taken(self):
        party = mdl(
            "Party",
            fields=[ident("party_number", ("Acme Corp", "Beta SARL")), fld("name")],
            identity=["party_number"],
        )
        result, _ = lint_spec(build_spec(make_root(), party))
        assert get_model(result, "Party").identity_fields == ["title"]

    def test_digit_bearing_number_name_untouched(self):
        result, report = lint_spec(build_spec(make_root()))
        assert get_model(result, "Doc").identity_fields == ["document_number"]
        assert not report.by_rule("R5")

    def test_strict_raises(self):
        party = mdl("Party", fields=[ident("party_number", ("Acme Corp", "Beta SARL"))])
        with pytest.raises(TemplateLintError, match="R5"):
            lint_spec(build_spec(make_root(), party), strict=True)


# ---------------------------------------------------------------------------
# R6 — invented / positional ids (advisory WARN)
# ---------------------------------------------------------------------------


class TestR6InventedIds:
    def test_positional_examples_stripped_with_warn_severity(self):
        item = mdl("Item", fields=[ident("name", ("ITEM 1", "XC-500"))])
        result, report = lint_spec(build_spec(make_root(), item))
        assert get_field(result, "Item", "name").examples == ["XC-500"]
        entries = repairs(report, "R6")
        assert entries and entries[0].severity == "warn"

    def test_underscore_variant_of_class_name_stripped(self):
        item = mdl("LineItem", fields=[ident("name", ("line_item-2", "USB Cable"))])
        result, _ = lint_spec(build_spec(make_root(), item))
        assert get_field(result, "LineItem", "name").examples == ["USB Cable"]

    def test_unrelated_prefix_kept(self):
        item = mdl("Item", fields=[ident("name", ("REF 1", "REF 2"))])
        result, report = lint_spec(build_spec(make_root(), item))
        assert get_field(result, "Item", "name").examples == ["REF 1", "REF 2"]
        assert not report.by_rule("R6")

    def test_invention_sentence_deleted_from_description(self):
        description = (
            "Assign a sequential id when the document lacks one. Printed in the first column."
        )
        item = mdl(
            "Item",
            fields=[ident("name", ("USB Cable", "Dock DS"), description=description)],
        )
        result, report = lint_spec(build_spec(make_root(), item))
        assert get_field(result, "Item", "name").description == "Printed in the first column."
        assert repairs(report, "R6")


# ---------------------------------------------------------------------------
# R9 — edge labels
# ---------------------------------------------------------------------------


class TestR9EdgeLabels:
    def test_camel_case_label_normalized(self):
        root = make_root(edge_fld("seller", "Party", "issuedBy"))
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "seller").edge_label == "ISSUED_BY"
        assert repairs(report, "R9")

    def test_banned_label_rewritten_with_gap(self):
        root = make_root(edge_fld("seller", "Party", "HAS"))
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "seller").edge_label == "HAS_PARTY"
        assert any(g.kind == "missing_edge_label" for g in report.gaps)

    def test_same_field_name_same_target_unified_first_wins(self):
        address = mdl("Address", "component", fields=[fld("street")])
        org = mdl(
            "Org",
            fields=[
                ident("name", ("Acme Corp", "Beta SARL")),
                edge_fld("addresses", "Address", "LOCATED_AT", is_list=True),
            ],
        )
        person = mdl(
            "Person",
            fields=[
                ident("full_name", ("John Doe", "Jane Roe")),
                edge_fld("addresses", "Address", "HAS_LOCATION", is_list=True),
            ],
        )
        result, report = lint_spec(build_spec(make_root(), org, person, address))
        assert get_field(result, "Person", "addresses").edge_label == "LOCATED_AT"
        assert repairs(report, "R9")

    def test_multi_role_labels_to_same_target_survive(self):
        # relationships.md "Multiple Edge Types to Same Entity" is deliberate.
        root = make_root(
            edge_fld("seller", "Party", "ISSUED_BY"),
            edge_fld("buyer", "Party", "BILLED_TO"),
        )
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "seller").edge_label == "ISSUED_BY"
        assert get_field(result, "Doc", "buyer").edge_label == "BILLED_TO"
        assert not report.by_rule("R9")

    def test_user_verb_phrase_labels_survive_untouched(self):
        # The from-spec escape hatch: user-chosen verb phrases are never
        # rewritten (OWNS_VEHICLE must not become HAS_OWNS_VEHICLE).
        root = make_root(
            edge_fld("vehicle", "Party", "OWNS_VEHICLE"),
            edge_fld("coverage", "Party", "GRANTS_COVERAGE"),
            edge_fld("contact", "Party", "KNOWS"),
        )
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "vehicle").edge_label == "OWNS_VEHICLE"
        assert get_field(result, "Doc", "coverage").edge_label == "GRANTS_COVERAGE"
        assert get_field(result, "Doc", "contact").edge_label == "KNOWS"
        assert not report.by_rule("R9")

    def test_unknown_verb_multi_token_label_kept_with_advisory(self):
        # Multi-token label whose first token is not a known verb: kept
        # as-is; R9 emits a warn-severity ADVISORY (repaired=False).
        root = make_root(edge_fld("manager", "Party", "MANAGED_BY"))
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "manager").edge_label == "MANAGED_BY"
        entries = report.by_rule("R9")
        assert len(entries) == 1
        assert entries[0].severity == "warn"
        assert entries[0].repaired is False
        assert "advisory" in entries[0].message

    def test_advisory_does_not_fail_strict_repair_mode(self):
        root = make_root(edge_fld("manager", "Party", "MANAGED_BY"))
        result, report = lint_spec(build_spec(root, party_model()), strict=True)
        assert get_field(result, "Doc", "manager").edge_label == "MANAGED_BY"
        assert report.by_rule("R9")

    def test_strict_raises(self):
        root = make_root(edge_fld("seller", "Party", "issuedBy"))
        with pytest.raises(TemplateLintError, match="R9"):
            lint_spec(build_spec(root, party_model()), strict=True)


# ---------------------------------------------------------------------------
# R10 — one canonical home per rich entity
# ---------------------------------------------------------------------------


class TestR10CanonicalHome:
    def _multi_path_spec(self) -> TemplateSpec:
        guarantee = mdl("Guarantee", fields=[ident("name", ("Fire", "Flood")), fld("ceiling")])
        section = mdl(
            "Section",
            fields=[
                ident("title", ("Part A", "Part B")),
                edge_fld("guarantees", "Guarantee", "COVERS", is_list=True),
            ],
        )
        root = make_root(
            edge_fld("sections", "Section", "HAS_SECTION", is_list=True),
            edge_fld("guarantees", "Guarantee", "COVERS", is_list=True),
        )
        return build_spec(root, section, guarantee)

    def test_deeper_full_nesting_flipped_to_reference(self):
        result, report = lint_spec(self._multi_path_spec())
        assert get_field(result, "Section", "guarantees").reference is True
        assert get_field(result, "Doc", "guarantees").reference is False
        assert get_model(result, "Guarantee").canonical_home == "Doc.guarantees"
        assert repairs(report, "R10")

    def test_premarked_canonical_home_wins(self):
        spec = self._multi_path_spec()
        get_model(spec, "Guarantee").canonical_home = "Section.guarantees"
        result, _ = lint_spec(spec)
        assert get_field(result, "Doc", "guarantees").reference is True
        assert get_field(result, "Section", "guarantees").reference is False

    def test_multi_role_single_edges_stay_full(self):
        # The billing seller/buyer Party shape: every inbound non-reference
        # edge is a single (non-list) edge from the SAME parent — both stay
        # full, whatever the target's field count.
        root = make_root(
            edge_fld("seller", "Party", "ISSUED_BY"),
            edge_fld("buyer", "Party", "BILLED_TO"),
        )
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "seller").reference is False
        assert get_field(result, "Doc", "buyer").reference is False
        assert not report.by_rule("R10")

    def test_rich_target_same_parent_single_edges_stay_full(self):
        # Regression: the exception keys on SHAPE, not field count — the
        # canon seller/buyer Party carries 8 non-identity fields and must
        # never lose the buyer's data to a reference flip.
        rich_party = party_model(
            fld("email"),
            fld("phone"),
            fld("street"),
            fld("city"),
            fld("country"),
            fld("iban"),
        )
        root = make_root(
            edge_fld("seller", "Party", "ISSUED_BY"),
            edge_fld("buyer", "Party", "BILLED_TO"),
        )
        result, report = lint_spec(build_spec(root, rich_party))
        assert get_field(result, "Doc", "seller").reference is False
        assert get_field(result, "Doc", "buyer").reference is False
        assert not report.by_rule("R10")

    def test_same_parent_list_edge_defeats_the_exception(self):
        # One inbound edge is a list -> not the multi-role shape -> the
        # non-canonical path flips to reference.
        root = make_root(
            edge_fld("seller", "Party", "ISSUED_BY"),
            edge_fld("agents", "Party", "REPRESENTS", is_list=True),
        )
        result, report = lint_spec(build_spec(root, party_model()))
        assert get_field(result, "Doc", "seller").reference is False
        assert get_field(result, "Doc", "agents").reference is True
        assert get_model(result, "Party").canonical_home == "Doc.seller"
        assert repairs(report, "R10")

    def test_different_parents_defeat_the_exception(self):
        section = mdl(
            "Section",
            fields=[
                ident("title", ("Part A", "Part B")),
                edge_fld("contact", "Party", "REPRESENTS"),
            ],
        )
        root = make_root(
            edge_fld("seller", "Party", "ISSUED_BY"),
            edge_fld("sections", "Section", "HAS_SECTION", is_list=True),
        )
        result, report = lint_spec(build_spec(root, section, party_model()))
        assert get_field(result, "Doc", "seller").reference is False
        assert get_field(result, "Section", "contact").reference is True
        assert repairs(report, "R10")

    def test_strict_raises(self):
        with pytest.raises(TemplateLintError, match="R10"):
            lint_spec(self._multi_path_spec(), strict=True)


# ---------------------------------------------------------------------------
# R11 — reference targets need identity and a canonical home elsewhere
# ---------------------------------------------------------------------------


class TestR11ReferenceTargets:
    def test_reference_to_component_unreferenced(self):
        details = mdl("Details", "component", fields=[fld("text")])
        root = make_root(edge_fld("details", "Details", "HAS_DETAILS", reference=True))
        result, report = lint_spec(build_spec(root, details))
        assert get_field(result, "Doc", "details").reference is False
        assert repairs(report, "R11")

    def test_only_path_reference_to_rich_entity_flipped_off(self):
        guarantee = mdl("Guarantee", fields=[ident("name", ("Fire", "Flood")), fld("ceiling")])
        root = make_root(
            edge_fld("guarantees", "Guarantee", "COVERS", is_list=True, reference=True)
        )
        result, report = lint_spec(build_spec(root, guarantee))
        assert get_field(result, "Doc", "guarantees").reference is False
        assert repairs(report, "R11")

    def test_identity_only_shared_node_reference_kept(self):
        # best-practices.md Person pattern: identity-only nodes may live on refs.
        person = mdl("Person", fields=[ident("full_name", ("John Doe", "Jane Roe"))])
        root = make_root(edge_fld("person", "Person", "IS_PERSON", reference=True))
        result, report = lint_spec(build_spec(root, person))
        assert get_field(result, "Doc", "person").reference is True
        assert not report.by_rule("R11")

    def test_one_of_two_references_becomes_the_home(self):
        guarantee = mdl("Guarantee", fields=[ident("name", ("Fire", "Flood")), fld("ceiling")])
        root = make_root(
            edge_fld("g_first", "Guarantee", "COVERS", is_list=True, reference=True),
            edge_fld("g_second", "Guarantee", "EXCLUDES", is_list=True, reference=True),
        )
        result, report = lint_spec(build_spec(root, guarantee))
        assert get_field(result, "Doc", "g_first").reference is False
        assert get_field(result, "Doc", "g_second").reference is True
        assert len(repairs(report, "R11")) == 1

    def test_strict_raises(self):
        details = mdl("Details", "component", fields=[fld("text")])
        root = make_root(edge_fld("details", "Details", "HAS_DETAILS", reference=True))
        with pytest.raises(TemplateLintError, match="R11"):
            lint_spec(build_spec(root, details), strict=True)


# ---------------------------------------------------------------------------
# R12 — closed_catalog requires a canonical catalog home elsewhere
# ---------------------------------------------------------------------------


class TestR12ClosedCatalog:
    def test_closed_catalog_without_home_cleared_reference_kept(self):
        person = mdl("Person", fields=[ident("full_name", ("John Doe", "Jane Roe"))])
        root = make_root(
            edge_fld(
                "people", "Person", "MENTIONS", is_list=True, reference=True, closed_catalog=True
            )
        )
        result, report = lint_spec(build_spec(root, person))
        field = get_field(result, "Doc", "people")
        assert field.closed_catalog is False
        assert field.reference is True
        assert repairs(report, "R12")

    def test_closed_catalog_with_canonical_home_kept(self):
        guarantee = mdl("Guarantee", fields=[ident("name", ("Fire", "Flood")), fld("ceiling")])
        root = make_root(
            edge_fld("guarantees", "Guarantee", "COVERS", is_list=True),
            edge_fld(
                "excluded",
                "Guarantee",
                "EXCLUDES",
                is_list=True,
                reference=True,
                closed_catalog=True,
            ),
        )
        result, report = lint_spec(build_spec(root, guarantee))
        assert get_field(result, "Doc", "excluded").closed_catalog is True
        assert not report.by_rule("R12")


# ---------------------------------------------------------------------------
# R13 — max_instances requires a cardinality sentence (and the 2x contract)
# ---------------------------------------------------------------------------


class TestR13Cardinality:
    def test_sentence_injected_when_docstring_lacks_cardinality(self):
        segment = mdl(
            "Segment",
            fields=[ident("name", ("Retail", "Cloud"))],
            max_instances=12,
            docstring="A reportable segment from the segment note.",
        )
        result, report = lint_spec(build_spec(make_root(), segment))
        assert get_model(result, "Segment").docstring.endswith("At most 6 expected per document.")
        assert repairs(report, "R13")

    def test_existing_cardinality_sentence_untouched(self):
        docstring = "One of the 3-6 reportable segments named in the segment note."
        segment = mdl(
            "Segment",
            fields=[ident("name", ("Retail", "Cloud"))],
            max_instances=12,
            docstring=docstring,
        )
        result, report = lint_spec(build_spec(make_root(), segment))
        assert get_model(result, "Segment").docstring == docstring
        assert not report.by_rule("R13")

    def test_repair_draft_doubles_documented_max_once(self):
        segment = dmdl(
            "Segment",
            "entity",
            [dfld("name", role="identity", examples=["Retail", "Cloud"])],
            identity_fields=["name"],
            max_instances=6,
            docstring="One of the 3-6 reportable segments named in the segment note.",
        )
        spec, report = repair_draft(ddraft(droot(), segment))
        assert get_model(spec, "Segment").max_instances == 12
        info = report.by_rule("R13")
        assert info and "2x the documented maximum" in info[0].message
        assert not info[0].repaired

    def test_doubling_is_not_a_strict_violation(self):
        segment = dmdl(
            "Segment",
            "entity",
            [dfld("name", role="identity", examples=["Retail", "Cloud"])],
            identity_fields=["name"],
            max_instances=6,
            docstring="One of the 3-6 reportable segments named in the segment note.",
        )
        spec, _ = repair_draft(ddraft(droot(), segment), strict=True)  # must not raise
        assert get_model(spec, "Segment").max_instances == 12

    def test_strict_raises_on_injection(self):
        segment = mdl(
            "Segment",
            fields=[ident("name", ("Retail", "Cloud"))],
            max_instances=12,
            docstring="A reportable segment from the segment note.",
        )
        with pytest.raises(TemplateLintError, match="R13"):
            lint_spec(build_spec(make_root(), segment), strict=True)


# ---------------------------------------------------------------------------
# R14 — nesting depth 2-4
# ---------------------------------------------------------------------------


def _chain_spec() -> TemplateSpec:
    # Doc -> Level5 -> Level4 -> Level3 -> Level2 -> Level1 (depth 5).
    models = []
    previous = None
    for index in range(1, 6):
        name = f"Level{index}"
        fields = [ident("name", (f"L{index}A", f"L{index}B")), fld("data")]
        if previous is not None:
            fields.append(edge_fld("child", previous, "CONTAINS_CHILD"))
        models.append(mdl(name, fields=fields))
        previous = name
    root = make_root(edge_fld("child", "Level5", "CONTAINS_CHILD"))
    return build_spec(root, *models)


class TestR14NestingDepth:
    def test_deep_single_chain_flagged_not_repaired(self):
        # Level1 sits at depth 5; its only full path cannot be flipped without
        # orphaning its content, so R14 flags instead of repairing.
        result, report = lint_spec(_chain_spec())
        entries = report.by_rule("R14")
        assert entries and not entries[0].repaired
        assert get_field(result, "Level2", "child").reference is False

    def test_deep_multi_path_entity_resolved_by_canonical_home(self):
        # With a shallow second home, R10 (canonical home) repairs the depth
        # violation before R14 sees it.
        spec = _chain_spec()
        root = get_model(spec, "Doc")
        root.fields.append(edge_fld("deep_direct", "Level1", "CONTAINS_CHILD"))
        spec = TemplateSpec.model_validate(spec.model_dump())
        result, report = lint_spec(spec)
        assert get_field(result, "Level2", "child").reference is True
        assert repairs(report, "R10")
        assert not [e for e in report.by_rule("R14") if e.model == "Level2"]


# ---------------------------------------------------------------------------
# R15 — self-references and mutual cycles
# ---------------------------------------------------------------------------


class TestR15Cycles:
    def test_self_reference_reported_for_renderer(self):
        employee = mdl(
            "Employee",
            fields=[
                ident("full_name", ("Ann Lee", "Bo Kim")),
                edge_fld("manager", "Employee", "REFERENCES_MANAGER"),
            ],
        )
        root = make_root(edge_fld("employees", "Employee", "EMPLOYS", is_list=True))
        result, report = lint_spec(build_spec(root, employee))
        entries = report.by_rule("R15")
        assert entries and entries[0].severity == "info" and not entries[0].repaired
        # stays a full self-edge; the renderer emits the forward reference
        assert get_field(result, "Employee", "manager").reference is False

    def test_mutual_cycle_back_edge_flipped_to_reference(self):
        party = mdl(
            "Party",
            fields=[
                ident("name", ("Acme Corp", "Beta SARL")),
                edge_fld("issued_doc", "Doc", "ISSUED_DOC"),
            ],
        )
        root = make_root(edge_fld("seller", "Party", "ISSUED_BY"))
        result, report = lint_spec(build_spec(root, party))
        assert get_field(result, "Party", "issued_doc").reference is True
        assert get_field(result, "Doc", "seller").reference is False
        assert repairs(report, "R15")

    def test_strict_raises_on_cycle_repair(self):
        party = mdl(
            "Party",
            fields=[
                ident("name", ("Acme Corp", "Beta SARL")),
                edge_fld("issued_doc", "Doc", "ISSUED_DOC"),
            ],
        )
        root = make_root(edge_fld("seller", "Party", "ISSUED_BY"))
        with pytest.raises(TemplateLintError, match="R15"):
            lint_spec(build_spec(root, party), strict=True)


# ---------------------------------------------------------------------------
# R16 — description scrub
# ---------------------------------------------------------------------------


class TestR16DescriptionScrub:
    def test_computation_sentence_removed(self):
        description = "Compute the total by multiplying quantity by price. LOOK FOR the totals row."
        root = make_root(fld("total", description=description))
        result, report = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "total").description == "LOOK FOR the totals row."
        assert repairs(report, "R16")

    def test_global_rule_restatement_removed(self):
        description = "Use N/A when the value is missing. Printed near the header."
        root = make_root(fld("code", description=description))
        result, _ = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "code").description == "Printed near the header."

    def test_emptied_description_raises_gap(self):
        root = make_root(fld("amount", description="Convert the amount to EUR."))
        result, report = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "amount").description == ""
        assert any(g.kind == "missing_description" and g.field == "amount" for g in report.gaps)

    def test_clean_description_untouched(self):
        description = "LOOK FOR the currency code in the totals section."
        root = make_root(fld("currency", description=description))
        result, report = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "currency").description == description
        assert not report.by_rule("R16")

    def test_strict_raises(self):
        root = make_root(fld("amount", description="Convert the amount to EUR."))
        with pytest.raises(TemplateLintError, match="R16"):
            lint_spec(build_spec(root), strict=True)


# ---------------------------------------------------------------------------
# R19 — identity-less root lists get the dedup validator
# ---------------------------------------------------------------------------


class TestR19RootListDedup:
    def test_scalar_and_component_lists_scheduled(self):
        amount = mdl("Amount", "component", fields=[fld("value", type="float")])
        root = make_root(
            fld("keywords", is_list=True),
            fld("amounts", type="Amount", is_list=True),
            edge_fld("parties", "Party", "MENTIONS", is_list=True),
        )
        result, report = lint_spec(build_spec(root, amount, party_model()))
        assert set(result.needs_root_list_dedup) == {"keywords", "amounts"}
        assert len(repairs(report, "R19")) == 2

    def test_entity_lists_excluded(self):
        root = make_root(edge_fld("parties", "Party", "MENTIONS", is_list=True))
        result, report = lint_spec(build_spec(root, party_model()))
        assert result.needs_root_list_dedup == []
        assert not report.by_rule("R19")

    def test_already_scheduled_field_not_duplicated(self):
        root = make_root(fld("keywords", is_list=True))
        spec = build_spec(root, needs_root_list_dedup=["keywords"])
        result, report = lint_spec(spec)
        assert result.needs_root_list_dedup == ["keywords"]
        assert not report.by_rule("R19")


# ---------------------------------------------------------------------------
# R20/R21 — naming (keywords/builtins/reserved node attrs) + rename cascades
# ---------------------------------------------------------------------------


class TestR20R21Naming:
    def test_keyword_field_renamed_r20(self):
        root = make_root(fld("class"))
        result, report = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "class_field") is not None
        assert repairs(report, "R20")

    def test_reserved_node_attr_field_renamed_r21(self):
        root = make_root(fld("label"))
        result, report = lint_spec(build_spec(root))
        assert get_field(result, "Doc", "name_label") is not None
        assert repairs(report, "R21")

    def test_rename_cascades_to_identity_dedup_and_canonical_home(self):
        party = mdl(
            "Party",
            fields=[ident("name", ("Acme Corp", "Beta SARL"))],
            canonical_home="Doc.label",
        )
        root = mdl(
            "Doc",
            "root",
            fields=[
                ident("id", ("INV-001", "INV-002")),
                fld("type", is_list=True),
                edge_fld("label", "Party", "ISSUED_BY"),
            ],
            identity=["id"],
        )
        spec = build_spec(root, party, needs_root_list_dedup=["type"])
        result, report = lint_spec(spec)
        doc = get_model(result, "Doc")
        assert doc.identity_fields == ["identifier"]
        assert result.needs_root_list_dedup == ["category"]
        assert get_model(result, "Party").canonical_home == "Doc.name_label"
        assert len(repairs(report, "R21")) == 3

    def test_model_shadowing_template_import_renamed_with_type_cascade(self):
        item = mdl("Field", fields=[ident("name", ("USB Cable", "Dock DS-300"))])
        root = make_root(edge_fld("items", "Field", "CONTAINS_LINE", is_list=True))
        result, report = lint_spec(build_spec(root, item))
        assert {m.name for m in result.models} == {"Doc", "FieldModel"}
        assert get_field(result, "Doc", "items").type == "FieldModel"
        assert repairs(report, "R20")

    def test_strict_raises(self):
        root = make_root(fld("label"))
        with pytest.raises(TemplateLintError, match="R21"):
            lint_spec(build_spec(root), strict=True)


# ---------------------------------------------------------------------------
# R18 — role data on shared entities (advisory)
# ---------------------------------------------------------------------------


class TestR18RoleData:
    def test_role_field_on_shared_entity_flagged(self):
        person = mdl(
            "Person",
            fields=[ident("full_name", ("John Doe", "Jane Roe")), fld("title")],
        )
        root = make_root(
            edge_fld("created_by", "Person", "CREATED_BY"),
            edge_fld("approved_by", "Person", "APPROVED_BY"),
        )
        _, report = lint_spec(build_spec(root, person))
        entries = report.by_rule("R18")
        assert entries and not entries[0].repaired
        assert entries[0].field == "title"

    def test_single_path_entity_not_flagged(self):
        person = mdl(
            "Person",
            fields=[ident("full_name", ("John Doe", "Jane Roe")), fld("title")],
        )
        root = make_root(edge_fld("created_by", "Person", "CREATED_BY"))
        _, report = lint_spec(build_spec(root, person))
        assert not report.by_rule("R18")


# ---------------------------------------------------------------------------
# R22 — evidence / "Rare:" flag on induced fields (report-only)
# ---------------------------------------------------------------------------


class TestR22Evidence:
    def test_induced_field_without_evidence_reported(self):
        root = make_root(fld("notes"), provenance="induced")
        _, report = lint_spec(build_spec(root))
        entries = report.by_rule("R22")
        assert entries and entries[0].severity == "info" and entries[0].field == "notes"

    def test_evidence_rare_flag_or_examples_suppress(self):
        root = make_root(
            fld("total", evidence=["| Total | 129.00 |"]),
            fld("discount", description="Rare: only on discounted invoices."),
            fld("currency", examples=["EUR", "USD"]),
            provenance="induced",
        )
        _, report = lint_spec(build_spec(root))
        assert not report.by_rule("R22")

    def test_user_provenance_silent(self):
        root = make_root(fld("notes"))  # provenance="user" via factory
        _, report = lint_spec(build_spec(root))
        assert not report.by_rule("R22")


# ---------------------------------------------------------------------------
# Scalar-type-named models: rename cascades must not corrupt scalar fields
# ---------------------------------------------------------------------------


class TestScalarNamedModels:
    def _draft(self) -> dict:
        date_class = dmdl(
            "date",
            "entity",
            [dfld("name", role="identity", examples=["Expiry", "Renewal"])],
            identity_fields=["name"],
        )
        contract = dmdl(
            "Contract",
            "entity",
            [
                dfld("name", role="identity", examples=["C-100", "C-200"]),
                dfld("signed_on", type="date"),  # the scalar, NOT the model
            ],
            identity_fields=["name"],
        )
        root = droot(
            dfld("milestone", type="date", role="edge", edge_label="HAS_MILESTONE"),
            dfld(
                "contracts",
                type="Contract",
                role="edge",
                edge_label="HAS_CONTRACTS",
                is_list=True,
            ),
        )
        return ddraft(root, date_class, contract)

    def test_repair_draft_suffixes_model_and_preserves_scalar_fields(self):
        spec, report = repair_draft(self._draft())
        names = {m.name for m in spec.models}
        assert "DateModel" in names and "date" not in names
        # the edge to the model follows the rename ...
        assert get_field(spec, "Doc", "milestone").type == "DateModel"
        # ... while the unrelated scalar date field stays a scalar
        assert get_field(spec, "Contract", "signed_on").type == "date"
        assert any("DateModel" in e.message for e in repairs(report, "R20"))

    def test_lint_spec_cascade_guards_scalar_fields(self):
        date_model = mdl("date", fields=[ident("name", ("Expiry", "Renewal"))])
        contract = mdl(
            "Contract",
            fields=[ident("name", ("C-100", "C-200")), fld("signed_on", type="date")],
        )
        root = make_root(
            edge_fld("milestone", "date", "HAS_MILESTONE"),
            edge_fld("contracts", "Contract", "HAS_CONTRACTS", is_list=True),
        )
        result, report = lint_spec(build_spec(root, date_model, contract))
        assert {m.name for m in result.models} == {"Doc", "DateModel", "Contract"}
        assert get_field(result, "Doc", "milestone").type == "DateModel"
        assert get_field(result, "Contract", "signed_on").type == "date"
        assert repairs(report, "R20")


# ---------------------------------------------------------------------------
# Sanitize-stable collision suffixes + fixpoint exhaustion surfacing
# ---------------------------------------------------------------------------


class TestSanitizeStableCollisions:
    def _entity(self, name: str, examples: tuple) -> dict:
        return dmdl(
            name,
            "entity",
            [dfld("name", role="identity", examples=list(examples))],
            identity_fields=["name"],
        )

    def test_collision_chain_converges_with_stable_names(self):
        # 'Field' sanitizes to 'FieldModel', colliding with the real
        # 'FieldModel'; with the old '_2' suffix each lint pass re-sanitized
        # 'FieldModel_2' -> 'FieldModel2' -> collision -> '..._2' and the
        # fixpoint loop exhausted with a raw AssertionError.
        draft = ddraft(
            droot(),
            self._entity("Field", ("Alpha One", "Beta Two")),
            self._entity("FieldModel", ("Gamma Three", "Delta Four")),
            self._entity("FieldModel2", ("Epsilon Five", "Zeta Six")),
            self._entity("FieldModel22", ("Eta Seven", "Theta Eight")),
        )
        spec, report = repair_draft(draft)
        names = {m.name for m in spec.models}
        assert "Doc" in names and len(names) == 5
        assert report.has_repairs
        # every assigned name is a fixed point: a re-lint performs no renames
        relinted, second = lint_spec(spec)
        assert not second.has_repairs
        assert {m.name for m in relinted.models} == names

    def test_fixpoint_exhaustion_raises_template_lint_error(self, monkeypatch):
        from docling_graph.templategen import linter as linter_module

        def never_satisfied(spec: TemplateSpec, ctx: Any) -> None:
            ctx.add("R99", "warn", "Doc", None, "always repairs", repaired=True)

        monkeypatch.setattr(linter_module, "_RULES", (never_satisfied,))
        with pytest.raises(TemplateLintError, match="fixpoint") as exc_info:
            lint_spec(build_spec(make_root()))
        # never a raw AssertionError; the collected report stays attached
        assert exc_info.value.report.entries
        assert all(e.rule_id == "R99" for e in exc_info.value.report.entries)


# ---------------------------------------------------------------------------
# Template-module field shadowing (date/datetime/edge/logger)
# ---------------------------------------------------------------------------


class TestTemplateModuleFieldShadowing:
    def test_shadowing_fields_renamed_with_suffix(self):
        root = make_root(
            fld("date", type="date"),
            fld("datetime", type="datetime"),
            fld("edge"),
            fld("logger"),
            fld("issue_date", type="date"),  # unaffected: only exact names collide
        )
        result, report = lint_spec(build_spec(root))
        names = {f.name for f in get_model(result, "Doc").fields}
        assert {"date_field", "datetime_field", "edge_field", "logger_field"} <= names
        assert names.isdisjoint({"date", "datetime", "edge", "logger"})
        assert "issue_date" in names
        assert len(repairs(report, "R20")) == 4


# ---------------------------------------------------------------------------
# Model-typed property fields traverse like edges (R14/R15)
# ---------------------------------------------------------------------------


class TestPropertyFieldNesting:
    def test_property_cycle_back_field_promoted_to_reference_edge(self):
        alpha = mdl(
            "Alpha",
            fields=[ident("name", ("A-100", "A-200")), fld("partner", type="Beta")],
        )
        beta = mdl(
            "Beta",
            fields=[ident("name", ("B-100", "B-200")), fld("partner", type="Alpha")],
        )
        root = make_root(edge_fld("alpha", "Alpha", "HAS_ALPHA"))
        result, report = lint_spec(build_spec(root, alpha, beta))
        back = get_field(result, "Beta", "partner")
        assert back.role == "edge"
        assert back.reference is True
        assert back.edge_label == "HAS_PARTNER"
        forward = get_field(result, "Alpha", "partner")
        assert forward.role == "property" and forward.reference is False
        assert repairs(report, "R15")

    def test_property_self_reference_reported_for_renderer(self):
        node = mdl(
            "Node",
            fields=[ident("name", ("N-100", "N-200")), fld("parent", type="Node")],
        )
        root = make_root(edge_fld("nodes", "Node", "CONTAINS_CHILD", is_list=True))
        _, report = lint_spec(build_spec(root, node))
        entries = report.by_rule("R15")
        assert entries and entries[0].severity == "info"
        assert "forward reference" in entries[0].message

    def test_property_nesting_counts_toward_depth_budget(self):
        # Doc -> Level5 -> ... -> Level1 nested purely via model-typed
        # property fields: R14 must see the same depth-5 chain it would see
        # through edge fields (build_node_catalog walks both identically).
        models = []
        previous = None
        for index in range(1, 6):
            name = f"Level{index}"
            fields = [ident("name", (f"L{index}A", f"L{index}B")), fld("data")]
            if previous is not None:
                fields.append(fld("child", type=previous))
            models.append(mdl(name, fields=fields))
            previous = name
        root = make_root(fld("child", type="Level5"))
        _, report = lint_spec(build_spec(root, *models))
        assert report.by_rule("R14")


# ---------------------------------------------------------------------------
# Fixpoint termination + idempotence on a pathological draft
# ---------------------------------------------------------------------------


def _pathological_draft() -> dict:
    return {
        "root": "Doc",
        "models": [
            {
                "name": "Doc",
                "kind": "root",
                "docstring": "",
                "fields": [
                    {"name": "notes", "type": "str"},
                    {"name": "sections", "type": "class", "role": "edge", "is_list": True},
                    {
                        "name": "details",
                        "type": "Details",
                        "role": "edge",
                        "edge_label": "HAS",
                        "reference": True,
                    },
                ],
            },
            {
                "name": "class",
                "kind": "entity",
                "docstring": "A section of the document.",
                "identity_fields": ["a", "b", "c"],
                "fields": [
                    {"name": "a", "role": "identity", "type": "str", "examples": ["S-1", "S-2"]},
                    {"name": "b", "role": "identity", "type": "Kind", "examples": ["xy"]},
                    {
                        "name": "c",
                        "role": "identity",
                        "type": "str",
                        "examples": ["A very long prose example"],
                    },
                ],
            },
            {
                "name": "Details",
                "kind": "component",
                "docstring": "Details block.",
                "identity_fields": ["text"],
                "max_instances": 3,
                "fields": [{"name": "text", "type": "str", "role": "identity"}],
            },
        ],
        "enums": [{"name": "Kind", "members": ["X", "Y"]}],
    }


class TestFixpointAndIdempotence:
    def test_pathological_draft_terminates(self):
        spec, report = repair_draft(_pathological_draft())
        assert report.has_repairs
        # every headline repair landed
        assert spec.root == "Doc"
        assert get_model(spec, "Doc").identity_fields == ["document_reference"]
        assert get_model(spec, "Class").identity_fields == ["a", "b"]
        assert get_model(spec, "Details").identity_fields == []
        assert get_model(spec, "Details").max_instances is None
        assert get_field(spec, "Doc", "sections").edge_label == "HAS_SECTIONS"
        assert get_field(spec, "Doc", "details").edge_label == "HAS_DETAILS"
        assert get_field(spec, "Doc", "details").reference is False

    def test_repaired_spec_is_a_fixpoint(self):
        spec, _ = repair_draft(_pathological_draft())
        relinted, second = lint_spec(spec)
        assert not second.has_repairs
        assert relinted.model_dump() == spec.model_dump()

    def test_input_draft_never_mutated(self):
        draft = _pathological_draft()
        import copy

        snapshot = copy.deepcopy(draft)
        repair_draft(draft)
        assert draft == snapshot

    def test_max_passes_is_bounded(self):
        assert MAX_LINT_PASSES == 3


# ---------------------------------------------------------------------------
# repair=False (report-only) and strict-mode plumbing
# ---------------------------------------------------------------------------


class TestReportOnlyMode:
    def _violating_spec(self) -> TemplateSpec:
        segment = mdl(
            "Segment",
            fields=[ident("name", ("Retail", "Cloud"))],
            max_instances=12,
            docstring="A reportable segment from the segment note.",
        )
        return build_spec(make_root(), segment)

    def test_repair_false_reports_without_mutating(self):
        spec = self._violating_spec()
        before = spec.model_dump()
        result, report = lint_spec(spec, repair=False)
        assert result is spec
        assert spec.model_dump() == before
        assert report.by_rule("R13")
        assert not report.has_repairs

    def test_repair_true_returns_new_spec_and_never_mutates_input(self):
        spec = self._violating_spec()
        before = spec.model_dump()
        result, report = lint_spec(spec)
        assert spec.model_dump() == before
        assert result.model_dump() != before
        assert report.has_repairs


class TestStrictMode:
    def test_strict_raises_with_report_and_spec_attached(self):
        segment = mdl(
            "Segment",
            fields=[ident("name", ("Retail", "Cloud"))],
            max_instances=12,
            docstring="A reportable segment from the segment note.",
        )
        spec = build_spec(make_root(), segment)
        with pytest.raises(TemplateLintError) as exc_info:
            lint_spec(spec, strict=True)
        error = exc_info.value
        assert error.report.has_repairs
        assert error.spec is not None
        assert "R13" in str(error)

    def test_clean_spec_passes_strict(self):
        spec = build_spec(make_root())
        result, report = lint_spec(spec, strict=True)
        assert not report.has_repairs
        assert result.model_dump() == spec.model_dump()

    def test_advisory_findings_do_not_fail_strict(self):
        # R3 (missing examples) has no deterministic repair — advisory only.
        spec = build_spec(make_root(), mdl("Party", fields=[ident("name", ("Acme Corp",))]))
        _, report = lint_spec(spec, strict=True)
        assert report.by_rule("R3")


# ---------------------------------------------------------------------------
# Enum pre-validation repairs (R17's draft half)
# ---------------------------------------------------------------------------


class TestEnumDraftRepairs:
    def test_empty_enum_dropped_and_fields_retyped(self):
        model = dmdl(
            "Ticket",
            "entity",
            [
                dfld("name", role="identity", examples=["T-1", "T-2"]),
                dfld("status", type="Status"),
            ],
            identity_fields=["name"],
        )
        draft = ddraft(droot(), model, enums=[{"name": "Status", "members": []}])
        spec, report = repair_draft(draft)
        assert spec.enums == []
        assert get_field(spec, "Ticket", "status").type == "str"
        assert repairs(report, "R17")

    def test_unknown_synonyms_pruned(self):
        enum = {"name": "Status", "members": ["Open"], "synonyms": {"Gone": ["closed"]}}
        draft = ddraft(droot(), enums=[enum])
        spec, report = repair_draft(draft)
        assert spec.enums[0].synonyms == {}
        assert repairs(report, "R17")
