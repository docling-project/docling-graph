"""Unit tests for the JSON Schema -> SPEC-draft compiler (design §5.3), the
format sniffer/dispatcher, and the zero-LLM-by-construction guard."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

import docling_graph.templategen.ontology as ontology_pkg
from docling_graph.templategen.linter import repair_draft
from docling_graph.templategen.ontology import (
    jsonschema as jsonschema_mod,
    linkml as linkml_mod,
    owl as owl_mod,
    sniff_ontology_format,
    spec_draft_from_ontology,
)
from docling_graph.templategen.ontology.jsonschema import spec_draft_from_jsonschema
from docling_graph.templategen.spec import TemplateSpec

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "templategen" / "ontologies"
INVOICE = FIXTURES / "invoice.schema.json"


def compile_fixture(name: str, **kwargs: Any):
    return spec_draft_from_jsonschema(FIXTURES / name, **kwargs)


def models_by_name(draft: dict) -> dict[str, dict]:
    return {m["name"]: m for m in draft["models"]}


def field(model: dict, name: str) -> dict:
    return next(f for f in model["fields"] if f["name"] == name)


@pytest.fixture(scope="module")
def invoice() -> dict:
    draft, _ = spec_draft_from_jsonschema(INVOICE)
    return draft


@pytest.fixture(scope="module")
def invoice_gaps() -> list:
    _, gaps = spec_draft_from_jsonschema(INVOICE)
    return gaps


# ---------------------------------------------------------------------------
# Root, models, fields
# ---------------------------------------------------------------------------


class TestRootAndModels:
    def test_title_becomes_root_name(self, invoice):
        assert invoice["root"] == "Invoice"
        assert models_by_name(invoice)["Invoice"]["kind"] == "root"

    def test_root_name_override(self):
        draft, _ = compile_fixture("invoice.schema.json", root="PurchaseDoc")
        assert draft["root"] == "PurchaseDoc"

    def test_referenced_defs_become_models(self, invoice):
        assert {"LineItem", "Party", "Address"} <= set(models_by_name(invoice))

    def test_description_becomes_docstring(self, invoice):
        assert models_by_name(invoice)["LineItem"]["docstring"].startswith(
            "One row of the billing table."
        )

    def test_missing_description_gets_placeholder_and_gap(self, invoice, invoice_gaps):
        assert models_by_name(invoice)["Address"]["docstring"]
        assert any(g.model == "Address" and g.kind == "missing_docstring" for g in invoice_gaps)

    @pytest.mark.parametrize(
        ("field_name", "expected_type"),
        [
            ("invoice_number", "str"),
            ("issue_date", "date"),  # format: date
            ("created_at", "datetime"),  # format: date-time
        ],
    )
    def test_scalar_and_format_mapping(self, invoice, field_name, expected_type):
        assert field(models_by_name(invoice)["Invoice"], field_name)["type"] == expected_type

    def test_array_of_scalars(self, invoice):
        tags = field(models_by_name(invoice)["Invoice"], "tags")
        assert tags["type"] == "str"
        assert tags["is_list"] is True

    def test_property_examples_harvested(self, invoice):
        assert field(models_by_name(invoice)["Invoice"], "invoice_number")["examples"] == [
            "INV-2024-0113",
            "INV-2024-0207",
        ]


# ---------------------------------------------------------------------------
# Inline objects -> components
# ---------------------------------------------------------------------------


class TestInlineObjects:
    def test_inline_object_becomes_component(self, invoice):
        total = models_by_name(invoice)["Total"]
        assert total["kind"] == "component"
        assert {f["name"] for f in total["fields"]} == {"amount", "currency"}
        assert field(total, "amount")["type"] == "float"

    def test_inline_component_nested_as_property_field(self, invoice):
        total_field = field(models_by_name(invoice)["Invoice"], "total")
        assert total_field["role"] == "property"
        assert total_field["type"] == "Total"

    def test_component_target_field_stays_property(self, invoice):
        # Address is a demoted component: no edge, no label.
        address = field(models_by_name(invoice)["Party"], "address")
        assert address["role"] == "property"
        assert address.get("edge_label") is None


# ---------------------------------------------------------------------------
# Identity: required candidates only
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_required_ladder_named_string_wins(self, invoice):
        assert models_by_name(invoice)["Invoice"]["identity_fields"] == ["invoice_number"]
        assert models_by_name(invoice)["LineItem"]["identity_fields"] == ["line_number"]
        assert models_by_name(invoice)["Party"]["identity_fields"] == ["name"]

    def test_non_ladder_required_fields_stay_optional_with_note(self, invoice):
        notes = invoice["generator"]["notes"]
        assert any("issue_date" in n and "Optionality Law" in n for n in notes)
        assert any("description" in n and "Optionality Law" in n for n in notes)

    def test_ladder_named_but_not_required_is_not_identity(self, invoice):
        # vat_id matches the ladder but is not in Party's required array.
        assert "vat_id" not in models_by_name(invoice)["Party"]["identity_fields"]

    def test_def_without_candidate_demoted_to_component(self, invoice, invoice_gaps):
        assert models_by_name(invoice)["Address"]["kind"] == "component"
        assert any(g.model == "Address" and g.kind == "missing_identity" for g in invoice_gaps)

    def test_identity_without_examples_raises_gap(self, invoice_gaps):
        assert any(
            g.model == "Party" and g.field == "name" and g.kind == "missing_examples"
            for g in invoice_gaps
        )

    def test_identity_with_examples_has_no_gap(self, invoice_gaps):
        assert not any(
            g.field == "invoice_number" and g.kind == "missing_examples" for g in invoice_gaps
        )

    def test_identity_less_root_synthesizes_document_reference(self, tmp_path):
        source = tmp_path / "noid.schema.json"
        source.write_text(
            '{"title": "Note", "type": "object", "properties": {"body": {"type": "string"}}}'
        )
        draft, gaps = spec_draft_from_jsonschema(source)
        root = models_by_name(draft)["Note"]
        assert root["identity_fields"] == ["document_reference"]
        assert any(g.kind == "missing_identity" for g in gaps)
        TemplateSpec.model_validate(draft)


# ---------------------------------------------------------------------------
# Edges, enums, cardinality
# ---------------------------------------------------------------------------


class TestEdgesEnumsCardinality:
    def test_ref_to_entity_becomes_edge(self, invoice):
        line_items = field(models_by_name(invoice)["Invoice"], "line_items")
        assert line_items["role"] == "edge"
        assert line_items["type"] == "LineItem"
        # Multi-token label with a non-verb first token is kept as-is (the
        # linter warns advisory); only single unknown tokens get HAS_.
        assert line_items["edge_label"] == "LINE_ITEMS"
        assert line_items["is_list"] is True

    def test_shared_target_keeps_both_full_edges(self, invoice):
        # No multi-path rule in the JSON Schema table: seller AND buyer stay full.
        seller = field(models_by_name(invoice)["Invoice"], "seller")
        buyer = field(models_by_name(invoice)["Invoice"], "buyer")
        assert seller["role"] == buyer["role"] == "edge"
        assert seller.get("reference") is not True
        assert buyer.get("reference") is not True

    def test_enum_becomes_enum_spec(self, invoice):
        assert invoice["enums"] == [
            {
                "name": "Status",
                "members": ["draft", "sent", "paid"],
                "synonyms": {},
                "include_other": True,
            }
        ]
        status = field(models_by_name(invoice)["Invoice"], "status")
        assert status["type"] == "Status"
        assert status["normalizer"] == "enum"

    def test_max_items_keeps_documented_max(self, invoice):
        # Drafts carry the DOCUMENTED maximum; repair_draft doubles exactly once.
        line_item = models_by_name(invoice)["LineItem"]
        assert line_item["max_instances"] == 10
        assert "At most 10 per document." in line_item["docstring"]

    def test_repair_draft_doubles_documented_max_exactly_once(self):
        # End-to-end R13 contract: graph_max_instances == 2 x the documented n
        # (a doubled draft would end up at 4x — a live runtime bound).
        draft, _ = spec_draft_from_ontology(INVOICE)
        spec, _ = repair_draft(draft)
        line_item = next(m for m in spec.models if m.name == "LineItem")
        assert line_item.max_instances == 20  # 2 x maxItems 10

    def test_unbounded_targets_get_no_max_instances(self, invoice):
        assert models_by_name(invoice)["Party"]["max_instances"] is None


# ---------------------------------------------------------------------------
# allOf merge + oneOf/anyOf common fields
# ---------------------------------------------------------------------------


class TestComposition:
    def test_allof_merges_referenced_base(self):
        draft, _ = compile_fixture("claim_allof.schema.json")
        vehicle = models_by_name(draft)["InsuredVehicle"]
        assert {f["name"] for f in vehicle["fields"]} == {
            "make",
            "model_year",
            "registration_code",
        }
        assert vehicle["identity_fields"] == ["registration_code"]

    def test_allof_base_def_not_emitted(self):
        draft, _ = compile_fixture("claim_allof.schema.json")
        assert set(models_by_name(draft)) == {"Claim", "InsuredVehicle"}

    def test_definitions_key_supported(self):
        # claim_allof uses draft-07 "definitions" instead of "$defs".
        draft, _ = compile_fixture("claim_allof.schema.json")
        assert draft["root"] == "Claim"

    def test_oneof_of_objects_keeps_common_fields_with_note(self):
        draft, _ = compile_fixture("payment_oneof.schema.json")
        method = models_by_name(draft)["Method"]
        assert method["kind"] == "component"
        assert {f["name"] for f in method["fields"]} == {"kind"}
        assert any("oneOf/anyOf" in note for note in draft["generator"]["notes"])


# ---------------------------------------------------------------------------
# $ref to scalar/enum definitions (only object defs become models)
# ---------------------------------------------------------------------------


def write_schema(tmp_path: Path, schema: dict) -> Path:
    source = tmp_path / "schema.json"
    source.write_text(json.dumps(schema), encoding="utf-8")
    return source


class TestScalarAndEnumDefs:
    """A ``$ref`` to a non-object definition must not compile to an EMPTY
    model (which silently drops the definition's enum/type constraint)."""

    @pytest.fixture()
    def draft(self, tmp_path) -> dict:
        source = write_schema(
            tmp_path,
            {
                "title": "Order",
                "type": "object",
                "required": ["order_number"],
                "properties": {
                    "order_number": {"type": "string", "examples": ["A-1", "A-2"]},
                    "price_currency": {"$ref": "#/$defs/Currency"},
                    "refund_currency": {"$ref": "#/$defs/Currency"},
                    "money_code": {"$ref": "#/$defs/CurrencyAlias"},
                    "external_ref": {"$ref": "#/$defs/Reference"},
                    "due": {"$ref": "#/$defs/DueDate"},
                    "retries": {"$ref": "#/$defs/RetryCount"},
                },
                "$defs": {
                    "Currency": {"type": "string", "enum": ["EUR", "USD"]},
                    "CurrencyAlias": {"$ref": "#/$defs/Currency"},
                    "Reference": {"type": "string"},
                    "DueDate": {"type": "string", "format": "date"},
                    "RetryCount": {"type": "integer"},
                },
            },
        )
        draft, _ = spec_draft_from_jsonschema(source)
        return draft

    def test_enum_def_becomes_enum_spec_under_def_name(self, draft):
        assert draft["enums"] == [
            {
                "name": "Currency",
                "members": ["EUR", "USD"],
                "synonyms": {},
                "include_other": True,
            }
        ]
        currency = field(models_by_name(draft)["Order"], "price_currency")
        assert currency["type"] == "Currency"
        assert currency["normalizer"] == "enum"

    def test_enum_def_reused_across_fields_not_duplicated(self, draft):
        assert len(draft["enums"]) == 1
        refund = field(models_by_name(draft)["Order"], "refund_currency")
        assert refund["type"] == "Currency"

    def test_ref_chain_to_enum_def_followed(self, draft):
        money = field(models_by_name(draft)["Order"], "money_code")
        assert money["type"] == "Currency"
        assert money["normalizer"] == "enum"

    def test_scalar_alias_def_maps_to_scalar_field(self, draft):
        order = models_by_name(draft)["Order"]
        assert field(order, "external_ref")["type"] == "str"
        assert field(order, "due")["type"] == "date"  # format honoured
        assert field(order, "retries")["type"] == "int"

    def test_non_object_defs_emit_no_models(self, draft):
        assert set(models_by_name(draft)) == {"Order"}
        assert all(m["fields"] for m in draft["models"])  # no empty component

    def test_scalar_enum_def_draft_validates(self, draft):
        assert TemplateSpec.model_validate(draft).root == "Order"


# ---------------------------------------------------------------------------
# allOf/$ref cycles fail with an actionable error (design §9)
# ---------------------------------------------------------------------------


class TestAllofRefCycles:
    def test_self_referencing_allof_raises_naming_the_cycle(self, tmp_path):
        source = write_schema(
            tmp_path,
            {
                "title": "Doc",
                "type": "object",
                "properties": {"part": {"$ref": "#/$defs/Part"}},
                "$defs": {
                    "Part": {
                        "allOf": [{"$ref": "#/$defs/Part"}],
                        "properties": {"code": {"type": "string"}},
                    }
                },
            },
        )
        with pytest.raises(ValueError, match=r"cycle.*'Part'"):
            spec_draft_from_jsonschema(source)

    def test_mutual_allof_cycle_raises_naming_the_cycle(self, tmp_path):
        source = write_schema(
            tmp_path,
            {
                "title": "Doc",
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {
                    "A": {"allOf": [{"$ref": "#/$defs/B"}]},
                    "B": {"allOf": [{"$ref": "#/$defs/A"}]},
                },
            },
        )
        with pytest.raises(ValueError, match="cycle") as exc_info:
            spec_draft_from_jsonschema(source)
        assert "A" in str(exc_info.value)
        assert "B" in str(exc_info.value)

    def test_acyclic_allof_still_merges(self):
        # Regression guard: the cycle guard must not break legal allOf reuse.
        draft, _ = compile_fixture("claim_allof.schema.json")
        vehicle = models_by_name(draft)["InsuredVehicle"]
        assert {f["name"] for f in vehicle["fields"]} == {
            "make",
            "model_year",
            "registration_code",
        }


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruning:
    def test_depth_bounds_nested_refs(self):
        # Address sits at depth 2 (Invoice -> Party -> Address).
        draft, _ = compile_fixture("invoice.schema.json", depth=1)
        assert "Address" not in models_by_name(draft)
        assert not any(f["name"] == "address" for f in models_by_name(draft)["Party"]["fields"])
        assert any("depth" in note for note in draft["generator"]["notes"])

    def test_exclude_prunes_def_and_its_edges(self):
        draft, _ = compile_fixture("invoice.schema.json", exclude=["Party"])
        assert "Party" not in models_by_name(draft)
        invoice_fields = {f["name"] for f in models_by_name(draft)["Invoice"]["fields"]}
        assert "seller" not in invoice_fields
        assert "buyer" not in invoice_fields


# ---------------------------------------------------------------------------
# Draft validity
# ---------------------------------------------------------------------------


class TestDraftValidity:
    @pytest.mark.parametrize(
        "fixture",
        ["invoice.schema.json", "claim_allof.schema.json", "payment_oneof.schema.json"],
    )
    def test_draft_validates_directly(self, fixture):
        # Constructibility: repair_draft is a safety net, not a crutch.
        draft, _ = compile_fixture(fixture)
        spec = TemplateSpec.model_validate(draft)
        assert spec.root == draft["root"]

    def test_invalid_json_rejected(self, tmp_path):
        source = tmp_path / "broken.schema.json"
        source.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            spec_draft_from_jsonschema(source)


# ---------------------------------------------------------------------------
# Format sniffing + dispatcher
# ---------------------------------------------------------------------------


class TestSniffing:
    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            ("policy_basic.ttl", "owl"),
            ("skos_only.ttl", "owl"),
            ("library.yaml", "linkml"),
            ("invoice.schema.json", "jsonschema"),
            ("claim_allof.schema.json", "jsonschema"),
            ("payment_oneof.schema.json", "jsonschema"),
        ],
    )
    def test_sniffs_fixture_formats(self, fixture, expected):
        assert sniff_ontology_format(FIXTURES / fixture) == expected

    def test_sniffs_json_schema_content_without_extension_hint(self, tmp_path):
        source = tmp_path / "schema.txt"
        source.write_text('{"$schema": "https://json-schema.org/draft/2020-12/schema"}')
        assert sniff_ontology_format(source) == "jsonschema"

    def test_sniffs_linkml_content_without_extension_hint(self, tmp_path):
        source = tmp_path / "schema.txt"
        source.write_text("classes:\n  A: {}\nslots:\n  s: {}\n")
        assert sniff_ontology_format(source) == "linkml"

    def test_undetectable_content_fails_with_hint(self, tmp_path):
        pytest.importorskip("rdflib")
        source = tmp_path / "mystery.txt"
        source.write_text("just some free text, no ontology here")
        with pytest.raises(ValueError, match="--format"):
            sniff_ontology_format(source)


class TestDispatcher:
    def test_dispatches_jsonschema_end_to_end(self):
        draft, gaps = spec_draft_from_ontology(INVOICE)
        assert draft["generator"]["format"] == "jsonschema"
        assert draft["root"] == "Invoice"
        assert all(g.model for g in gaps)

    def test_explicit_format_skips_sniffing(self):
        draft, _ = spec_draft_from_ontology(INVOICE, fmt="jsonschema")
        assert draft["root"] == "Invoice"

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError, match="Unknown ontology format"):
            spec_draft_from_ontology(INVOICE, fmt="xsd")

    def test_max_models_cap_lists_largest_classes(self):
        with pytest.raises(ValueError, match="max_models=2"):
            spec_draft_from_ontology(INVOICE, max_models=2)

    def test_dispatcher_forwards_pruning_options(self):
        draft, _ = spec_draft_from_ontology(INVOICE, exclude=["Party"], depth=3)
        assert "Party" not in models_by_name(draft)


# ---------------------------------------------------------------------------
# Zero LLM calls — by construction
# ---------------------------------------------------------------------------

ONTOLOGY_MODULES = [ontology_pkg, owl_mod, linkml_mod, jsonschema_mod]


class TestZeroLlmByConstruction:
    @pytest.mark.parametrize(
        "module", ONTOLOGY_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
    )
    def test_no_llm_imports_anywhere(self, module):
        """The ontology path is a pure compiler: no client import can exist.

        Scans every import statement (module-level AND function-local, so lazy
        imports are covered) for llm_clients / litellm / anything llm-flavored.
        """
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module)
                imported.update(f"{node.module or ''}.{a.name}" for a in node.names)
        offenders = {name for name in imported if "llm" in name.lower()}
        assert not offenders, f"{module.__name__} imports LLM machinery: {offenders}"

    def test_compilers_run_without_any_client(self):
        # No client fixture, no mock, no network: the call simply succeeds.
        draft, _ = spec_draft_from_jsonschema(INVOICE)
        assert draft["models"]
