"""Docs-drift guards and behavioral checks for the templategen snippet library."""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from docling_graph.templategen import snippets

REPO_ROOT = Path(__file__).resolve().parents[3]
INSURANCE_CANON = REPO_ROOT / "docs" / "examples" / "templates" / "insurance_terms.py"
BILLING_CANON = REPO_ROOT / "docs" / "examples" / "templates" / "billing_document.py"
TEMPLATE_BASICS = REPO_ROOT / "docs" / "fundamentals" / "schema-definition" / "template-basics.md"
VALIDATION_DOC = REPO_ROOT / "docs" / "fundamentals" / "schema-definition" / "validation.md"
GRAPH_CONVERTER = REPO_ROOT / "docling_graph" / "core" / "converters" / "graph_converter.py"
DENSE_CATALOG = (
    REPO_ROOT / "docling_graph" / "core" / "extractors" / "contracts" / "dense" / "catalog.py"
)


def _split_edge_helper() -> tuple[str, str]:
    """Split EDGE_HELPER into (signature, body) around its docstring.

    The docstring is deliberately English (the canon's is French); signature
    and body are pinned character-for-character against the canon file.
    """
    signature, _, rest = snippets.EDGE_HELPER.partition('"""')
    _, _, body = rest.partition('"""')
    return signature, body


# ---------------------------------------------------------------------------
# Docs-drift guards
# ---------------------------------------------------------------------------


class TestEdgeHelperDriftGuards:
    def test_signature_matches_insurance_canon_character_for_character(self):
        canon = INSURANCE_CANON.read_text(encoding="utf-8")
        signature, _ = _split_edge_helper()
        assert signature in canon, (
            "EDGE_HELPER signature drifted from docs/examples/templates/insurance_terms.py"
        )

    def test_body_matches_insurance_canon_character_for_character(self):
        canon = INSURANCE_CANON.read_text(encoding="utf-8")
        _, body = _split_edge_helper()
        assert body in canon, (
            "EDGE_HELPER body drifted from docs/examples/templates/insurance_terms.py"
        )

    def test_metadata_keys_present_in_helper(self):
        assert f'"{snippets.EDGE_METADATA_KEY_LABEL}"' in snippets.EDGE_HELPER
        assert f'"{snippets.EDGE_METADATA_KEY_REFERENCE}"' in snippets.EDGE_HELPER
        assert f'"{snippets.EDGE_METADATA_KEY_CLOSED_CATALOG}"' in snippets.EDGE_HELPER

    def test_metadata_keys_match_what_the_runtime_reads(self):
        # GraphConverter reads edge_label and reference_closed_catalog; the
        # dense catalog reads graph_reference. If either file stops mentioning
        # its key, the emitted metadata would silently become inert.
        converter_src = GRAPH_CONVERTER.read_text(encoding="utf-8")
        catalog_src = DENSE_CATALOG.read_text(encoding="utf-8")
        assert f'"{snippets.EDGE_METADATA_KEY_LABEL}"' in converter_src
        assert f'"{snippets.EDGE_METADATA_KEY_CLOSED_CATALOG}"' in converter_src
        assert f'"{snippets.EDGE_METADATA_KEY_REFERENCE}"' in catalog_src

    def test_billing_canon_single_edges_default_optional(self):
        # Canon reconciliation (design §7.1): billing_document.py's helper must
        # follow the Optionality Law — single edges default to None, never `...`.
        billing = BILLING_CANON.read_text(encoding="utf-8")
        assert 'kwargs["default"] = None' in billing
        assert 'kwargs["default"] = ...' not in billing


class TestImportBlockDriftGuards:
    def test_import_block_matches_template_basics(self):
        doc = TEMPLATE_BASICS.read_text(encoding="utf-8")
        assert snippets.IMPORT_BLOCK in doc, (
            "IMPORT_BLOCK drifted from template-basics.md's required import block"
        )

    def test_optional_imports_appear_in_docs(self):
        basics = TEMPLATE_BASICS.read_text(encoding="utf-8")
        validation = VALIDATION_DOC.read_text(encoding="utf-8")
        assert snippets.OPTIONAL_IMPORT_DATETIME in basics
        assert snippets.OPTIONAL_IMPORT_ENUM in basics
        assert snippets.OPTIONAL_IMPORT_RE in basics
        assert snippets.OPTIONAL_IMPORT_LOGGING in validation


class TestNeverRaiseLaw:
    def test_no_snippet_contains_raise(self):
        # The never-reject law holds by construction: raising validators simply
        # do not exist in the snippet library.
        constants = {
            name: value
            for name, value in vars(snippets).items()
            if name.isupper() and isinstance(value, str)
        }
        assert constants, "snippet library exposes no constants?"
        for name, value in constants.items():
            assert "raise " not in value, f"snippet {name} contains a raise statement"


# ---------------------------------------------------------------------------
# Behavioral checks (the snippets are real code — prove they run)
# ---------------------------------------------------------------------------


class TestEdgeHelperBehavior:
    @pytest.fixture()
    def edge(self):
        namespace: dict[str, Any] = {"Any": Any, "Field": Field}
        exec(snippets.EDGE_HELPER, namespace)
        return namespace["edge"]

    def test_single_edge_defaults_to_none(self, edge):
        class Target(BaseModel):
            model_config = ConfigDict(graph_id_fields=["name"])
            name: str

        class Holder(BaseModel):
            single: Target | None = edge(label="HAS_TARGET")

        assert Holder().single is None

    def test_list_edge_keeps_default_factory(self, edge):
        class Target(BaseModel):
            model_config = ConfigDict(graph_id_fields=["name"])
            name: str

        class Holder(BaseModel):
            many: List[Target] = edge(label="CONTAINS_ITEM", default_factory=list)

        assert Holder().many == []

    def test_metadata_lands_in_json_schema_extra(self, edge):
        class Target(BaseModel):
            model_config = ConfigDict(graph_id_fields=["name"])
            name: str

        class Holder(BaseModel):
            single: Target | None = edge(label="HAS_TARGET")
            refs: List[Target] = edge(
                label="EXCLUDES_ITEM",
                default_factory=list,
                reference=True,
                closed_catalog=True,
            )

        single_extra = Holder.model_fields["single"].json_schema_extra
        assert single_extra == {"edge_label": "HAS_TARGET"}
        refs_extra = Holder.model_fields["refs"].json_schema_extra
        assert refs_extra == {
            "edge_label": "EXCLUDES_ITEM",
            "graph_reference": True,
            "reference_closed_catalog": True,
        }


class TestNormalizeEnumBehavior:
    @pytest.fixture()
    def normalize(self):
        import logging
        import re
        from typing import Type

        namespace: dict[str, Any] = {
            "Any": Any,
            "Type": Type,
            "Enum": Enum,
            "re": re,
            "logger": logging.getLogger("templategen-test"),
        }
        exec(snippets.NORMALIZE_ENUM_HELPER, namespace)
        return namespace["_normalize_enum"]

    def test_maps_case_and_separator_variants(self, normalize):
        class DocumentType(str, Enum):
            CREDIT_NOTE = "Credit Note"
            OTHER = "Other"

        assert normalize(DocumentType, "credit note") is DocumentType.CREDIT_NOTE
        assert normalize(DocumentType, "CREDIT_NOTE") is DocumentType.CREDIT_NOTE
        assert normalize(DocumentType, "Credit Note") is DocumentType.CREDIT_NOTE

    def test_falls_back_to_other_instead_of_raising(self, normalize):
        class DocumentType(str, Enum):
            INVOICE = "Invoice"
            OTHER = "Other"

        assert normalize(DocumentType, "warranty claim") is DocumentType.OTHER
        assert normalize(DocumentType, 42) is DocumentType.OTHER

    def test_without_other_returns_value_unchanged(self, normalize):
        class Rigid(str, Enum):
            A = "A"

        assert normalize(Rigid, "unknown") == "unknown"

    def test_passes_through_enum_instances(self, normalize):
        class DocumentType(str, Enum):
            INVOICE = "Invoice"
            OTHER = "Other"

        assert normalize(DocumentType, DocumentType.INVOICE) is DocumentType.INVOICE


class TestValidatorTemplates:
    @pytest.fixture()
    def sample_model(self):
        source = "\n".join(
            [
                snippets.IMPORT_BLOCK,
                snippets.OPTIONAL_IMPORT_ENUM,
                snippets.OPTIONAL_IMPORT_LOGGING,
                snippets.OPTIONAL_IMPORT_RE,
                snippets.LOGGER_SETUP,
                "",
                snippets.NORMALIZE_ENUM_HELPER,
                "",
                "class Status(str, Enum):",
                '    ACTIVE = "Active"',
                '    ON_HOLD = "On Hold"',
                '    OTHER = "Other"',
                "",
                "class Sample(BaseModel):",
                "    amount: Optional[float] = Field(None)",
                "    currency: Optional[str] = Field(None)",
                "    tags: List[str] = Field(default_factory=list)",
                "    status: Status = Field(Status.OTHER)",
                "    statuses: List[Status] = Field(default_factory=list)",
                snippets.NUMERIC_VALIDATOR_TEMPLATE.format(field="amount"),
                snippets.CURRENCY_VALIDATOR_TEMPLATE.format(field="currency"),
                snippets.STRING_LIST_VALIDATOR_TEMPLATE.format(field="tags"),
                snippets.ENUM_FIELD_VALIDATOR_TEMPLATE.format(field="status", enum_name="Status"),
                snippets.ENUM_LIST_FIELD_VALIDATOR_TEMPLATE.format(
                    field="statuses", enum_name="Status"
                ),
                snippets.ROOT_LIST_DEDUP_TEMPLATE.format(
                    field="tags", key_expr="str(item).strip().lower()"
                ),
                snippets.STR_METHOD_TEMPLATE.format(parts="self.currency, self.amount"),
            ]
        )
        namespace: dict[str, Any] = {}
        exec(compile(source, "<generated-sample>", "exec"), namespace)
        return namespace["Sample"]

    def test_numeric_coercion(self, sample_model):
        assert sample_model(amount="1 500,00 €").amount == 1500.0
        assert sample_model(amount="$1,500.00").amount == 1500.0
        assert sample_model(amount=12.5).amount == 12.5

    def test_numeric_never_rejects(self, sample_model):
        assert sample_model(amount="not a number").amount is None

    def test_currency_normalization(self, sample_model):
        assert sample_model(currency="€").currency == "EUR"
        assert sample_model(currency="usd").currency == "USD"

    def test_string_list_coercion_and_dedup(self, sample_model):
        instance = sample_model(tags="alpha, beta, Alpha")
        assert instance.tags == ["alpha", "beta"]

    def test_enum_normalization(self, sample_model):
        assert sample_model(status="active").status.value == "Active"
        assert sample_model(status="void").status.value == "Other"

    def test_enum_list_normalizes_every_item_never_rejects(self, sample_model):
        # Regression (never-reject law for List[Enum] fields): case variants
        # and separator variants normalize per item; unknowns fall to OTHER
        # instead of raising ValidationError.
        instance = sample_model(statuses=["active", "ON_HOLD", "on hold", "bogus value"])
        assert [s.value for s in instance.statuses] == ["Active", "On Hold", "On Hold", "Other"]

    def test_enum_list_delegates_scalar_input_to_normalizer(self, sample_model):
        # Non-list input goes through the scalar path (Pydantic then judges
        # the list shape) — the validator itself never raises.
        instance = sample_model(statuses=["ACTIVE"])
        assert [s.value for s in instance.statuses] == ["Active"]

    def test_str_method_with_unknown_fallback(self, sample_model):
        assert str(sample_model(currency="EUR", amount="5")) == "EUR 5.0"
        assert str(sample_model()) == "Unknown"
