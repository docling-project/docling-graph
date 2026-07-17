"""Tests for template -> SPEC reconstruction (the ``template lint`` reverse flow).

Covers the three contracts of ``reverse.py``:

- **round-trip fidelity**: every golden fixture SPEC survives
  render -> exec -> reverse structurally intact (kinds, identity, edges with
  their labels/reference/closed_catalog markers, cardinality bounds, enums);
- **canon-lint regression**: reversing the shipped production templates
  (``docs/examples/templates/``) yields exactly the documented repair sets —
  pinning both the reverse walk and the linter's judgment of the canon;
- **leniency**: constructs the IR cannot represent become findings, never
  crashes; and the dotted-path entry point enforces the import allowlist
  *before* executing anything.
"""

import importlib
import importlib.util
import subprocess
import sys
import types
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

from docling_graph.templategen.linter import TemplateLintError
from docling_graph.templategen.renderer import render_template
from docling_graph.templategen.reverse import (
    reverse_draft,
    spec_from_dotted_path,
    spec_from_template,
)
from docling_graph.templategen.spec import TemplateSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "tests" / "fixtures" / "templategen" / "specs"
CANON_DIR = REPO_ROOT / "docs" / "examples" / "templates"

FIXTURE_NAMES = sorted(path.stem for path in SPEC_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_spec(name: str) -> TemplateSpec:
    return TemplateSpec.from_yaml((SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def exec_rendered(source: str, module_name: str) -> types.ModuleType:
    """Execute rendered template source inside a registered module (like V2)."""
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    sys.modules[module_name] = module
    exec(compile(source, f"<{module_name}>", "exec"), module.__dict__)
    return module


@pytest.fixture()
def module_registry(monkeypatch):
    """Track dynamically created modules and drop them from sys.modules after."""
    created: list[str] = []

    def register(name: str) -> str:
        created.append(name)
        return name

    yield register
    for name in created:
        sys.modules.pop(name, None)


def load_canon_module(stem: str) -> types.ModuleType:
    path = CANON_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def edges_of_draft(draft: dict) -> set[tuple]:
    return {
        (
            model["name"],
            field["name"],
            field["type"],
            field.get("edge_label"),
            field.get("reference", False),
            field.get("closed_catalog", False),
            field["is_list"],
        )
        for model in draft["models"]
        for field in model["fields"]
        if field["role"] == "edge"
    }


def edges_of_spec(spec: TemplateSpec) -> set[tuple]:
    return {
        (m.name, f.name, f.type, f.edge_label, f.reference, f.closed_catalog, f.is_list)
        for m in spec.models
        for f in m.fields
        if f.role == "edge"
    }


def normalized_enum_members(members: list[str], include_other: bool) -> set[str]:
    # The renderer appends OTHER = "Other" for include_other enums; a literal
    # "Other" member and the safety-net member are indistinguishable in code.
    return set(members) | ({"Other"} if include_other else set())


# ---------------------------------------------------------------------------
# Round-trip: fixture SPEC -> render -> exec -> reverse -> compare
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
class TestRoundTrip:
    @pytest.fixture()
    def reversed_draft(self, name, module_registry):
        spec = load_spec(name)
        source = render_template(spec)
        module = exec_rendered(source, module_registry(f"templategen_roundtrip_{name}"))
        root_cls = getattr(module, spec.root)
        draft, findings = reverse_draft(root_cls)
        return spec, draft, findings, root_cls

    def test_rendered_canon_reverses_without_findings(self, reversed_draft):
        # Everything the renderer can emit, the reverse walk can represent.
        _, _, findings, _ = reversed_draft
        assert findings == []

    def test_root_and_kinds_recovered(self, reversed_draft):
        spec, draft, _, _ = reversed_draft
        assert draft["root"] == spec.root
        assert {m["name"]: m["kind"] for m in draft["models"]} == {
            m.name: m.kind for m in spec.models
        }

    def test_identity_fields_recovered(self, reversed_draft):
        spec, draft, _, _ = reversed_draft
        assert {m["name"]: m["identity_fields"] for m in draft["models"]} == {
            m.name: m.identity_fields for m in spec.models
        }

    def test_identity_examples_recovered_verbatim(self, reversed_draft):
        spec, draft, _, _ = reversed_draft
        draft_examples = {
            (m["name"], f["name"]): f["examples"]
            for m in draft["models"]
            for f in m["fields"]
            if f["role"] == "identity"
        }
        for model in spec.models:
            for field in model.fields:
                if field.role == "identity":
                    assert draft_examples[(model.name, field.name)] == field.examples

    def test_edges_with_markers_recovered(self, reversed_draft):
        spec, draft, _, _ = reversed_draft
        assert edges_of_draft(draft) == edges_of_spec(spec)

    def test_max_instances_round_trips_through_doubling(self, reversed_draft):
        # The spec stores the doubled bound; drafts carry the documented
        # figure, which repair_draft doubles exactly once (linter contract).
        spec, draft, _, _ = reversed_draft
        for model in spec.models:
            drafted = next(m for m in draft["models"] if m["name"] == model.name)
            if model.max_instances is None:
                assert drafted.get("max_instances") is None
            else:
                assert drafted["max_instances"] * 2 == model.max_instances

    def test_enums_recovered(self, reversed_draft):
        spec, draft, _, _ = reversed_draft
        drafted = {e["name"]: e for e in draft["enums"]}
        assert set(drafted) == {e.name for e in spec.enums}
        for enum in spec.enums:
            entry = drafted[enum.name]
            assert entry["include_other"] == enum.include_other
            assert normalized_enum_members(
                entry["members"], entry["include_other"]
            ) == normalized_enum_members(enum.members, enum.include_other)

    def test_spec_from_template_repairs_to_valid_spec(self, reversed_draft):
        spec, _, _, root_cls = reversed_draft
        recovered, report = spec_from_template(root_cls)
        assert isinstance(recovered, TemplateSpec)
        assert recovered.root == spec.root
        assert {m.name: m.kind for m in recovered.models} == {m.name: m.kind for m in spec.models}
        assert {m.name: tuple(m.identity_fields) for m in recovered.models} == {
            m.name: tuple(m.identity_fields) for m in spec.models
        }
        # max_instances survives the reverse + re-repair doubling unchanged.
        assert {m.name: m.max_instances for m in recovered.models} == {
            m.name: m.max_instances for m in spec.models
        }
        # No structural repairs beyond re-derivable bookkeeping (R19 re-adds
        # the root dedup entries the reverse walk cannot see in compiled
        # validators — and must re-derive them identically).
        assert {e.rule_id for e in report.entries if e.repaired} <= {"R19"}
        assert set(recovered.needs_root_list_dedup) == set(spec.needs_root_list_dedup)


# ---------------------------------------------------------------------------
# Canon-lint regression: the shipped production templates
# ---------------------------------------------------------------------------


class TestBillingCanon:
    """docs/examples/templates/billing_document.py — expected repair set.

    The canon is intentionally NOT repair-free; each entry below is a known,
    documented judgment of the linter over the shipped template:

    - R10 does NOT fire: seller/buyer both nest Party in full as single
      (non-list) edges from the same parent — the multi-role exception keys
      on that shape (regardless of Party's field count), so neither edge is
      flipped and no buyer data is lost.
    - R6:  Item.name's identity description says "never invent or number
      items"; the invention-verb scrub deletes the sentence (advisory rule).
    - R16: currency / Payment.due_date / DocumentReference.ref_date
      descriptions instruct conversion ("CONVERT: EUR", "Parse and convert").
    - R19: taxes/references are identity-less (component) root lists — the
      dedup model_validator is scheduled.
    """

    EXPECTED_REPAIR_RULES = {"R6", "R16", "R19"}

    @pytest.fixture()
    def billing_root(self, module_registry):
        module = load_canon_module(module_registry("billing_document"))
        return module.BillingDocument

    def test_reverse_walk_is_finding_free(self, billing_root):
        # Everything in the billing canon is representable in the IR.
        _, findings = reverse_draft(billing_root)
        assert findings == []

    def test_structure_recovered(self, billing_root):
        draft, _ = reverse_draft(billing_root)
        assert draft["root"] == "BillingDocument"
        assert {m["name"]: m["kind"] for m in draft["models"]} == {
            "BillingDocument": "root",
            "Party": "entity",
            "Item": "entity",
            "LineItem": "entity",
            "Tax": "component",
            "Payment": "component",
            "Delivery": "component",
            "DocumentReference": "component",
        }
        labels = {edge[3] for edge in edges_of_draft(draft)}
        assert labels == {
            "ISSUED_BY",
            "BILLED_TO",
            "CONTAINS_LINE",
            "REFERENCES_ITEM",
            "HAS_TAX",
            "HAS_PAYMENT_INFO",
            "HAS_DELIVERY_INFO",
            "REFERENCES_DOCUMENT",
        }
        # The shipped canon uses no reference/closed-catalog edges.
        assert all(not edge[4] and not edge[5] for edge in edges_of_draft(draft))

    def test_enums_recovered_with_other_detection(self, billing_root):
        draft, _ = reverse_draft(billing_root)
        enums = {e["name"]: e for e in draft["enums"]}
        assert set(enums) == {"DocumentType", "TaxType", "PaymentMethod"}
        assert all(e["include_other"] for e in enums.values())
        assert enums["DocumentType"]["members"] == [
            "Invoice",
            "Credit Note",
            "Debit Note",
            "Receipt",
        ]

    def test_repair_report_matches_documented_canon_set(self, billing_root):
        spec, report = spec_from_template(billing_root)
        repaired = {e.rule_id for e in report.entries if e.repaired}
        assert repaired == self.EXPECTED_REPAIR_RULES
        by_rule = {
            rule: {(e.model, e.field) for e in report.entries if e.repaired and e.rule_id == rule}
            for rule in repaired
        }
        assert by_rule["R6"] == {("Item", "name")}
        assert by_rule["R16"] == {
            ("BillingDocument", "currency"),
            ("Payment", "due_date"),
            ("DocumentReference", "ref_date"),
        }
        assert by_rule["R19"] == {
            ("BillingDocument", "taxes"),
            ("BillingDocument", "references"),
        }
        # The repaired spec is valid and keeps the canonical structure: the
        # multi-role seller/buyer Party edges BOTH stay full (no R10 flip, no
        # buyer data loss — the regression the shape-keyed exception pins).
        assert "R10" not in repaired
        by_name = {f.name: f for m in spec.models if m.name == "BillingDocument" for f in m.fields}
        assert by_name["buyer"].reference is False
        assert by_name["seller"].reference is False


class TestInsuranceCanon:
    """docs/examples/templates/insurance_terms.py — expected repair set.

    - REV findings: the root identity has a default (renderer would make it
      required), Bien.nom carries 7 examples (IR caps at 5), and
      Franchise.montant nests a component without edge() metadata.
    - R21: Franchise.type collides with the reserved node-attr key ``type``
      written by GraphConverter — renamed to ``category`` (the latent-bug
      catch the design calls out).
    - R9:  every French edge label (AGARANTIE, COUVREBIEN, ...) reads as a
      bare noun to the English verb vocabulary — normalized to HAS_<NOUN>.
    - R10: Exclusion nests in full at the root AND under Garantie; the root
      home wins and Garantie.exclusions_specifiques flips to reference.
    - R12/R11: every inbound edge to Bien is a reference, so no canonical
      home exists — the closed_catalog marker is cleared and one path is
      un-referenced so Bien keeps a full home.
    - R16: Option.etend_garanties says "Do not leave empty ..." (a global
      prompt-rule restatement).
    - R4:  Exclusion's docstring overruns the 240-char Phase-1 window.
    """

    EXPECTED_REPAIR_RULES = {"R21", "R9", "R10", "R12", "R11", "R16", "R4"}

    @pytest.fixture()
    def insurance_root(self, module_registry):
        module = load_canon_module(module_registry("insurance_terms"))
        return module.AssuranceMRH

    def test_reverse_findings_are_the_documented_three(self, insurance_root):
        _, findings = reverse_draft(insurance_root)
        assert len(findings) == 3
        assert any("reference_document" in f and "not required" in f for f in findings)
        assert any("Bien.nom" in f and "truncated to 5" in f for f in findings)
        assert any("Franchise.montant" in f and "without edge() metadata" in f for f in findings)

    def test_structure_and_reference_markers_recovered(self, insurance_root):
        draft, _ = reverse_draft(insurance_root)
        assert {m["name"]: m["kind"] for m in draft["models"]} == {
            "AssuranceMRH": "root",
            "Garantie": "entity",
            "Offre": "entity",
            "Option": "entity",
            "Exclusion": "entity",
            "Bien": "entity",
            "Montant": "component",
            "Franchise": "component",
            "Condition": "component",
        }
        markers = {(edge[0], edge[1]): (edge[4], edge[5]) for edge in edges_of_draft(draft)}
        assert markers[("Exclusion", "biens_exclus")] == (True, True)
        assert markers[("Garantie", "biens_couverts")] == (True, False)
        assert markers[("Option", "biens_couverts")] == (True, False)
        assert markers[("Option", "etend_garanties")] == (True, False)
        assert markers[("Offre", "garanties_incluses")] == (True, False)
        assert markers[("Offre", "options_disponibles")] == (True, False)
        assert markers[("AssuranceMRH", "garanties")] == (False, False)

    def test_repair_report_matches_documented_canon_set(self, insurance_root):
        spec, report = spec_from_template(insurance_root)
        repaired = {e.rule_id for e in report.entries if e.repaired}
        assert repaired == self.EXPECTED_REPAIR_RULES
        repaired_locations = {(e.rule_id, e.model, e.field) for e in report.entries if e.repaired}
        assert ("R21", "Franchise", "type") in repaired_locations
        assert ("R10", "Garantie", "exclusions_specifiques") in repaired_locations
        assert ("R12", "Exclusion", "biens_exclus") in repaired_locations
        assert ("R11", "Garantie", "biens_couverts") in repaired_locations
        assert ("R4", "Exclusion", None) in repaired_locations
        # Every French-labelled edge field normalizes to HAS_<NOUN> (13
        # distinct field names; biens_couverts recurs on Garantie and Option).
        r9_fields = {e.field for e in report.entries if e.repaired and e.rule_id == "R9"}
        assert len(r9_fields) == 13
        franchise = next(m for m in spec.models if m.name == "Franchise")
        assert "category" in {f.name for f in franchise.fields}
        assert "type" not in {f.name for f in franchise.fields}


# ---------------------------------------------------------------------------
# Leniency: non-representable constructs yield findings, not crashes
# ---------------------------------------------------------------------------


class WeirdKind(Enum):  # not a (str, Enum); int values
    ALPHA = 1
    BETA = 2


class Widget(BaseModel):
    model_config = ConfigDict(graph_id_fields=["serial", "batch", "shelf"])

    serial: str = Field(...)
    batch: str = Field(...)
    shelf: str = Field(...)
    mandatory_note: str = Field(...)  # required non-identity (Optionality Law)
    metadata: Dict[str, str] = Field(default_factory=dict)  # unrepresentable type
    mixed: Union[int, str, None] = Field(None)  # multi-type union
    kind: WeirdKind | None = Field(None)  # int-valued enum
    tags: frozenset[str] = Field(default_factory=frozenset)  # non-list container
    callable_extra: str | None = Field(
        None, json_schema_extra=lambda schema: schema.pop("default", None)
    )
    over_exampled: str | None = Field(None, examples=["a", "b", "c", "d", "e", "f", "g"])


class Orphan(BaseModel):
    # Neither graph_id_fields nor is_entity=False: runtime treats it as an
    # entity, but the IR has no identity-less entity — demoted to component.
    note: str | None = Field(None)


class WeirdRoot(BaseModel):
    model_config = ConfigDict(graph_id_fields=["ref"], graph_max_instances=5)

    ref: str = Field(..., examples=["R-1", "R-2"])
    widget: Widget | None = Field(None, json_schema_extra={"edge_label": "HAS_WIDGET"})
    orphan: Orphan | None = Field(None)  # model-typed, no edge metadata
    scalar_edge: str | None = Field(None, json_schema_extra={"edge_label": "HAS_NOTHING"})
    stray_marker: str | None = Field(None, json_schema_extra={"graph_reference": True})


class TestLeniency:
    def test_every_oddity_becomes_a_finding_and_repair_still_constructs(self):
        _draft, findings = reverse_draft(WeirdRoot)
        joined = "\n".join(findings)
        assert "3 identity fields exceed" in joined
        assert "required non-identity field" in joined and "mandatory_note" in joined
        assert "unknown scalar type" in joined  # Dict[str, str]
        assert "multi-type union" in joined  # Union[int, str]
        assert "non-string value" in joined  # WeirdKind int members
        assert "'frozenset' container treated as a list" in joined
        assert "json_schema_extra is not a mapping" in joined
        assert "7 examples truncated to 5" in joined
        assert "graph_max_instances=5 is not an even 2x bound" in joined
        assert "no graph_id_fields" in joined and "Orphan" in joined
        assert "model-typed field without edge() metadata" in joined
        assert "edge_label 'HAS_NOTHING' on a non-model field" in joined
        assert "reference markers without an edge_label" in joined

        # The loose draft must still repair into a valid spec — no crashes.
        spec, report = spec_from_template(WeirdRoot)
        assert isinstance(spec, TemplateSpec)
        assert spec.root == "WeirdRoot"
        widget = next(m for m in spec.models if m.name == "Widget")
        assert len(widget.identity_fields) == 2  # R1 trims to the identity budget
        orphan = next(m for m in spec.models if m.name == "Orphan")
        assert orphan.kind == "component"  # never invent ids
        root = next(m for m in spec.models if m.name == "WeirdRoot")
        assert root.max_instances == 4  # documented 2 (5 // 2), re-doubled once
        # Findings surface in the lint report as REV info entries.
        assert any(e.rule_id == "REV" and e.severity == "info" for e in report.entries)

    def test_reverse_draft_rejects_non_template_input(self):
        with pytest.raises(TypeError, match="BaseModel subclass"):
            reverse_draft("not a class")  # type: ignore[arg-type]

    def test_cycles_are_walked_once(self):
        class Node(BaseModel):
            model_config = ConfigDict(graph_id_fields=["name"])
            name: str = Field(...)
            children: List["Node"] = Field(
                default_factory=list, json_schema_extra={"edge_label": "HAS_CHILD"}
            )

        Node.model_rebuild()
        draft, _ = reverse_draft(Node)
        assert [m["name"] for m in draft["models"]] == ["Node"]
        child_edge = draft["models"][0]["fields"][1]
        assert child_edge["type"] == "Node" and child_edge["role"] == "edge"

    def test_duplicate_class_names_are_disambiguated(self):
        def make_part() -> type[BaseModel]:
            class Part(BaseModel):
                model_config = ConfigDict(is_entity=False)
                text: str | None = Field(None)

            return Part

        part_a, part_b = make_part(), make_part()
        dup_root = create_model(
            "DupRoot",
            __config__=ConfigDict(graph_id_fields=["ref"]),
            ref=(str, Field(...)),
            first=(part_a | None, Field(None)),
            second=(part_b | None, Field(None)),
        )
        draft, findings = reverse_draft(dup_root)
        names = [m["name"] for m in draft["models"]]
        assert "Part" in names and "Part_2" in names
        assert any("duplicate class name" in f for f in findings)


# ---------------------------------------------------------------------------
# Dotted-path loading + the import-allowlist pre-check
# ---------------------------------------------------------------------------


GOOD_MODULE = '''
"""A minimal allowlist-clean template module."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Doc(BaseModel):
    """A document identified by its printed reference."""

    model_config = ConfigDict(graph_id_fields=["ref"])

    ref: str = Field(..., examples=["A-1", "B-2"])
    note: Optional[str] = Field(None, description="Free-form note printed at the bottom.")
'''

BAD_MODULE = """
import os

from pydantic import BaseModel, ConfigDict, Field


class Doc(BaseModel):
    model_config = ConfigDict(graph_id_fields=["ref"])

    ref: str = Field(...)
"""


class TestDottedPath:
    def test_loads_and_lints_an_allowlist_clean_module(
        self, tmp_path, monkeypatch, module_registry
    ):
        module_registry("linted_good_template")
        (tmp_path / "linted_good_template.py").write_text(GOOD_MODULE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        spec, report, template = spec_from_dotted_path("linted_good_template.Doc")
        assert spec.root == "Doc"
        assert template.__name__ == "Doc"
        assert issubclass(template, BaseModel)
        assert not any(e.repaired for e in report.entries)

    def test_allowlist_violation_rejects_before_import(
        self, tmp_path, monkeypatch, module_registry
    ):
        module_registry("linted_bad_template")
        (tmp_path / "linted_bad_template.py").write_text(BAD_MODULE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TemplateLintError, match="import allowlist") as exc_info:
            spec_from_dotted_path("linted_bad_template.Doc")
        # The precheck must reject BEFORE executing the module.
        assert "linted_bad_template" not in sys.modules
        report = exc_info.value.report
        assert [e.rule_id for e in report.entries] == ["V1b"]
        assert "import os" in report.entries[0].message

    def test_unparseable_module_is_rejected_without_import(
        self, tmp_path, monkeypatch, module_registry
    ):
        module_registry("linted_broken_template")
        (tmp_path / "linted_broken_template.py").write_text("def broken(:\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TemplateLintError, match="does not parse"):
            spec_from_dotted_path("linted_broken_template.Doc")
        assert "linted_broken_template" not in sys.modules

    def test_missing_module_raises_the_loader_error(self):
        from docling_graph.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            spec_from_dotted_path("no_such_module_anywhere.Doc")


# ---------------------------------------------------------------------------
# Public API: import works without the templategen extra, heavy paths lazy
# ---------------------------------------------------------------------------


class TestPublicApi:
    PUBLIC_API = [
        "TemplateSpec",
        "EnumSpec",
        "FieldSpec",
        "ModelSpec",
        "SpecGap",
        "ScalarType",
        "repair_draft",
        "lint_spec",
        "LintReport",
        "TemplateLintError",
        "render_template",
        "verify_template_source",
        "VerificationReport",
        "synthesize_sample",
        "spec_draft_from_ontology",
        "induce_spec_from_documents",
        "DocumentContent",
        "fill_gaps",
        "reverse_draft",
        "spec_from_template",
        "spec_from_dotted_path",
        "evaluate_template",
        "EvaluationReport",
        "generate_template",
        "GenerationResult",
        "build_llm_call_fn",
        "TemplateGenSettings",
        "load_templategen_settings",
    ]

    # Spawns a cold interpreter that imports the whole package: comfortably over
    # pytest.ini's 10s default on a CI runner. The inner subprocess timeout below
    # stays lower so a genuine hang is still reported as one.
    @pytest.mark.timeout(240)
    def test_imports_without_optional_deps_and_defers_heavy_paths(self):
        # Subprocess: a pristine interpreter with rdflib/linkml_runtime blocked
        # (the sys.modules[None] pattern of the ontology tests), so the check
        # cannot be satisfied by modules another test already imported.
        script = f"""
import sys

sys.modules["rdflib"] = None
sys.modules["linkml_runtime"] = None

import docling_graph.templategen as tg

assert sorted(tg.__all__) == sorted({self.PUBLIC_API!r}), tg.__all__

lazy_modules = (
    "docling_graph.templategen.evaluate",
    "docling_graph.templategen.ontology",
    "docling_graph.templategen.induce.documents",
    "docling_graph.templategen.induce.gapfill",
)
eager = [name for name in lazy_modules if name in sys.modules]
assert not eager, f"heavy paths imported eagerly: {{eager}}"

# Every export resolves — including the lazy ones, still without rdflib.
for name in tg.__all__:
    assert getattr(tg, name) is not None, name
assert "docling_graph.templategen.evaluate" in sys.modules
assert set(tg.__all__) <= set(dir(tg))
print("PUBLIC-API-OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "PUBLIC-API-OK" in result.stdout

    def test_public_api_matches_all(self):
        package = importlib.import_module("docling_graph.templategen")
        assert sorted(package.__all__) == sorted(self.PUBLIC_API)

    def test_unknown_attribute_raises(self):
        package = importlib.import_module("docling_graph.templategen")
        with pytest.raises(AttributeError, match="no attribute"):
            _ = package.does_not_exist
