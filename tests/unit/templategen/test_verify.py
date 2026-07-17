"""Unit tests for the V1-V6 verification gate.

V4/V6 run against the REAL dense catalog and GraphConverter (pure Python +
networkx, no mocks) — the whole point of the gate is proving templates against
the actual runtime surfaces.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from docling_graph.templategen.renderer import render_template
from docling_graph.templategen.spec import FieldSpec, ModelSpec, TemplateSpec
from docling_graph.templategen.verify import (
    GateResult,
    VerificationReport,
    synthesize_sample,
    synthesize_sample_plan,
    verify_template_source,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "tests" / "fixtures" / "templategen" / "specs"


def load_spec(name: str) -> TemplateSpec:
    return TemplateSpec.from_yaml((SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def invoice_spec() -> TemplateSpec:
    return load_spec("invoice")


@pytest.fixture(scope="module")
def invoice_source(invoice_spec) -> str:
    return render_template(invoice_spec)


def gate(report: VerificationReport, gate_id: str) -> GateResult:
    return next(result for result in report.gates if result.gate == gate_id)


# ---------------------------------------------------------------------------
# Full pass on golden templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["invoice", "policy_ontology", "org_chart"])
def test_all_gates_pass_on_rendered_fixture(name):
    spec = load_spec(name)
    report = verify_template_source(render_template(spec), root_class=spec.root, spec=spec)
    assert report.passed, report.summary()
    assert [g.gate for g in report.gates] == ["V1", "V1b", "V2", "V3", "V4", "V5", "V6"]
    assert not any(g.skipped for g in report.gates)


def test_semantic_guide_preview_in_report(invoice_spec, invoice_source):
    report = verify_template_source(invoice_source, root_class=invoice_spec.root, spec=invoice_spec)
    assert report.semantic_guide_preview.startswith("Field guidance:")
    assert "document_number" in report.semantic_guide_preview


def test_summary_and_failures_api(invoice_spec, invoice_source):
    report = verify_template_source(invoice_source, root_class=invoice_spec.root, spec=invoice_spec)
    assert report.failures() == []
    summary = report.summary()
    assert "V1   PASS" in summary
    assert "V6   PASS" in summary


def test_without_spec_v6_skipped_by_design(invoice_spec, invoice_source):
    report = verify_template_source(invoice_source, root_class=invoice_spec.root)
    assert report.passed
    v6 = gate(report, "V6")
    assert v6.skipped and v6.passed
    assert "spec-level assertions skipped" in gate(report, "V4").detail


# ---------------------------------------------------------------------------
# V1 — structure
# ---------------------------------------------------------------------------


def test_v1_syntax_error_skips_everything():
    report = verify_template_source("def broken(:", root_class="Doc")
    assert not report.passed
    assert "SyntaxError" in gate(report, "V1").detail
    assert all(g.skipped for g in report.gates if g.gate != "V1")


def test_v1_fails_without_edge_function():
    source = 'from pydantic import BaseModel\n\n\nclass Doc(BaseModel):\n    """Root."""\n'
    report = verify_template_source(source, root_class="Doc")
    assert "exactly one top-level edge() definition" in gate(report, "V1").detail


def test_v1_fails_when_edge_defined_after_first_class():
    source = (
        "from typing import Any\n"
        "from pydantic import BaseModel, Field\n\n\n"
        "class Doc(BaseModel):\n"
        '    """Root."""\n\n\n'
        "def edge(label: str, **kwargs: Any) -> Any:\n"
        '    return Field(None, json_schema_extra={"edge_label": label}, **kwargs)\n'
    )
    report = verify_template_source(source, root_class="Doc")
    assert "edge() must be defined before the first class" in gate(report, "V1").detail


def test_v1_fails_when_root_is_not_last_class():
    source = (
        "from typing import Any\n"
        "from pydantic import BaseModel, Field\n\n\n"
        "def edge(label: str, **kwargs: Any) -> Any:\n"
        '    return Field(None, json_schema_extra={"edge_label": label}, **kwargs)\n\n\n'
        "class Doc(BaseModel):\n"
        '    """Root."""\n\n\n'
        "class Trailing(BaseModel):\n"
        '    """Not the root."""\n'
    )
    report = verify_template_source(source, root_class="Doc")
    assert "root class must be the last class definition" in gate(report, "V1").detail
    assert "'Trailing'" in gate(report, "V1").detail


# ---------------------------------------------------------------------------
# V1b — import allowlist (checked BEFORE any execution)
# ---------------------------------------------------------------------------


def test_v1b_bad_import_named_and_execution_refused(invoice_source):
    poisoned = invoice_source.replace("import logging\n", "import logging\nimport os\n", 1)
    report = verify_template_source(poisoned, root_class="Invoice")
    v1b = gate(report, "V1b")
    assert not v1b.passed
    assert "import os" in v1b.detail
    # V2-V6 must be skipped: the source is never executed.
    for gate_id in ("V2", "V3", "V4", "V5", "V6"):
        result = gate(report, gate_id)
        assert result.skipped and not result.passed
        assert "refusing to execute" in result.detail


def test_v1b_flags_forbidden_builtins(invoice_source):
    poisoned = invoice_source.replace(
        "logger = logging.getLogger(__name__)",
        'logger = logging.getLogger(__name__)\n_payload = eval("1 + 1")',
        1,
    )
    report = verify_template_source(poisoned, root_class="Invoice")
    assert "use of 'eval'" in gate(report, "V1b").detail


def test_v1b_flags_disallowed_from_import(invoice_source):
    poisoned = invoice_source.replace(
        "import logging\n", "import logging\nfrom pathlib import Path\n", 1
    )
    report = verify_template_source(poisoned, root_class="Invoice")
    assert "from pathlib import" in gate(report, "V1b").detail


# ---------------------------------------------------------------------------
# V2 — exec + loader predicate
# ---------------------------------------------------------------------------


def test_v2_fails_on_shadowed_root(invoice_source):
    shadowed = invoice_source + '\n\nInvoice = "shadowed"\n'
    report = verify_template_source(shadowed, root_class="Invoice")
    v2 = gate(report, "V2")
    assert not v2.passed
    assert "not a BaseModel subclass" in v2.detail
    assert all(gate(report, g).skipped for g in ("V3", "V4", "V5", "V6"))


def test_v2_fails_on_missing_root_class(invoice_source):
    report = verify_template_source(invoice_source, root_class="NoSuchRoot")
    # V1 also flags the last-class mismatch; V2 must fail independently.
    assert not gate(report, "V2").passed


def test_v2_fails_on_runtime_error(invoice_source):
    exploding = invoice_source + '\n\nraise RuntimeError("boom at import time")\n'
    report = verify_template_source(exploding, root_class="Invoice")
    v2 = gate(report, "V2")
    assert not v2.passed
    assert "RuntimeError" in v2.detail


# ---------------------------------------------------------------------------
# The renderer-can't-produce case: a required non-identity field passes V1-V5
# (the Optionality Law is enforced by construction in the renderer and by
# linter rule R7 — the verification gate deliberately does not re-check it).
# ---------------------------------------------------------------------------

REQUIRED_NON_IDENTITY_TEMPLATE = '''\
"""Hand-written template with a required non-identity field."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def edge(label: str, **kwargs: Any) -> Any:
    """Graph edge helper."""
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = None
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


class Doc(BaseModel):
    """Root document."""

    model_config = ConfigDict(graph_id_fields=["doc_id"])

    doc_id: str = Field(..., description="Identifier.", examples=["D-1"])
    amount: float = Field(..., description="Required non-identity field.")
'''


def test_required_non_identity_field_passes_v1_to_v5():
    report = verify_template_source(REQUIRED_NON_IDENTITY_TEMPLATE, root_class="Doc")
    for gate_id in ("V1", "V1b", "V2", "V3", "V4", "V5"):
        assert gate(report, gate_id).passed, report.summary()
    assert gate(report, "V6").skipped  # no spec supplied


# ---------------------------------------------------------------------------
# V4 — dense catalog assertions (real build_node_catalog)
# ---------------------------------------------------------------------------


def test_v4_fails_when_reference_marker_stripped(invoice_spec, invoice_source):
    tampered = invoice_source.replace("        reference=True,\n", "").replace(
        "        closed_catalog=True,\n", ""
    )
    report = verify_template_source(tampered, root_class="Invoice", spec=invoice_spec)
    v4 = gate(report, "V4")
    assert not v4.passed
    assert "reference field 'LineItem.item'" in v4.detail


def test_v4_fails_on_id_fields_mismatch(invoice_spec, invoice_source):
    tampered = invoice_source.replace(
        'ConfigDict(graph_id_fields=["line_number"])',
        'ConfigDict(graph_id_fields=["item_code"])',
    )
    report = verify_template_source(tampered, root_class="Invoice", spec=invoice_spec)
    v4 = gate(report, "V4")
    assert not v4.passed
    assert "id_fields" in v4.detail


def test_v4_full_edge_cycle_walks_safely():
    # A mutual full-edge cycle used to send build_node_catalog into unbounded
    # recursion; the walk now prunes recursive ancestry, so the cycle renders
    # AND verifies — each class keeps its non-recursive discovery path (the
    # recurrence itself is pruned, resolved by the converter at runtime).
    spec = TemplateSpec(
        root="Doc",
        models=[
            ModelSpec(
                name="A",
                kind="entity",
                docstring="Entity A.",
                identity_fields=["name"],
                fields=[
                    FieldSpec(name="name", type="str", role="identity", examples=["a"]),
                    FieldSpec(name="partner", type="B", role="edge", edge_label="PARTNERED_WITH"),
                ],
            ),
            ModelSpec(
                name="B",
                kind="entity",
                docstring="Entity B.",
                identity_fields=["name"],
                fields=[
                    FieldSpec(name="name", type="str", role="identity", examples=["b"]),
                    FieldSpec(name="partner", type="A", role="edge", edge_label="PARTNERED_WITH"),
                ],
            ),
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="Root document.",
                identity_fields=["title"],
                fields=[
                    FieldSpec(name="title", type="str", role="identity", examples=["t"]),
                    FieldSpec(name="a", type="A", role="edge", edge_label="HAS_A"),
                ],
            ),
        ],
    )
    report = verify_template_source(render_template(spec), root_class="Doc", spec=spec)
    v4 = gate(report, "V4")
    assert v4.passed, v4.detail
    assert "3 discovery path(s)" in v4.detail  # Doc, a, a.partner — recurrence pruned


# ---------------------------------------------------------------------------
# V6 — graph-shape smoke test (real GraphConverter)
# ---------------------------------------------------------------------------

EMPTY_IDENTITY_TEMPLATE = '''\
"""Hand-written template whose validator empties the identity field."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def edge(label: str, **kwargs: Any) -> Any:
    """Graph edge helper."""
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = None
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


class Doc(BaseModel):
    """Root document."""

    model_config = ConfigDict(graph_id_fields=["doc_id"])

    doc_id: Optional[str] = Field(None, description="Identifier.", examples=["D-1"])

    @field_validator("doc_id", mode="before")
    @classmethod
    def _clear_doc_id(cls, v: Any) -> Any:
        return None
'''


def empty_identity_spec() -> TemplateSpec:
    return TemplateSpec(
        root="Doc",
        models=[
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="Root document.",
                identity_fields=["doc_id"],
                fields=[
                    FieldSpec(
                        name="doc_id",
                        type="str",
                        role="identity",
                        description="Identifier.",
                        examples=["D-1"],
                    )
                ],
            )
        ],
    )


def test_v6_catches_empty_identity_sample():
    spec = empty_identity_spec()
    report = verify_template_source(EMPTY_IDENTITY_TEMPLATE, root_class="Doc", spec=spec)
    for gate_id in ("V1", "V1b", "V2", "V3", "V4", "V5"):
        assert gate(report, gate_id).passed, report.summary()
    v6 = gate(report, "V6")
    assert not v6.passed
    assert "empty identity" in v6.detail


def test_v6_detail_reports_graph_shape(invoice_spec, invoice_source):
    report = verify_template_source(invoice_source, root_class=invoice_spec.root, spec=invoice_spec)
    v6 = gate(report, "V6")
    assert "node(s)" in v6.detail
    assert "Invoice" in v6.detail


def sibling_closed_catalog_spec() -> TemplateSpec:
    """Two sibling closed-catalog reference edges to one target class.

    Regression (V6 phantom catalog member): the second sibling reference used
    to mint an occurrence-1 identity that existed nowhere as a full instance,
    so the closed-catalog enforcement dropped its edge and V6 failed on a
    perfectly valid spec.
    """
    return TemplateSpec(
        root="Doc",
        models=[
            ModelSpec(
                name="Item",
                kind="entity",
                docstring="A catalog item, identified by its printed name.",
                identity_fields=["name"],
                fields=[
                    FieldSpec(
                        name="name",
                        type="str",
                        role="identity",
                        description="Item name as printed.",
                        examples=["USB-C cable", "Dock DS-300"],
                    )
                ],
            ),
            ModelSpec(
                name="Section",
                kind="entity",
                docstring="A section of the document, identified by its title.",
                identity_fields=["title"],
                fields=[
                    FieldSpec(
                        name="title",
                        type="str",
                        role="identity",
                        description="Section title.",
                        examples=["Coverage", "Exclusions"],
                    ),
                    FieldSpec(
                        name="covered",
                        type="Item",
                        is_list=True,
                        role="edge",
                        edge_label="COVERS",
                        reference=True,
                        closed_catalog=True,
                        description="Catalog items covered by this section.",
                    ),
                    FieldSpec(
                        name="excluded",
                        type="Item",
                        is_list=True,
                        role="edge",
                        edge_label="EXCLUDES_ITEM",
                        reference=True,
                        closed_catalog=True,
                        description="Catalog items excluded by this section.",
                    ),
                ],
            ),
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="The document root.",
                identity_fields=["title"],
                fields=[
                    FieldSpec(
                        name="title",
                        type="str",
                        role="identity",
                        description="Document title.",
                        examples=["Policy 2026"],
                    ),
                    FieldSpec(
                        name="items",
                        type="Item",
                        is_list=True,
                        role="edge",
                        edge_label="CONTAINS_ITEM",
                        description="Canonical catalog of every item (full detail once).",
                    ),
                    FieldSpec(
                        name="section",
                        type="Section",
                        role="edge",
                        edge_label="HAS_SECTION",
                        description="The section referencing catalog items.",
                    ),
                ],
            ),
        ],
    )


def test_v6_passes_with_two_sibling_closed_catalog_references():
    spec = sibling_closed_catalog_spec()
    report = verify_template_source(render_template(spec), root_class=spec.root, spec=spec)
    assert report.passed, report.summary()
    v6 = gate(report, "V6")
    assert v6.passed and not v6.skipped


def test_sibling_references_reuse_the_canonical_occurrence():
    # Both sibling references must point at the catalog's occurrence-0
    # identity — never a phantom member that exists nowhere in full.
    plan = synthesize_sample_plan(sibling_closed_catalog_spec())
    canonical = plan.payload["items"][0]["name"]
    assert plan.payload["section"]["covered"] == [{"name": canonical}]
    assert plan.payload["section"]["excluded"] == [{"name": canonical}]


# ---------------------------------------------------------------------------
# Sample synthesis (shared with the renderer footer and the CLI trial-run)
# ---------------------------------------------------------------------------


def test_sample_plan_is_deterministic(invoice_spec):
    assert synthesize_sample_plan(invoice_spec) == synthesize_sample_plan(invoice_spec)


def test_sample_plan_identity_from_first_example(invoice_spec):
    plan = synthesize_sample_plan(invoice_spec)
    assert plan.payload["document_number"] == "INV-2024-0113"


def test_sample_plan_disambiguates_same_class_siblings(invoice_spec):
    # seller/buyer both target Party: identical identities would merge into one
    # node and collapse the two Invoice->Party edges into a single DiGraph edge.
    plan = synthesize_sample_plan(invoice_spec)
    assert plan.payload["seller"]["name"] != plan.payload["buyer"]["name"]


def test_sample_plan_reference_matches_canonical_identity(invoice_spec):
    plan = synthesize_sample_plan(invoice_spec)
    reference = plan.payload["line_items"][0]["item"]
    canonical = plan.payload["items"][0]
    assert reference == {"name": canonical["name"]}  # identity-only projection


def test_sample_plan_excludes_component_edge_labels(invoice_spec):
    plan = synthesize_sample_plan(invoice_spec)
    assert "HAS_TAX" not in plan.edge_labels  # Tax is a component: no graph edge
    assert {"ISSUED_BY", "BILLED_TO", "CONTAINS_ITEM", "CONTAINS_LINE", "REFERENCES_ITEM"} <= set(
        plan.edge_labels
    )
    assert plan.entity_classes == ["Invoice", "Party", "Item", "LineItem"]


def test_sample_plan_placeholders_for_gap_specs():
    plan = synthesize_sample_plan(load_spec("policy_ontology"))
    assert plan.payload["policy_number"] == "sample-policy_number"
    assert plan.payload["covers"] == [{"guarantee_name": "sample-guarantee_name"}]


def test_synthesize_sample_builds_live_instance(invoice_spec, invoice_source):
    namespace: dict[str, object] = {"__name__": "test_template_ns"}
    exec(compile(invoice_source, "<test>", "exec"), namespace)
    sample = synthesize_sample(invoice_spec, namespace)
    assert isinstance(sample, BaseModel)
    assert type(sample).__name__ == "Invoice"


def test_synthesize_sample_rejects_missing_root(invoice_spec):
    with pytest.raises(TypeError, match="BaseModel subclass"):
        synthesize_sample(invoice_spec, {})


# ---------------------------------------------------------------------------
# End to end: draft spec -> lint_spec -> render_template -> verify all-green
# ---------------------------------------------------------------------------


def test_end_to_end_draft_lint_render_verify():
    from docling_graph.templategen.linter import lint_spec

    draft = TemplateSpec(
        root="Report",
        models=[
            ModelSpec(
                name="Author",
                kind="entity",
                docstring="A named author of the report. NOT an organization.",
                identity_fields=["full_name"],
                fields=[
                    FieldSpec(
                        name="full_name",
                        type="str",
                        role="identity",
                        description="Author name as printed on the cover.",
                        examples=["Jane Doe", "John Smith"],
                    ),
                    # R21: 'label' collides with the reserved node-attr keys the
                    # GraphConverter writes — the linter must rename it.
                    FieldSpec(
                        name="label",
                        type="str",
                        description="Role label printed next to the name.",
                        examples=["Lead"],
                    ),
                ],
            ),
            ModelSpec(
                name="Report",
                kind="root",
                docstring="The report document, identified by its printed number.",
                identity_fields=["report_number"],
                fields=[
                    FieldSpec(
                        name="report_number",
                        type="str",
                        role="identity",
                        description="Number printed on the cover.",
                        examples=["R-001", "R-002"],
                    ),
                    # R9: 'HAS' is a banned vague edge label — must be rewritten.
                    FieldSpec(
                        name="authors",
                        type="Author",
                        is_list=True,
                        role="edge",
                        edge_label="HAS",
                        description="Authors listed on the cover.",
                    ),
                ],
            ),
        ],
    )
    repaired, lint_report = lint_spec(draft)
    assert any(entry.repaired for entry in lint_report.entries)

    source = render_template(repaired)
    report = verify_template_source(source, root_class=repaired.root, spec=repaired)
    assert report.passed, report.summary()
    assert not any(g.skipped for g in report.gates)


@pytest.mark.parametrize("shadow_name", ["date", "datetime", "edge", "logger"])
def test_field_shadowing_template_module_names_renders_and_loads(shadow_name):
    """Fields named date/datetime/edge/logger must not break V2.

    Without the rename, `date`/`datetime` rebind the annotation name in the
    class body (no `from __future__ import annotations`), `edge` shadows the
    helper for subsequent edge fields, and `logger` the module logger.
    """
    from docling_graph.templategen.linter import lint_spec

    draft = TemplateSpec(
        root="Doc",
        models=[
            ModelSpec(
                name="Party",
                kind="entity",
                docstring="A party named in the document. NOT the document itself.",
                identity_fields=["name"],
                fields=[
                    FieldSpec(
                        name="name",
                        type="str",
                        role="identity",
                        description="Name as printed.",
                        examples=["Acme Corp", "Beta SARL"],
                    )
                ],
            ),
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="The document, identified by its printed number.",
                identity_fields=["number"],
                fields=[
                    FieldSpec(
                        name="number",
                        type="str",
                        role="identity",
                        description="Number printed in the header.",
                        examples=["D-001", "D-002"],
                    ),
                    # the shadowing field, followed by fields that would break
                    # if the module-level name had been rebound above them
                    FieldSpec(
                        name=shadow_name,
                        type="str",
                        description="Field whose raw name shadows a module-level name.",
                    ),
                    FieldSpec(
                        name="issue_date",
                        type="date",
                        description="Issue date printed in the header.",
                        examples=["2026-01-15"],
                    ),
                    FieldSpec(
                        name="issued_at",
                        type="datetime",
                        description="Issue timestamp when printed.",
                        examples=["2026-01-15T10:00:00"],
                    ),
                    FieldSpec(
                        name="party",
                        type="Party",
                        role="edge",
                        edge_label="ISSUED_BY",
                        description="The issuing party.",
                    ),
                ],
            ),
        ],
    )
    repaired, lint_report = lint_spec(draft)
    root = next(m for m in repaired.models if m.kind == "root")
    field_names = {f.name for f in root.fields}
    assert f"{shadow_name}_field" in field_names
    assert shadow_name not in field_names
    assert any(e.rule_id == "R20" and e.repaired for e in lint_report.entries)

    source = render_template(repaired)
    report = verify_template_source(source, root_class=repaired.root, spec=repaired)
    assert report.passed, report.summary()


def test_property_field_cycle_lints_renders_and_verifies():
    """A cycle through model-typed PROPERTY fields gets the same back-edge
    treatment as an edge cycle (R15) and then passes V1-V6 — previously it
    slipped through lint and blew up V4 with a RecursionError."""
    from docling_graph.templategen.linter import lint_spec

    def entity(name: str, partner: str, examples: list[str]) -> ModelSpec:
        return ModelSpec(
            name=name,
            kind="entity",
            docstring=f"Entity {name} of the partnership pair.",
            identity_fields=["name"],
            fields=[
                FieldSpec(
                    name="name",
                    type="str",
                    role="identity",
                    description="Name as printed.",
                    examples=examples,
                ),
                # model-typed PROPERTY field (no edge metadata)
                FieldSpec(name="partner", type=partner, description="Partner record."),
            ],
        )

    spec = TemplateSpec(
        root="Doc",
        models=[
            entity("Alpha", "Beta", ["A-100", "A-200"]),
            entity("Beta", "Alpha", ["B-100", "B-200"]),
            ModelSpec(
                name="Doc",
                kind="root",
                docstring="Root document.",
                identity_fields=["title"],
                fields=[
                    FieldSpec(
                        name="title",
                        type="str",
                        role="identity",
                        description="Title.",
                        examples=["Partnerships 2026"],
                    ),
                    FieldSpec(
                        name="alpha",
                        type="Alpha",
                        role="edge",
                        edge_label="HAS_ALPHA",
                        description="The alpha record.",
                    ),
                ],
            ),
        ],
    )
    repaired, lint_report = lint_spec(spec)
    back = next(
        f for m in repaired.models if m.name == "Beta" for f in m.fields if f.name == "partner"
    )
    assert back.role == "edge" and back.reference is True
    assert any(e.rule_id == "R15" and e.repaired for e in lint_report.entries)

    source = render_template(repaired)
    report = verify_template_source(source, root_class=repaired.root, spec=repaired)
    assert report.passed, report.summary()
    assert not any(g.skipped for g in report.gates)


def test_v3_recursive_root_via_inverse_reference_passes():
    """An inverse reference edge back to the root (Author AUTHORED_BY-> Paper)
    hoists the root into $defs in pydantic's schema; V3 must still normalize
    to a strict top-level object (schema_utils dereferences the root $ref)."""
    spec = TemplateSpec(
        root="Paper",
        models=[
            ModelSpec(
                name="Author",
                kind="entity",
                docstring="An author of the paper.",
                identity_fields=["name"],
                fields=[
                    FieldSpec(name="name", type="str", role="identity", examples=["Ada B"]),
                    FieldSpec(
                        name="authored",
                        type="Paper",
                        role="edge",
                        edge_label="AUTHORED_BY",
                        reference=True,
                    ),
                ],
            ),
            ModelSpec(
                name="Paper",
                kind="root",
                docstring="The paper itself.",
                identity_fields=["title"],
                fields=[
                    FieldSpec(name="title", type="str", role="identity", examples=["Rheology"]),
                    FieldSpec(
                        name="authors",
                        type="Author",
                        role="edge",
                        edge_label="AUTHORED_BY",
                        is_list=True,
                    ),
                ],
            ),
        ],
    )
    report = verify_template_source(render_template(spec), root_class="Paper", spec=spec)
    v3 = gate(report, "V3")
    assert v3.passed, v3.detail
    assert report.passed, report.summary()
