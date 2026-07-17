"""Golden-file and property tests for the deterministic template renderer."""

import ast
import difflib
import os
from pathlib import Path

import pytest

from docling_graph.templategen.renderer import render_template
from docling_graph.templategen.spec import ModelSpec, TemplateSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "templategen"
SPEC_DIR = FIXTURES / "specs"
GOLDEN_DIR = FIXTURES / "golden"

FIXTURE_NAMES = sorted(path.stem for path in SPEC_DIR.glob("*.yaml"))


def load_spec(name: str) -> TemplateSpec:
    return TemplateSpec.from_yaml((SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered_by_name() -> dict[str, str]:
    return {name: render_template(load_spec(name)) for name in FIXTURE_NAMES}


def test_fixture_inventory():
    # The suite expects at least the three canonical shapes: a validator-heavy
    # invoice, an ontology spec with gaps, and a cyclic/forward-ref spec.
    assert {"invoice", "policy_ontology", "org_chart"} <= set(FIXTURE_NAMES)


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_matches_golden_file(name, rendered_by_name):
    rendered = rendered_by_name[name]
    golden_path = GOLDEN_DIR / f"{name}.py"
    if os.environ.get("REGENERATE_GOLDENS"):
        golden_path.write_text(rendered, encoding="utf-8")
    golden = golden_path.read_text(encoding="utf-8")
    if rendered != golden:
        diff = "\n".join(
            difflib.unified_diff(
                golden.splitlines(),
                rendered.splitlines(),
                fromfile=f"golden/{name}.py",
                tofile="rendered",
                lineterm="",
            )
        )
        pytest.fail(
            f"Rendered output drifted from tests/fixtures/templategen/golden/{name}.py.\n"
            "If the change is intentional, regenerate via:\n"
            "    REGENERATE_GOLDENS=1 .venv/bin/python -m pytest "
            "tests/unit/templategen/test_renderer.py\n\n" + diff
        )


# ---------------------------------------------------------------------------
# Table-driven property asserts over ALL fixture specs
# ---------------------------------------------------------------------------


def _model_class_defs(tree: ast.Module, spec: TemplateSpec) -> list[ast.ClassDef]:
    model_names = {model.name for model in spec.models}
    return [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in model_names
    ]


def _field_calls(class_def: ast.ClassDef) -> dict[str, tuple[ast.expr, ast.Call]]:
    """{field name: (annotation, call)} for the class's AnnAssign field lines."""
    fields: dict[str, tuple[ast.expr, ast.Call]] = {}
    for stmt in class_def.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if stmt.target.id == "model_config" or not isinstance(stmt.value, ast.Call):
            continue
        fields[stmt.target.id] = (stmt.annotation, stmt.value)
    return fields


def _is_required(call: ast.Call) -> bool:
    return bool(
        call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value is Ellipsis
    )


def _func_name(call: ast.Call) -> str:
    assert isinstance(call.func, ast.Name)
    return call.func.id


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_output_compiles(name, rendered_by_name):
    compile(rendered_by_name[name], f"<{name}>", "exec")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_root_class_is_last_class_def(name, rendered_by_name):
    spec = load_spec(name)
    tree = ast.parse(rendered_by_name[name])
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert class_defs and class_defs[-1].name == spec.root


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_single_edge_def_before_first_class(name, rendered_by_name):
    tree = ast.parse(rendered_by_name[name])
    edge_indices = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "edge"
    ]
    first_class = min(
        index for index, node in enumerate(tree.body) if isinstance(node, ast.ClassDef)
    )
    assert len(edge_indices) == 1
    assert edge_indices[0] < first_class


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_non_identity_field_is_optional_or_defaulted(name, rendered_by_name):
    spec = load_spec(name)
    tree = ast.parse(rendered_by_name[name])
    models: dict[str, ModelSpec] = {model.name: model for model in spec.models}
    for class_def in _model_class_defs(tree, spec):
        identity = set(models[class_def.name].identity_fields)
        for field_name, (_annotation, call) in _field_calls(class_def).items():
            if field_name in identity:
                assert _is_required(call), f"{class_def.name}.{field_name} must be required"
                continue
            assert not _is_required(call), (
                f"{class_def.name}.{field_name} is non-identity but required "
                "(violates the Optionality Law)"
            )
            has_default_factory = any(kw.arg == "default_factory" for kw in call.keywords)
            has_positional_default = bool(call.args)
            # edge() defaults single edges to None internally; Field() fields
            # must carry an explicit default or default_factory.
            if _func_name(call) == "Field":
                assert has_default_factory or has_positional_default, (
                    f"{class_def.name}.{field_name} has no default"
                )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_list_edge_has_default_factory(name, rendered_by_name):
    spec = load_spec(name)
    tree = ast.parse(rendered_by_name[name])
    for class_def in _model_class_defs(tree, spec):
        for field_name, (annotation, call) in _field_calls(class_def).items():
            if _func_name(call) != "edge":
                continue
            is_list = (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id == "List"
            )
            if not is_list:
                continue
            factory = next((kw.value for kw in call.keywords if kw.arg == "default_factory"), None)
            assert isinstance(factory, ast.Name) and factory.id == "list", (
                f"list edge {class_def.name}.{field_name} lacks default_factory=list"
            )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_no_raise_inside_mode_before_validators(name, rendered_by_name):
    tree = ast.parse(rendered_by_name[name])
    before_validators = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "field_validator"
                and any(
                    kw.arg == "mode"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "before"
                    for kw in decorator.keywords
                )
            ):
                before_validators.append(node)
    for validator in before_validators:
        raises = [n for n in ast.walk(validator) if isinstance(n, ast.Raise)]
        assert not raises, f"mode='before' validator {validator.name} raises"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_enum_field_has_a_normalize_validator(name, rendered_by_name):
    # Scalar AND list enum fields both get the mode='before' normalizer —
    # a List[Enum] field without one rejects case/synonym variants,
    # violating the never-reject law the renderer claims by construction.
    spec = load_spec(name)
    enum_names = {enum.name for enum in spec.enums}
    tree = ast.parse(rendered_by_name[name])
    for class_def in _model_class_defs(tree, spec):
        model = next(model for model in spec.models if model.name == class_def.name)
        methods = {node.name for node in class_def.body if isinstance(node, ast.FunctionDef)}
        for field in model.fields:
            if field.type in enum_names:
                assert f"_normalize_{field.name}" in methods, (
                    f"{name}: enum field {class_def.name}.{field.name} "
                    f"(is_list={field.is_list}) lacks a normalize validator"
                )


def test_invoice_spec_has_a_list_enum_field_to_exercise():
    # Guard the guard: the fixture set must include a List[Enum] field.
    spec = load_spec("invoice")
    enum_names = {enum.name for enum in spec.enums}
    assert any(
        field.is_list and field.type in enum_names
        for model in spec.models
        for field in model.fields
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_render_is_byte_deterministic(name, rendered_by_name):
    assert render_template(load_spec(name)) == rendered_by_name[name]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_evidence_is_dropped_at_render_time(name, rendered_by_name):
    spec = load_spec(name)
    rendered = rendered_by_name[name]
    for model in spec.models:
        for field in model.fields:
            for quote in field.evidence:
                assert quote not in rendered
    assert "EVIDENCE-SENTINEL" not in rendered


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_verification_footer_present(name, rendered_by_name):
    rendered = rendered_by_name[name]
    assert "# Verification" in rendered
    # Short informational footer only: comments never reach extraction
    # prompts, and the V6 sample detail lives in the generation report.
    assert "docling-graph template lint" in rendered
    assert "never reach extraction prompts" in rendered


def test_invoice_spec_has_evidence_to_drop():
    # Guard the guard: the evidence-dropping assert must actually exercise data.
    spec = load_spec("invoice")
    assert any(field.evidence for model in spec.models for field in model.fields)


# ---------------------------------------------------------------------------
# Targeted shape asserts (documenting specific emission rules)
# ---------------------------------------------------------------------------


def test_self_imported_from_typing_extensions_only_when_dedup_used(rendered_by_name):
    # Repo floor is Python 3.10; typing.Self needs 3.11+. The dedup snippet is
    # the only emitted code returning Self, so the import appears exactly when
    # needs_root_list_dedup is non-empty — and never from `typing`.
    invoice = rendered_by_name["invoice"]
    assert "from typing_extensions import Self" in invoice
    for name, rendered in rendered_by_name.items():
        for line in rendered.splitlines():
            if line.startswith("from typing import"):
                assert "Self" not in line, f"{name}: Self leaked into the typing import"
        if not load_spec(name).needs_root_list_dedup:
            assert "typing_extensions" not in rendered


def test_optional_imports_tracked_by_usage(rendered_by_name):
    policy = rendered_by_name["policy_ontology"]
    assert "import logging" not in policy
    assert "import re" not in policy
    assert "from enum import Enum" not in policy
    assert "from datetime import date, datetime" in policy  # effective_date: date
    assert "field_validator" not in policy  # no validators emitted at all
    invoice = rendered_by_name["invoice"]
    assert "import logging" in invoice
    assert "logger = logging.getLogger(__name__)" in invoice


def test_gap_fields_render_runnable_todos(rendered_by_name):
    policy = rendered_by_name["policy_ontology"]
    assert "examples=[],  # TODO(docling-graph): add 2-5 verbatim examples" in policy
    assert 'description="",  # TODO(docling-graph):' in policy


def test_cycles_use_quoted_forward_refs_and_model_rebuild(rendered_by_name):
    org_chart = rendered_by_name["org_chart"]
    assert 'subsections: List["Section"]' in org_chart
    assert 'works_for: Optional["Organization"]' in org_chart
    rebuild_pos = org_chart.index("Section.model_rebuild()")
    assert "Person.model_rebuild()" in org_chart
    assert rebuild_pos > org_chart.index("class OrgChart(BaseModel):")


def test_enum_members_and_other_safety_net(rendered_by_name):
    invoice = rendered_by_name["invoice"]
    assert 'CREDIT_NOTE = "Credit Note"' in invoice
    assert 'OTHER = "Other"' in invoice
    assert "DocumentType.OTHER," in invoice  # defaulted enum field
    assert "_normalize_document_type" in invoice
    # Synonyms live in the field description, not the enum class.
    assert "'Facture' -> 'Invoice'" in invoice


def test_enum_synonyms_not_rendered_into_enum_class(rendered_by_name):
    invoice = rendered_by_name["invoice"]
    tree = ast.parse(invoice)
    enum_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DocumentType"
    )
    enum_source = ast.get_source_segment(invoice, enum_class)
    assert enum_source is not None
    assert "Facture" not in enum_source


def test_max_instances_rendered_as_graph_max_instances(rendered_by_name):
    policy = rendered_by_name["policy_ontology"]
    assert 'ConfigDict(graph_id_fields=["guarantee_name"], graph_max_instances=12)' in policy


def test_reference_and_closed_catalog_passed_through(rendered_by_name):
    invoice = rendered_by_name["invoice"]
    assert "reference=True," in invoice
    assert "closed_catalog=True," in invoice


def test_str_method_on_every_model(rendered_by_name):
    for name in FIXTURE_NAMES:
        spec = load_spec(name)
        tree = ast.parse(rendered_by_name[name])
        for class_def in _model_class_defs(tree, spec):
            methods = {node.name for node in class_def.body if isinstance(node, ast.FunctionDef)}
            assert "__str__" in methods, f"{name}: {class_def.name} lacks __str__"


def test_provenance_line_uses_only_generator_content(rendered_by_name):
    invoice = " ".join(rendered_by_name["invoice"].split())  # wrap-tolerant
    assert "Generated by docling-graph template generation v0.1.0" in invoice
    assert "invoice1.pdf, invoice2.pdf" in invoice
    assert "on 2026-07-16" in invoice
    # org_chart has an empty generator dict: the line stays bare — nothing invented.
    org_chart = rendered_by_name["org_chart"]
    assert "Generated by docling-graph template generation." in org_chart


def test_provenance_paths_never_wrap_mid_token():
    # A long hyphenated source path must survive on ONE line — the default
    # textwrap hyphen/word breaking would corrupt copy-pasted paths.
    path = (
        "/data/very-long-hyphenated-directory-name/another-long-segment/"
        "invoice-documents-2026-07-16-batch-0001-final-revision.pdf"
    )
    spec = load_spec("org_chart").model_copy(deep=True)
    spec.generator = {"version": "0.1.0", "sources": [path]}
    rendered = render_template(spec)
    assert any(path in line for line in rendered.splitlines()), (
        "provenance path was wrapped mid-token"
    )
    compile(rendered, "<provenance>", "exec")


def test_backslashes_in_docstrings_render_parseable_source():
    # '\U' / '\x' sequences in module or class docstrings are invalid string
    # escapes unless the emitted source doubles the backslashes.
    spec = load_spec("org_chart").model_copy(deep=True)
    spec.module_docstring = r"Matches files under \Users\xavier on the D:\data share."
    target = next(model for model in spec.models if model.name == "Person")
    target.docstring = (
        r"A person record sourced from \Users\xavier\exports. NOT an organization. "
        + "Long enough to force the wrapped multi-line docstring emission path "
        + "so both class-docstring branches stay covered by this regression test."
    )
    rendered = render_template(spec)
    tree = ast.parse(rendered)  # would raise SyntaxError without escaping
    module_doc = ast.get_docstring(tree)
    assert module_doc is not None and "\\Users\\xavier" in module_doc
    person = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Person"
    )
    class_doc = ast.get_docstring(person)
    assert class_doc is not None and "\\Users\\xavier\\exports" in class_doc
