"""
Verification gate (V1-V6) for rendered template modules.

Every generator command runs this gate between rendering and writing the file.
"Verified" here means *the pipeline demonstrably accepts and correctly graphs
the template*, not "we followed the docs": each gate exercises the actual
runtime surface the template must survive.

- **V1**  ``ast.parse`` + structure: exactly one ``edge`` function before the
  first class, root class last.
- **V1b** AST import allowlist (typing/typing_extensions/pydantic/datetime/
  enum/re/logging) and a ban on ``exec``/``eval``/``open``/``__import__``/
  ``compile`` — checked *before* the source is ever executed.
- **V2**  ``exec`` in a fresh namespace + the identical
  ``isinstance(obj, type) and issubclass(obj, BaseModel)`` predicate
  ``TemplateLoadingStage._load_from_string`` applies (pipeline/stages.py).
- **V3**  ``Root.model_json_schema()`` survives
  ``normalize_schema_for_response_format`` exactly as
  ``LiteLLMClient._build_request`` calls it (schema transported as JSON text,
  default ``top_level``/``name``) — the direct contract accepts the template.
- **V4**  ``build_node_catalog(Root)`` — the dense contract can walk it; with a
  SPEC provided, every entity appears with the expected ``id_fields`` and every
  ``reference=True`` field is projected out of the discovery paths.
- **V5**  ``build_compact_semantic_guide`` is non-empty; kept in the report as
  the "what the LLM sees" preview.
- **V6**  A sample instance synthesized from the SPEC round-trips through the
  real ``GraphConverter().pydantic_list_to_graph`` — the best-practices.md
  human verification ritual, automated.

V7 (``--trial-run``, a real extraction) belongs to the CLI layer, not here.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from datetime import date, datetime
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.extractors.contracts.dense.catalog import (
    NodeCatalog,
    build_node_catalog,
)
from docling_graph.llm_clients.schema_utils import (
    build_compact_semantic_guide,
    normalize_schema_for_response_format,
)

from .spec import SCALAR_TYPES, EnumSpec, FieldSpec, ModelSpec, TemplateSpec

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"typing", "typing_extensions", "pydantic", "datetime", "enum", "re", "logging"}
)
"""Module roots a generated template may import (verification gate V1b)."""

FORBIDDEN_NAMES: frozenset[str] = frozenset({"exec", "eval", "open", "__import__", "compile"})
"""Builtins whose mere mention fails V1b — templates are declarative modules."""

_EXEC_MODULE_NAME = "docling_graph_generated_template"

_GATE_ORDER: tuple[tuple[str, str], ...] = (
    ("V1", "ast parse + file structure"),
    ("V1b", "import allowlist"),
    ("V2", "exec + BaseModel loader predicate"),
    ("V3", "response_format schema normalization"),
    ("V4", "dense node catalog walk"),
    ("V5", "compact semantic guide"),
    ("V6", "graph-shape smoke test"),
)


class GateResult(BaseModel):
    """Outcome of a single verification gate."""

    model_config = ConfigDict(extra="forbid")

    gate: str
    name: str
    passed: bool
    skipped: bool = False
    detail: str = ""


class VerificationReport(BaseModel):
    """Per-gate results plus the V5 semantic-guide preview."""

    model_config = ConfigDict(extra="forbid")

    root_class: str
    gates: list[GateResult] = Field(default_factory=list)
    semantic_guide_preview: str = ""

    @property
    def passed(self) -> bool:
        """True when every gate passed (skipped-by-design gates count as passed)."""
        return all(gate.passed for gate in self.gates)

    def failures(self) -> list[GateResult]:
        """Gates that did not pass, in run order."""
        return [gate for gate in self.gates if not gate.passed]

    def summary(self) -> str:
        """Printable one-line-per-gate summary (for the CLI)."""
        lines = []
        for gate in self.gates:
            status = "SKIP" if gate.skipped else ("PASS" if gate.passed else "FAIL")
            line = f"{gate.gate:<4} {status:<4} {gate.name}"
            if gate.detail:
                line += f" — {gate.detail}"
            lines.append(line)
        return "\n".join(lines)


class SamplePlan(BaseModel):
    """Deterministic synthesis plan for the V6 graph-shape smoke test.

    ``payload`` feeds ``Root.model_validate``; ``entity_classes`` and
    ``edge_labels`` are the graph shapes V6 asserts afterwards (component
    targets are excluded — the converter embeds them inline, so their edge
    labels never appear in the graph).
    """

    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]
    entity_classes: list[str]
    edge_labels: list[str]


# ---------------------------------------------------------------------------
# Sample synthesis (shared by V6, the renderer's # Verification footer, and
# the CLI's --trial-run preamble)
# ---------------------------------------------------------------------------


def _placeholder_value(field: FieldSpec) -> Any:
    """Deterministic per-type placeholder when a field carries no usable example."""
    placeholders: dict[str, Any] = {
        "int": 1,
        "float": 1.0,
        "bool": True,
        "date": "2024-01-01",
        "datetime": "2024-01-01T00:00:00",
    }
    if field.type == "str":
        return f"sample-{field.name}"
    return placeholders.get(field.type, f"sample-{field.name}")


def _usable_example_values(field: FieldSpec) -> list[Any]:
    """Examples that coerce to the field's scalar type, in SPEC order."""
    values: list[Any] = []
    for example in field.examples:
        if not example.strip():
            continue
        try:
            if field.type == "str":
                values.append(example)
            elif field.type == "int":
                values.append(int(example))
            elif field.type == "float":
                values.append(float(example))
            elif field.type == "bool":
                lowered = example.strip().lower()
                if lowered in ("true", "yes", "1"):
                    values.append(True)
                elif lowered in ("false", "no", "0"):
                    values.append(False)
            elif field.type == "date":
                date.fromisoformat(example.strip())
                values.append(example.strip())
            elif field.type == "datetime":
                datetime.fromisoformat(example.strip())
                values.append(example.strip())
        except ValueError:
            continue
    return values


def _example_value(field: FieldSpec) -> Any | None:
    """First example that coerces to the field's scalar type, or None."""
    values = _usable_example_values(field)
    return values[0] if values else None


def _identity_value(field: FieldSpec, occurrence: int) -> Any:
    """Identity for the Nth same-class instance under one parent.

    Distinct occurrences get distinct identities (later examples, then suffixed
    placeholders): two single edges from one parent to the same class (e.g.
    seller/buyer -> Party) must not merge into one node, or their two edges
    would collapse into a single nx.DiGraph edge and lose a label.
    """
    values = _usable_example_values(field)
    if occurrence < len(values):
        return values[occurrence]
    value = _placeholder_value(field)
    if occurrence == 0 or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return f"{value}-{occurrence + 1}"
    if isinstance(value, int | float):
        return value + occurrence
    return value


def synthesize_sample_plan(spec: TemplateSpec) -> SamplePlan:
    """Synthesize one deterministic sample payload for the root model.

    Identity values come from the first usable example (or a typed
    placeholder), every edge gets one nested instance — identity-only, always
    reusing the canonical home's occurrence-0 identity, when the edge is
    ``reference=True`` — and cycles terminate by omitting the re-entrant
    field. The same SPEC always yields the same plan.
    """
    models: dict[str, ModelSpec] = {m.name: m for m in spec.models}
    enums: dict[str, EnumSpec] = {e.name: e for e in spec.enums}
    root = models[spec.root]

    entity_classes: list[str] = []
    edge_labels: list[str] = []

    def note_class(model: ModelSpec) -> None:
        if model.kind != "component" and model.name not in entity_classes:
            entity_classes.append(model.name)

    def identity_payload(model: ModelSpec, occurrence: int) -> dict[str, Any]:
        note_class(model)
        fields = {f.name: f for f in model.fields}
        return {name: _identity_value(fields[name], occurrence) for name in model.identity_fields}

    def full_payload(model: ModelSpec, path: tuple[str, ...], occurrence: int) -> dict[str, Any]:
        note_class(model)
        payload: dict[str, Any] = (
            identity_payload(model, occurrence) if model.identity_fields else {}
        )
        # Per-parent occurrence counter for FULL nestings only: sibling full
        # fields targeting the same class get distinct identities (see
        # _identity_value). Reference fields always reuse the occurrence-0
        # identity — the one the canonical home instantiates — so a second
        # sibling reference never mints a phantom member that a closed
        # catalog would drop.
        occurrences: dict[str, int] = {}
        # Surviving edge label per (target class, occurrence): edges between
        # one node pair merge in nx.DiGraph (last attribute write wins), so
        # only the final label written for a pair is asserted by V6.
        claimed: dict[tuple[str, int], str] = {}
        for field in model.fields:
            if field.role == "identity":
                continue
            if field.type in SCALAR_TYPES:
                value = _example_value(field)
                if value is None:
                    continue
                payload[field.name] = [value] if field.is_list else value
            elif field.type in enums:
                enum_spec = enums[field.type]
                member = field.examples[0] if field.examples else enum_spec.members[0]
                payload[field.name] = [member] if field.is_list else member
            else:
                target = models[field.type]
                if target.name in path:
                    continue  # cycle: the field's default (None / []) terminates it
                if field.reference and target.identity_fields:
                    nth = 0
                    nested: dict[str, Any] = identity_payload(target, 0)
                else:
                    nth = occurrences.get(target.name, 0)
                    occurrences[target.name] = nth + 1
                    nested = full_payload(target, (*path, target.name), nth)
                payload[field.name] = [nested] if field.is_list else nested
                if field.role == "edge" and field.edge_label and target.kind != "component":
                    claimed[(target.name, nth)] = field.edge_label
        for label in claimed.values():
            if label not in edge_labels:
                edge_labels.append(label)
        return payload

    payload = full_payload(root, (root.name,), 0)
    return SamplePlan(payload=payload, entity_classes=entity_classes, edge_labels=edge_labels)


def synthesize_sample(spec: TemplateSpec, namespace: Mapping[str, Any]) -> BaseModel:
    """Instantiate the synthesized sample against the template's live root class.

    ``namespace`` is the fresh module namespace produced by executing the
    rendered source (verification gate V2) — or any mapping that exposes the
    root class under ``spec.root``.
    """
    root_cls = namespace.get(spec.root)
    if not (isinstance(root_cls, type) and issubclass(root_cls, BaseModel)):
        raise TypeError(f"namespace does not define a BaseModel subclass named {spec.root!r}")
    plan = synthesize_sample_plan(spec)
    return root_cls.model_validate(plan.payload)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _check_v1_structure(tree: ast.Module, root_class: str) -> list[str]:
    """Structural assertions on the parsed module (gate V1)."""
    problems: list[str] = []
    edge_indices = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "edge"
    ]
    class_indices = [
        index for index, node in enumerate(tree.body) if isinstance(node, ast.ClassDef)
    ]
    if len(edge_indices) != 1:
        problems.append(
            f"expected exactly one top-level edge() definition, found {len(edge_indices)}"
        )
    if not class_indices:
        problems.append("no class definitions found")
    if edge_indices and class_indices and edge_indices[0] > class_indices[0]:
        problems.append("edge() must be defined before the first class")
    if class_indices:
        last_class = tree.body[class_indices[-1]]
        assert isinstance(last_class, ast.ClassDef)
        if last_class.name != root_class:
            problems.append(
                f"root class must be the last class definition, found '{last_class.name}'"
            )
    return problems


def _check_v1b_imports(tree: ast.Module) -> list[str]:
    """Import allowlist + forbidden-builtin scan (gate V1b)."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"line {node.lineno}: relative import")
                continue
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                violations.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"line {node.lineno}: use of {node.id!r}")
    return violations


def _check_v4_catalog(
    catalog: NodeCatalog, root_class: str, spec: TemplateSpec | None
) -> tuple[list[str], str]:
    """Dense-catalog assertions (gate V4). Returns (problems, success detail)."""
    if not catalog.nodes:
        return (["catalog is empty (root not walkable)"], "")
    if catalog.nodes[0].node_type != root_class:
        return ([f"catalog root is '{catalog.nodes[0].node_type}', expected '{root_class}'"], "")
    if spec is None:
        return ([], f"{len(catalog.nodes)} discovery path(s); spec-level assertions skipped")

    problems: list[str] = []
    path_to_type = {node.path: node.node_type for node in catalog.nodes}
    for model in spec.models:
        matches = [node for node in catalog.nodes if node.node_type == model.name]
        if model.kind == "component":
            if matches:
                problems.append(f"component '{model.name}' appears as a discovery path")
            continue
        if not matches:
            problems.append(f"entity '{model.name}' has no discovery path")
            continue
        for node in matches:
            if node.id_fields != model.identity_fields:
                problems.append(
                    f"entity '{model.name}' at path '{node.path}' has id_fields "
                    f"{node.id_fields}, expected {model.identity_fields}"
                )
    for model in spec.models:
        for field in model.fields:
            if field.role != "edge" or not field.reference:
                continue
            for node in catalog.nodes:
                if (
                    node.field_name == field.name
                    and path_to_type.get(node.parent_path) == model.name
                ):
                    problems.append(
                        f"reference field '{model.name}.{field.name}' was discovered at "
                        f"path '{node.path}' (should be projected id-only into the parent)"
                    )
    detail = f"{len(catalog.nodes)} discovery path(s); entities + references verified"
    return (problems, detail)


def _run_v6(spec: TemplateSpec, namespace: Mapping[str, Any]) -> tuple[list[str], str]:
    """Graph-shape smoke test through the real converter (gate V6)."""
    plan = synthesize_sample_plan(spec)
    sample = synthesize_sample(spec, namespace)
    graph, _metadata = GraphConverter().pydantic_list_to_graph([sample])

    problems: list[str] = []
    node_classes = {str(data.get("__class__") or "") for _, data in graph.nodes(data=True)}
    missing_classes = [name for name in plan.entity_classes if name not in node_classes]
    if missing_classes:
        problems.append(f"no node created for entity class(es): {missing_classes}")
    present_labels = {str(data.get("label") or "") for _, _, data in graph.edges(data=True)}
    missing_labels = [label for label in plan.edge_labels if label not in present_labels]
    if missing_labels:
        problems.append(f"edge label(s) missing from graph: {missing_labels}")
    empty_identity = graph.graph.get("empty_identity_nodes") or []
    if empty_identity:
        problems.append(f"nodes with empty identity fields: {empty_identity}")
    detail = (
        f"{graph.number_of_nodes()} node(s) / {graph.number_of_edges()} edge(s); "
        f"classes: {', '.join(plan.entity_classes)}"
    )
    return (problems, detail)


def verify_template_source(
    source: str,
    *,
    root_class: str,
    spec: TemplateSpec | None = None,
) -> VerificationReport:
    """Run gates V1-V6 over rendered template source.

    Never raises for template problems — every failure is a ``GateResult``
    with ``passed=False``. A parse failure skips everything downstream; a V1b
    failure skips V2-V6 (the source is never executed); a V2 failure skips
    V3-V6. Passing the originating ``spec`` arms the spec-level assertions of
    V4 and enables V6 (sample synthesis needs the SPEC); without it V6 is
    recorded as skipped-by-design.
    """
    report = VerificationReport(root_class=root_class)
    gate_names = dict(_GATE_ORDER)

    def record(gate: str, passed: bool, detail: str = "", *, skipped: bool = False) -> None:
        report.gates.append(
            GateResult(
                gate=gate, name=gate_names[gate], passed=passed, skipped=skipped, detail=detail
            )
        )

    def skip_rest(from_gate: str, reason: str) -> None:
        order = [gate for gate, _ in _GATE_ORDER]
        for gate in order[order.index(from_gate) :]:
            record(gate, False, f"skipped: {reason}", skipped=True)

    # V1 — parse + structure
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        record("V1", False, f"SyntaxError: {exc}")
        skip_rest("V1b", "source does not parse")
        return report
    problems = _check_v1_structure(tree, root_class)
    record("V1", not problems, "; ".join(problems) or "one edge() before first class; root last")

    # V1b — import allowlist (before any execution)
    violations = _check_v1b_imports(tree)
    record("V1b", not violations, "; ".join(violations) or "imports restricted to the allowlist")
    if violations:
        skip_rest("V2", "V1b failed — refusing to execute the source")
        return report

    # V2 — exec + the TemplateLoadingStage predicate. The source runs inside a
    # real (temporarily registered) module: pydantic resolves annotations
    # through sys.modules[cls.__module__], so a bare dict namespace would
    # defer the model build and fail every schema-touching gate downstream —
    # unlike the imported modules TemplateLoadingStage produces.
    module = types.ModuleType(_EXEC_MODULE_NAME)
    module.__file__ = f"<{_EXEC_MODULE_NAME}>"
    sys.modules[_EXEC_MODULE_NAME] = module
    try:
        namespace: dict[str, Any] = module.__dict__
        try:
            exec(compile(source, f"<{_EXEC_MODULE_NAME}>", "exec"), namespace)
        except Exception as exc:
            record("V2", False, f"{type(exc).__name__}: {exc}")
            skip_rest("V3", "V2 failed — no live classes")
            return report
        obj = namespace.get(root_class)
        if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
            record(
                "V2", False, f"'{root_class}' is not a BaseModel subclass in the module namespace"
            )
            skip_rest("V3", "V2 failed — no live root class")
            return report
        root_model: type[BaseModel] = obj
        record("V2", True, f"'{root_class}' loads under the TemplateLoadingStage predicate")
        _run_schema_gates(report, record, root_model, namespace, spec)
    finally:
        sys.modules.pop(_EXEC_MODULE_NAME, None)

    return report


def _run_schema_gates(
    report: VerificationReport,
    record: Callable[..., None],
    root_model: type[BaseModel],
    namespace: Mapping[str, Any],
    spec: TemplateSpec | None,
) -> None:
    """Gates V3-V6 (require the live root class from V2)."""
    # V3 — provider-safe response_format schema
    schema: dict[str, Any] | None = None
    try:
        schema = root_model.model_json_schema()
        # The client transports the schema as JSON text and normalizes with the
        # default top_level="object" / name="extraction_result" (litellm.py).
        payload = json.loads(json.dumps(schema))
        normalized = normalize_schema_for_response_format(payload)
        ok = (
            isinstance(normalized, dict)
            and normalized.get("strict") is True
            and isinstance(normalized.get("schema"), dict)
            and normalized["schema"].get("type") == "object"
        )
        detail = (
            f"schema '{normalized.get('name')}' normalized (strict object)"
            if ok
            else f"normalized payload malformed: {str(normalized)[:200]}"
        )
        record("V3", ok, detail)
    except Exception as exc:
        record("V3", False, f"{type(exc).__name__}: {exc}")

    # V4 — dense catalog walk
    try:
        catalog = build_node_catalog(root_model)
        problems, detail = _check_v4_catalog(catalog, report.root_class, spec)
        record("V4", not problems, "; ".join(problems) or detail)
    except Exception as exc:
        record("V4", False, f"{type(exc).__name__}: {exc}")

    # V5 — semantic guide preview
    try:
        guide = build_compact_semantic_guide(
            schema if schema is not None else root_model.model_json_schema()
        )
        if guide.strip():
            report.semantic_guide_preview = guide
            record("V5", True, f"guide preview: {len(guide)} chars")
        else:
            record("V5", False, "build_compact_semantic_guide returned empty text")
    except Exception as exc:
        record("V5", False, f"{type(exc).__name__}: {exc}")

    # V6 — graph-shape smoke test (needs the SPEC for sample synthesis)
    if spec is None:
        record(
            "V6",
            True,
            "skipped: no spec provided (sample synthesis requires the SPEC)",
            skipped=True,
        )
    else:
        try:
            problems, detail = _run_v6(spec, namespace)
            record("V6", not problems, "; ".join(problems) or detail)
        except Exception as exc:
            record("V6", False, f"{type(exc).__name__}: {exc}")
