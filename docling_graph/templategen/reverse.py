"""
Template -> SPEC reconstruction: the reverse flow behind ``template lint``.

Generated templates flow SPEC -> renderer -> Python; existing templates
(generated or hand-written) flow the other way — a live template class is
walked back into a loose :class:`~docling_graph.templategen.spec.TemplateSpec`
draft so the full rulebook linter can judge it. The walk mirrors the exact
runtime reads:

- ``model_config`` values via the duck-typed ``.get`` pattern of
  ``node_id_registry.get_model_config_value`` (``graph_id_fields``,
  ``is_entity``, ``graph_max_instances``);
- ``json_schema_extra`` markers exactly as the converter and the dense catalog
  read them: ``edge_label`` (``GraphConverter._get_edge_label`` — a Mapping
  carrying a str), ``graph_reference`` (``dense/catalog._is_reference_field``
  — honored only when ``is True``), ``reference_closed_catalog``
  (``GraphConverter._edge_properties`` — truthy).

The reconstruction is deliberately **lenient**: anything the IR cannot
represent (three identity fields, a required non-identity field, an unknown
scalar type, a callable ``json_schema_extra``, ...) becomes a human-readable
*finding*, never a crash — ``template lint`` must always produce a report.
Constructs the IR has no slot for at all (custom validators, field aliases,
enum synonym tables folded into descriptions) are simply not recovered.

``max_instances`` contract: a live template stores the already-doubled
``graph_max_instances`` bound while drafts carry the *documented* maximum
(``linter.repair_draft`` doubles exactly once), so the walk emits
``graph_max_instances // 2`` and flags odd bounds that cannot round-trip.

:func:`spec_from_template` chains the walk into ``repair_draft``: for an
existing template the repair report *is* the lint output — every entry states
what the rulebook would change. :func:`spec_from_dotted_path` adds the
AST import-allowlist pre-check (reusing the verification gate's V1b rules)
before the module is ever imported, because linting executes user code.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Mapping, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .linter import LintEntry, LintReport, TemplateLintError, repair_draft
from .spec import MAX_FIELD_EXAMPLES, MAX_IDENTITY_FIELDS, TemplateSpec
from .verify import _check_v1b_imports

__all__ = [
    "reverse_draft",
    "spec_from_dotted_path",
    "spec_from_template",
]

try:  # Python 3.10+: X | Y annotations
    from types import UnionType

    _UNION_ORIGINS: tuple[Any, ...] = (Union, UnionType)
except ImportError:  # pragma: no cover - 3.10 is the repo floor
    _UNION_ORIGINS = (Union,)

_SCALAR_BY_TYPE: dict[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    date: "date",
    datetime: "datetime",
}

_LIST_LIKE_ORIGINS: tuple[type, ...] = (list, set, frozenset, tuple)

_REVERSE_DOC_REF = "docling_graph/templategen/reverse.py (template -> SPEC reconstruction)"


# ---------------------------------------------------------------------------
# Duck-typed runtime reads (mirroring node_id_registry / graph_converter /
# dense/catalog exactly)
# ---------------------------------------------------------------------------


def _config_value(model: type[BaseModel], key: str, default: Any) -> Any:
    """``get_model_config_value`` for a class: duck-typed ``.get`` first."""
    config = model.model_config
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _edge_label_of(field_info: FieldInfo) -> str | None:
    """Mirror ``GraphConverter._get_edge_label``: Mapping extra, str value."""
    extra = field_info.json_schema_extra
    if isinstance(extra, Mapping):
        value = extra.get("edge_label")
        if isinstance(value, str):
            return value
    return None


def _is_reference(field_info: FieldInfo) -> bool:
    """Mirror ``dense/catalog._is_reference_field``: honored only when ``is True``."""
    extra = field_info.json_schema_extra
    return isinstance(extra, Mapping) and extra.get("graph_reference") is True


def _is_closed_catalog(field_info: FieldInfo) -> bool:
    """Mirror ``GraphConverter._edge_properties``: truthy marker."""
    extra = field_info.json_schema_extra
    return bool(isinstance(extra, Mapping) and extra.get("reference_closed_catalog"))


# ---------------------------------------------------------------------------
# Annotation unwrapping
# ---------------------------------------------------------------------------


def _unwrap_annotation(annotation: Any) -> tuple[Any, bool, list[str]]:
    """Peel Optional/List/Union/Annotated layers off a field annotation.

    Returns ``(core, is_list, notes)`` where ``core`` is the innermost type
    (scalar / Enum / BaseModel / anything else) and ``notes`` describe layers
    the IR cannot represent (multi-type unions, non-list containers, nested
    list-of-list shapes).
    """
    is_list = False
    notes: list[str] = []
    current = annotation
    for _ in range(16):  # annotations are finite; guard against pathologies
        origin = get_origin(current)
        if origin is Annotated:
            current = get_args(current)[0]
        elif origin in _UNION_ORIGINS:
            args = [a for a in get_args(current) if a is not type(None)]
            if not args:
                notes.append("annotation is None-only")
                return (type(None), is_list, notes)
            if len(args) > 1:
                model_args = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
                chosen = model_args[0] if model_args else args[0]
                notes.append(
                    f"multi-type union collapsed to '{getattr(chosen, '__name__', chosen)}'"
                )
                current = chosen
            else:
                current = args[0]
        elif origin in _LIST_LIKE_ORIGINS:
            if origin is not list:
                notes.append(f"'{origin.__name__}' container treated as a list")
            if is_list:
                notes.append("nested list-of-list flattened to a single list")
            is_list = True
            item_args = get_args(current)
            if not item_args:
                return (Any, is_list, notes)
            current = item_args[0]
        else:
            return (current, is_list, notes)
    notes.append("annotation nesting too deep to unwrap")
    return (current, is_list, notes)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


class _Walk:
    """Mutable state of one reverse walk (cycle-safe via visited class ids)."""

    def __init__(self, root: type[BaseModel]) -> None:
        self.root = root
        self.findings: list[str] = []
        self.models: list[dict[str, Any]] = []
        self.enum_specs: dict[int, dict[str, Any]] = {}
        self._names: dict[int, str] = {}
        self._taken: set[str] = set()
        self._queue: list[type[BaseModel]] = []
        self._seen: set[int] = set()

    def note(self, where: str, message: str) -> None:
        self.findings.append(f"{where}: {message}")

    def assign_name(self, cls: type) -> str:
        """Stable per-class name; distinct same-named classes get a suffix."""
        key = id(cls)
        if key in self._names:
            return self._names[key]
        base = cls.__name__
        name = base
        suffix = 2
        while name in self._taken:
            name = f"{base}_{suffix}"
            suffix += 1
        if name != base:
            self.note(base, f"duplicate class name across the template — renamed to '{name}'")
        self._names[key] = name
        self._taken.add(name)
        return name

    def enqueue_model(self, cls: type[BaseModel]) -> str:
        name = self.assign_name(cls)
        if id(cls) not in self._seen:
            self._seen.add(id(cls))
            self._queue.append(cls)
        return name

    def register_enum(self, cls: type[Enum]) -> str:
        key = id(cls)
        if key in self.enum_specs:
            return str(self.enum_specs[key]["name"])
        name = self.assign_name(cls)
        members: list[str] = []
        include_other = False
        for member in cls:
            if member.name == "OTHER":
                include_other = True
                continue
            if not isinstance(member.value, str):
                self.note(
                    name,
                    f"enum member '{member.name}' has a non-string value "
                    f"{member.value!r} — stringified (enums render as (str, Enum))",
                )
            members.append(str(member.value))
        self.enum_specs[key] = {
            "name": name,
            "members": members,
            "include_other": include_other,
        }
        return name

    def run(self) -> None:
        self.enqueue_model(self.root)
        while self._queue:
            cls = self._queue.pop(0)
            self.models.append(self._reverse_model(cls))

    # -- per-model ----------------------------------------------------------

    def _reverse_model(self, cls: type[BaseModel]) -> dict[str, Any]:
        name = self.assign_name(cls)
        is_entity_flag = _config_value(cls, "is_entity", True)
        kind = (
            "root" if cls is self.root else ("component" if is_entity_flag is False else "entity")
        )

        raw_ids = _config_value(cls, "graph_id_fields", [])
        if not isinstance(raw_ids, list | tuple):
            self.note(name, f"graph_id_fields is not a list ({raw_ids!r}) — ignored")
            raw_ids = []
        identity: list[str] = []
        for id_name in raw_ids:
            if not isinstance(id_name, str):
                self.note(name, f"non-string graph_id_fields entry {id_name!r} — dropped")
            elif id_name not in cls.model_fields:
                self.note(name, f"graph_id_fields names undeclared field '{id_name}' — dropped")
            elif id_name not in identity:
                identity.append(id_name)

        if kind == "component" and identity:
            self.note(
                name,
                "is_entity=False with graph_id_fields — the converter embeds components "
                "and ignores their identity; the linter clears it",
            )
        if kind == "entity" and not identity:
            self.note(
                name,
                "no graph_id_fields and not marked is_entity=False — the IR has no such "
                "kind; the linter demotes it to a component (never invent ids)",
            )
        if len(identity) > MAX_IDENTITY_FIELDS:
            self.note(
                name,
                f"{len(identity)} identity fields exceed the {MAX_IDENTITY_FIELDS}-field "
                "identity budget — the linter keeps the best two",
            )

        max_instances = self._reverse_max_instances(cls, name)
        docstring = " ".join((cls.__doc__ or "").split())
        if not docstring:
            self.note(name, "class has no docstring — the linter injects a placeholder")

        fields = [
            self._reverse_field(cls, name, field_name, field_info, identity)
            for field_name, field_info in cls.model_fields.items()
        ]
        model: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "docstring": docstring,
            "identity_fields": identity,
            "fields": fields,
            "provenance": "user",
        }
        if max_instances is not None:
            model["max_instances"] = max_instances
        return model

    def _reverse_max_instances(self, cls: type[BaseModel], name: str) -> int | None:
        raw = _config_value(cls, "graph_max_instances", None)
        if raw is None:
            return None
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
            self.note(name, f"graph_max_instances={raw!r} is not a positive int — ignored")
            return None
        documented = max(1, raw // 2)
        if documented * 2 != raw:
            self.note(
                name,
                f"graph_max_instances={raw} is not an even 2x bound — the draft carries "
                f"the documented maximum {documented} (re-renders as {documented * 2})",
            )
        return documented

    # -- per-field ----------------------------------------------------------

    def _reverse_field(
        self,
        cls: type[BaseModel],
        model_name: str,
        field_name: str,
        field_info: FieldInfo,
        identity: list[str],
    ) -> dict[str, Any]:
        where = f"{model_name}.{field_name}"
        core, is_list, notes = _unwrap_annotation(field_info.annotation)
        for note in notes:
            self.note(where, note)

        if isinstance(core, type) and issubclass(core, BaseModel):
            type_name = self.enqueue_model(core)
            is_model = True
        elif isinstance(core, type) and issubclass(core, Enum):
            type_name = self.register_enum(core)
            is_model = False
        elif isinstance(core, type) and core in _SCALAR_BY_TYPE:
            type_name = _SCALAR_BY_TYPE[core]
            is_model = False
        else:
            shown = getattr(core, "__name__", None) or str(core)
            self.note(where, f"unknown scalar type '{shown}' — carried as 'str'")
            type_name = "str"
            is_model = False

        extra = field_info.json_schema_extra
        if extra is not None and not isinstance(extra, Mapping):
            self.note(
                where,
                "json_schema_extra is not a mapping (callable?) — edge markers unreadable",
            )
        edge_label = _edge_label_of(field_info)
        reference = _is_reference(field_info)
        closed_catalog = _is_closed_catalog(field_info)

        if field_name in identity:
            role = "identity"
            if edge_label or reference or closed_catalog:
                self.note(where, "identity field carries edge markers — the linter clears them")
            if not field_info.is_required():
                self.note(
                    where,
                    "identity field is not required in the source template — the renderer "
                    "emits identity fields as required",
                )
        elif edge_label is not None:
            role = "edge"
            if not is_model:
                self.note(
                    where,
                    f"edge_label '{edge_label}' on a non-model field — the linter demotes "
                    "it to a property",
                )
        else:
            role = "property"
            if is_model:
                self.note(
                    where,
                    "model-typed field without edge() metadata — the converter falls back "
                    "to the field name as the edge label",
                )
            if reference or closed_catalog:
                self.note(where, "reference markers without an edge_label — the linter clears them")

        if field_info.is_required() and role != "identity":
            self.note(
                where,
                "required non-identity field — the Optionality Law renders every "
                "non-identity field Optional/defaulted",
            )

        examples = [str(example) for example in (field_info.examples or [])]
        if len(examples) > MAX_FIELD_EXAMPLES:
            self.note(where, f"{len(examples)} examples truncated to {MAX_FIELD_EXAMPLES}")
            examples = examples[:MAX_FIELD_EXAMPLES]

        field: dict[str, Any] = {
            "name": field_name,
            "type": type_name,
            "is_list": is_list,
            "description": field_info.description or "",
            "examples": examples,
            "role": role,
        }
        if edge_label is not None:
            field["edge_label"] = edge_label
        if reference:
            field["reference"] = True
        if closed_catalog:
            field["closed_catalog"] = True
        return field


def reverse_draft(template: type[BaseModel]) -> tuple[dict[str, Any], list[str]]:
    """Walk a live template class back into a loose ``TemplateSpec`` draft.

    The returned draft is a plain dict shaped like the IR — the same producer
    format the ontology compilers emit — plus human-readable findings for
    every construct the IR cannot represent. Feed the draft to
    ``linter.repair_draft`` for the canonical spec (or use
    :func:`spec_from_template`, which does exactly that).

    Args:
        template: The root template class (any BaseModel subclass).

    Returns:
        ``(draft, findings)``. The draft is never mutated by this module
        afterwards; findings are ordered by the walk (breadth-first from the
        root, fields in declaration order).

    Raises:
        TypeError: ``template`` is not a BaseModel subclass.
    """
    if not (isinstance(template, type) and issubclass(template, BaseModel)):
        raise TypeError(f"reverse_draft expects a pydantic BaseModel subclass, got {template!r}")
    walk = _Walk(template)
    walk.run()
    draft: dict[str, Any] = {
        "module_docstring": "",
        "root": walk.assign_name(template),
        "enums": list(walk.enum_specs.values()),
        "models": walk.models,
        "generator": {"source": f"{template.__module__}.{template.__qualname__}"},
    }
    return draft, walk.findings


def spec_from_template(template: type[BaseModel]) -> tuple[TemplateSpec, LintReport]:
    """Reconstruct and lint a SPEC from a live template class.

    This is ``template lint``'s engine for existing templates: the repair
    report conveys exactly what the rulebook *would* change, and the reverse
    walk's findings are prepended as ``REV`` info entries so nothing the IR
    could not represent goes unreported.
    """
    draft, findings = reverse_draft(template)
    spec, report = repair_draft(draft)
    rev_entries = [
        LintEntry(
            rule_id="REV",
            severity="info",
            model=template.__name__,
            field=None,
            message=finding,
            repaired=False,
            doc_ref=_REVERSE_DOC_REF,
        )
        for finding in findings
    ]
    return spec, LintReport(entries=rev_entries + report.entries, gaps=report.gaps)


# ---------------------------------------------------------------------------
# Dotted-path entry point (AST allowlist pre-check + pipeline-identical load)
# ---------------------------------------------------------------------------


def _resolve_module_source(module_path: str) -> Path | None:
    """Best-effort source file for ``module_path`` (cwd fallback like the loader).

    Returns ``None`` when the source is not resolvable (missing module,
    namespace package, frozen module) — the pre-check is then skipped and the
    loader produces its own error.
    """

    def find() -> Any:
        try:
            return importlib.util.find_spec(module_path)
        except (ImportError, ValueError):
            return None

    spec = find()
    if spec is None:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
            try:
                spec = find()
            finally:
                if cwd in sys.path:
                    sys.path.remove(cwd)
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin)
    if origin.suffix != ".py" or not origin.is_file():
        return None
    return origin


def _precheck_module_file(module_path: str, source_file: Path) -> None:
    """Run the V1b import-allowlist gate over the module file (never executes it)."""
    text = source_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        entry = LintEntry(
            rule_id="V1",
            severity="error",
            model=module_path,
            field=None,
            message=f"module does not parse: {exc}",
            repaired=False,
            doc_ref="docling_graph/templategen/verify.py (gate V1)",
        )
        raise TemplateLintError(
            f"Template module '{source_file}' does not parse: {exc}",
            report=LintReport(entries=[entry]),
        ) from exc
    violations = _check_v1b_imports(tree)
    if violations:
        entries = [
            LintEntry(
                rule_id="V1b",
                severity="error",
                model=module_path,
                field=None,
                message=violation,
                repaired=False,
                doc_ref="docling_graph/templategen/verify.py (import allowlist)",
            )
            for violation in violations
        ]
        raise TemplateLintError(
            f"Template module '{source_file}' fails the import allowlist "
            f"({len(violations)} violation(s)); linting executes the module, so it is "
            "not imported. Violations:\n" + "\n".join(violations),
            report=LintReport(entries=entries),
        )


def spec_from_dotted_path(path: str) -> tuple[TemplateSpec, LintReport, type[BaseModel]]:
    """Load a template by dotted path and reconstruct + lint its SPEC.

    Safety order (design §6.2): the module *file* is AST-checked against the
    verification gate's import allowlist **before** any import — linting runs
    arbitrary user code otherwise — then loaded through the pipeline's own
    ``TemplateLoadingStage._load_from_string`` (identical cwd fallback and
    BaseModel predicate), then reversed via :func:`spec_from_template`.

    Args:
        path: Dotted path to the root class, e.g. ``templates.invoices.Invoice``.

    Returns:
        ``(spec, report, template)`` — the reconstructed spec, the lint/repair
        report, and the loaded live class (for downstream verification).

    Raises:
        TemplateLintError: The module file fails the import allowlist or does
            not parse (the module is never executed in either case).
        ConfigurationError: The dotted path cannot be loaded or does not name
            a BaseModel subclass (``TemplateLoadingStage`` semantics).
    """
    module_path, _, _class_name = path.rpartition(".")
    if module_path:
        source_file = _resolve_module_source(module_path)
        if source_file is not None:
            _precheck_module_file(module_path, source_file)

    # Lazy: pipeline.stages pulls the extraction stack; keep reverse import-light.
    from docling_graph.pipeline.stages import TemplateLoadingStage

    template = TemplateLoadingStage._load_from_string(path)
    spec, report = spec_from_template(template)
    return spec, report, template
