"""
LinkML schema -> TemplateSpec draft compiler (linkml-runtime SchemaView).

A pure compiler (design §5.2): ``SchemaView.class_induced_slots`` resolves
inheritance/mixins/``slot_usage`` natively, so flattening is free. **Zero LLM
calls.**

Mapping summary (tested in ``tests/unit/templategen/test_ontology_linkml.py``):

- ``ClassDefinition`` (non-abstract, non-mixin) -> ModelSpec; ``description``
  -> docstring; ``tree_root: true`` -> root (else ``root=``).
- ``class_induced_slots`` -> fields with inheritance already applied.
- slot ``identifier: true`` / ``key: true`` -> identity; else the heuristic
  identity ladder; else demotion to component + ``missing_identity`` gap
  (identity-less root -> synthesized ``document_reference``).
- slot ``range`` = class -> edge field (``multivalued`` -> list;
  ``inlined: false`` -> ``reference=True`` — LinkML's not-inlined *is*
  docling-graph's reference edge).
- slot ``range`` = abstract/mixin class -> fanned out to one edge per concrete
  descendant (``<field>_<child_snake>``, mirroring the OWL dropped-abstract
  convention); no concrete descendant -> scalar ``str`` + a generator note.
- slot ``range`` = enum -> EnumSpec from ``permissible_values`` (+ their
  ``title``/``meaning`` as synonyms).
- ``maximum_cardinality``: 1 -> single optional; n>1 -> list + the documented
  ``max_instances = n`` on the target (the linter's ``repair_draft`` doubles
  exactly once) + "At most n per document." sentence.
  ``minimum_cardinality >= 1`` -> description note only.
- ``required: true`` on a non-identity slot -> stays optional; recorded in the
  draft's generator notes (the Optionality Law outranks the schema).
- slot ``examples`` -> field examples (the ontology path's best example source).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from docling_graph.templategen.naming import (
    normalize_edge_label,
    sanitize_class_name,
    sanitize_field_name,
    to_snake_case,
)
from docling_graph.templategen.ontology import (
    MIN_CARDINALITY_NOTE,
    cardinality_sentence,
    class_passes_filters,
    document_reference_field,
    pick_ladder_identity,
    placeholder_docstring,
    require_optional_dependency,
    unique_name,
)
from docling_graph.templategen.spec import SpecGap

if TYPE_CHECKING:
    # linkml-runtime ships no py.typed marker: both imports are Any to mypy.
    from linkml_runtime.linkml_model.meta import SlotDefinition  # type: ignore[import-untyped]
    from linkml_runtime.utils.schemaview import SchemaView  # type: ignore[import-untyped]

LINKML_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "str": "str",
    "integer": "int",
    "int": "int",
    "float": "float",
    "double": "float",
    "decimal": "float",
    "boolean": "bool",
    "bool": "bool",
    "date": "date",
    "datetime": "datetime",
    "date_or_datetime": "str",
    "time": "str",
    "uri": "str",
    "uriorcurie": "str",
    "curie": "str",
    "ncname": "str",
    "objectidentifier": "str",
    "nodeidentifier": "str",
    "jsonpointer": "str",
    "jsonpath": "str",
    "sparqlpath": "str",
}
"""LinkML built-in type name -> SPEC scalar; custom types resolve via ``typeof``."""


def spec_draft_from_linkml(
    path: str | Path,
    *,
    root: str | None = None,
    depth: int = 4,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[SpecGap]]:
    """Compile a LinkML YAML schema into a loose TemplateSpec draft + gaps.

    Args:
        path: LinkML schema file (YAML with ``classes:`` and ``slots:``).
        root: Root class name. ``None`` uses the single ``tree_root: true``
            class, else the single class that is no slot's range.
        depth: Maximum BFS depth from the root over class-ranged slots.
        include: Glob patterns over class names; non-matching classes pruned.
        exclude: Glob patterns over class names to drop (wins over include).

    Raises:
        ImportError: linkml-runtime is not installed
            (``pip install 'docling-graph[templategen]'``).
        ValueError: Unparseable schema or unresolvable/ambiguous root.
    """
    require_optional_dependency("linkml_runtime", purpose="LinkML schema compilation")
    from linkml_runtime.utils.schemaview import SchemaView

    try:
        view = SchemaView(str(path))
        view.all_classes()
    except Exception as exc:
        raise ValueError(f"linkml-runtime could not load '{path}': {exc}") from exc
    return _LinkmlCompiler(
        view, source=Path(path), root=root, depth=depth, include=include, exclude=exclude
    ).compile()


class _LinkmlCompiler:
    """Single-use compiler: one SchemaView -> one draft dict + gaps."""

    def __init__(
        self,
        view: SchemaView,
        *,
        source: Path,
        root: str | None,
        depth: int,
        include: Sequence[str] | None,
        exclude: Sequence[str] | None,
    ) -> None:
        self._view = view
        self._source = source
        self._root_arg = root
        self._depth = depth
        self._include = include
        self._exclude = exclude
        self._notes: list[str] = []
        self._gaps: list[SpecGap] = []
        all_classes = view.all_classes()
        self._concrete: dict[str, Any] = {
            str(name): cls
            for name, cls in all_classes.items()
            if not cls.abstract and not cls.mixin
        }
        self._abstract: dict[str, Any] = {
            str(name): cls for name, cls in all_classes.items() if cls.abstract or cls.mixin
        }
        self._enum_defs = view.all_enums()
        self._closure: dict[str, int] = {}
        self._model_names: dict[str, str] = {}
        self._taken_names: set[str] = set()
        self._enums_used: dict[str, str] = {}
        self._enum_drafts: list[dict[str, Any]] = []
        self._bounds: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def compile(self) -> tuple[dict[str, Any], list[SpecGap]]:
        root_class = self._resolve_root()
        self._walk_closure(root_class)
        for class_name in self._closure:
            self._model_names[class_name] = unique_name(
                sanitize_class_name(class_name), self._taken_names
            )

        models = [self._build_model(name, root_class) for name in self._closure]
        self._apply_cardinality_bounds(models)

        draft = {
            "module_docstring": self._view.schema.description
            or (
                f"Template draft compiled from LinkML schema '{self._source.name}' "
                f"(root: {self._model_names[root_class]})."
            ),
            "root": self._model_names[root_class],
            "enums": self._enum_drafts,
            "models": models,
            "needs_root_list_dedup": [],
            "generator": {
                "format": "linkml",
                "source": str(self._source),
                "root": root_class,
                "notes": self._notes,
            },
        }
        return draft, self._gaps

    # ------------------------------------------------------------------ #
    # Root + closure
    # ------------------------------------------------------------------ #

    def _resolve_root(self) -> str:
        if self._root_arg is not None:
            if self._root_arg in self._concrete:
                return self._root_arg
            by_pascal = {sanitize_class_name(name): name for name in sorted(self._concrete)}
            resolved = by_pascal.get(sanitize_class_name(self._root_arg))
            if resolved is not None:
                return resolved
            available = ", ".join(sorted(self._concrete))
            raise ValueError(
                f"No class named '{self._root_arg}' in the schema. Available classes: {available}."
            )

        tree_roots = sorted(name for name, cls in self._concrete.items() if cls.tree_root)
        if len(tree_roots) == 1:
            return tree_roots[0]
        if len(tree_roots) > 1:
            raise ValueError(
                f"Multiple tree_root classes ({', '.join(tree_roots)}); pass --root explicitly."
            )

        ranged: set[str] = set()
        for name in self._concrete:
            for slot in self._induced_slots(name):
                target = str(slot.range) if slot.range else ""
                if target in self._concrete:
                    ranged.add(target)
                elif target in self._abstract:
                    # Abstract ranges fan out: their concrete descendants are ranged.
                    ranged.update(self._concrete_descendants(target))
        candidates = sorted(name for name in self._concrete if name not in ranged)
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            "Cannot auto-detect the root class (no tree_root, and classes that are "
            f"no slot's range = {', '.join(candidates) or 'none'}). "
            "Pass --root explicitly."
        )

    def _induced_slots(self, class_name: str) -> list[SlotDefinition]:
        return list(self._view.class_induced_slots(class_name))

    def _concrete_descendants(self, class_name: str) -> list[str]:
        """Concrete (non-abstract, non-mixin) descendants of an abstract/mixin class."""
        return sorted(
            str(descendant)
            for descendant in self._view.class_descendants(class_name)
            if str(descendant) in self._concrete
        )

    def _passes_filters(self, class_name: str, *, is_root: bool = False) -> bool:
        return class_passes_filters(class_name, self._include, self._exclude, is_root=is_root)

    def _walk_closure(self, root_class: str) -> None:
        queue: deque[tuple[str, int]] = deque([(root_class, 0)])
        while queue:
            current, current_depth = queue.popleft()
            if current in self._closure:
                continue
            self._closure[current] = current_depth
            if current_depth >= self._depth:
                continue
            for slot in self._induced_slots(current):
                target = str(slot.range) if slot.range else ""
                if target in self._abstract:
                    # Abstract/mixin ranges fan out to their concrete descendants.
                    for descendant in self._concrete_descendants(target):
                        if descendant not in self._closure and self._passes_filters(descendant):
                            queue.append((descendant, current_depth + 1))
                elif (
                    target in self._concrete
                    and target not in self._closure
                    and self._passes_filters(target)
                ):
                    queue.append((target, current_depth + 1))

    # ------------------------------------------------------------------ #
    # Model building
    # ------------------------------------------------------------------ #

    def _build_model(self, class_name: str, root_class: str) -> dict[str, Any]:
        cls = self._concrete[class_name]
        model_name = self._model_names[class_name]
        slots = self._induced_slots(class_name)

        fields: list[dict[str, Any]] = []
        scalar_field_names: list[str] = []
        declared_identity: list[str] = []
        used: set[str] = set()

        for slot in sorted(slots, key=lambda s: str(s.name)):
            for field_draft in self._build_field(slot, model_name, used):
                fields.append(field_draft)
                is_scalar = field_draft["type"] in LINKML_TYPE_MAP.values()
                if field_draft.pop("_declared_identity", False):
                    declared_identity.append(field_draft["name"])
                elif is_scalar and field_draft.get("role") == "property":
                    scalar_field_names.append(field_draft["name"])

        identity = self._select_identity(model_name, declared_identity, scalar_field_names)

        kind = "root" if class_name == root_class else ("entity" if identity else "component")
        if class_name == root_class and not identity:
            fields.insert(0, document_reference_field())
            identity = ["document_reference"]
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    field="document_reference",
                    kind="missing_identity",
                    note=(
                        "The schema gives the root no identifier/key slot; synthesized "
                        "'document_reference'. Replace with a real printed identifier."
                    ),
                )
            )
        elif not identity:
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    kind="missing_identity",
                    note=(
                        "No identifier/key slot and no identity-like slot name; "
                        "demoted to component (never invent ids)."
                    ),
                )
            )

        for field_draft in fields:
            if field_draft["name"] in identity:
                field_draft["role"] = "identity"
                field_draft["is_list"] = False
                if field_draft["type"] not in LINKML_TYPE_MAP.values():
                    self._notes.append(
                        f"Identity slot '{field_draft['name']}' on '{model_name}' is not "
                        "scalar-ranged; coerced to 'str' (identity must be scalar)."
                    )
                    field_draft["type"] = "str"
                field_draft.pop("edge_label", None)
                field_draft.pop("reference", None)
                field_draft.pop("normalizer", None)
                if len(field_draft.get("examples", [])) < 2:
                    self._gaps.append(
                        SpecGap(
                            model=model_name,
                            field=field_draft["name"],
                            kind="missing_examples",
                            note="Add 2-5 verbatim examples from a real document "
                            "(slot examples are harvested when present).",
                        )
                    )

        docstring = str(cls.description) if cls.description else ""
        if not docstring:
            docstring = placeholder_docstring(model_name)
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    kind="missing_docstring",
                    note="No class description in the schema.",
                )
            )
        if cls.aliases:
            docstring = f"{docstring} Also called: {', '.join(sorted(map(str, cls.aliases)))}."

        return {
            "name": model_name,
            "kind": kind,
            "docstring": docstring,
            "identity_fields": identity,
            "max_instances": None,
            "fields": fields,
            "canonical_home": None,
            "provenance": "ontology",
            "source_ref": self._class_uri(class_name),
        }

    def _build_field(
        self, slot: SlotDefinition, model_name: str, used: set[str]
    ) -> list[dict[str, Any]]:
        """Field draft(s) for one induced slot.

        Usually one draft; an abstract/mixin-ranged slot fans out to one edge
        per concrete descendant; a pruned edge target yields none.
        """
        base_name = sanitize_field_name(str(slot.name))
        field_name = base_name
        counter = 2
        while field_name in used:
            field_name = f"{base_name}_{counter}"
            counter += 1

        max_card = _as_int(slot.maximum_cardinality)
        min_card = _as_int(slot.minimum_cardinality)
        is_list = bool(slot.multivalued) and max_card != 1
        description = str(slot.description) if slot.description else ""
        if min_card is not None and min_card >= 1:
            description = _join_sentences(description, MIN_CARDINALITY_NOTE)

        declared_identity = bool(slot.identifier) or bool(slot.key)
        if slot.required and not declared_identity:
            self._notes.append(
                f"Slot '{slot.name}' on '{model_name}' is required in the schema; "
                "rendered optional (the Optionality Law: required = identity, "
                "nothing else)."
            )

        examples = [
            str(example.value) for example in (slot.examples or []) if example.value is not None
        ][:5]

        target = str(slot.range) if slot.range else ""
        field_draft: dict[str, Any] = {
            "name": field_name,
            "is_list": is_list,
            "description": description,
            "examples": examples,
            "role": "property",
            "_declared_identity": declared_identity,
        }

        if target in self._concrete:
            if target not in self._closure:
                self._notes.append(
                    f"Edge '{model_name}.{field_name}' dropped: target '{target}' is "
                    "outside the closure (depth/include/exclude)."
                )
                return []
            field_draft["type"] = self._model_names[target]
            field_draft["role"] = "edge"
            field_draft["edge_label"] = normalize_edge_label(str(slot.name), target=target)
            if slot.inlined is False:
                # LinkML not-inlined is docling-graph's reference edge (1:1 match).
                field_draft["reference"] = True
            if max_card is not None and max_card > 1:
                bound_name = self._model_names[target]
                self._bounds[bound_name] = max(self._bounds.get(bound_name, 0), max_card)
        elif target in self._abstract and not declared_identity:
            return self._fan_out_abstract(slot, target, field_draft, model_name, used, max_card)
        elif target in self._enum_defs:
            field_draft["type"] = self._ensure_enum(target)
            field_draft["normalizer"] = "enum"
        else:
            if target in self._abstract:
                # Identity must be scalar (R2); an abstract-ranged identifier
                # slot cannot fan out into edges, so it degrades to str.
                self._notes.append(
                    f"Identifier slot '{slot.name}' on '{model_name}' ranges on "
                    f"abstract class '{target}'; coerced to 'str' (identity must "
                    "be scalar)."
                )
                target = "string"
            field_draft["type"] = self._scalar_for_type(target)
            if max_card is not None and max_card > 1:
                field_draft["description"] = _join_sentences(
                    field_draft["description"],
                    f"At most {max_card} values per document.",
                )

        used.add(field_name)
        return [field_draft]

    def _fan_out_abstract(
        self,
        slot: SlotDefinition,
        target: str,
        field_draft: dict[str, Any],
        model_name: str,
        used: set[str],
        max_card: int | None,
    ) -> list[dict[str, Any]]:
        """Fan an abstract/mixin-ranged slot out to its concrete descendants.

        Mirrors the OWL dropped-abstract convention: one ``<field>_<child_snake>``
        edge per emitted descendant, all sharing the slot's edge label. A range
        with no concrete descendant degrades to a scalar ``str`` field (never a
        silent drop) plus a generator note.
        """
        field_name = str(field_draft["name"])
        descendants = self._concrete_descendants(target)
        if not descendants:
            self._notes.append(
                f"Slot '{slot.name}' on '{model_name}' ranges on abstract class "
                f"'{target}' with no concrete descendant; typed as 'str'."
            )
            field_draft["type"] = "str"
            used.add(field_name)
            return [field_draft]

        drafts: list[dict[str, Any]] = []
        fanned: list[str] = []
        for descendant in descendants:
            if descendant not in self._closure:
                self._notes.append(
                    f"Edge '{model_name}.{field_name}' fan-out to '{descendant}' dropped: "
                    "outside the closure (depth/include/exclude)."
                )
                continue
            child_base = f"{field_name}_{to_snake_case(self._model_names[descendant])}"
            child_name = child_base
            counter = 2
            while child_name in used:
                child_name = f"{child_base}_{counter}"
                counter += 1
            child_draft = dict(field_draft)
            child_draft["name"] = child_name
            child_draft["type"] = self._model_names[descendant]
            child_draft["role"] = "edge"
            child_draft["edge_label"] = normalize_edge_label(str(slot.name), target=descendant)
            if slot.inlined is False:
                child_draft["reference"] = True
            if max_card is not None and max_card > 1:
                bound_name = self._model_names[descendant]
                self._bounds[bound_name] = max(self._bounds.get(bound_name, 0), max_card)
            used.add(child_name)
            drafts.append(child_draft)
            fanned.append(self._model_names[descendant])
        if fanned:
            self._notes.append(
                f"Slot '{slot.name}' on '{model_name}' ranges on abstract class "
                f"'{target}'; fanned out per concrete descendant: {', '.join(fanned)}."
            )
        else:
            self._notes.append(
                f"Edge '{model_name}.{field_name}' dropped: abstract range '{target}' "
                "contributes no descendant inside the closure."
            )
        return drafts

    def _select_identity(
        self, model_name: str, declared: list[str], scalar_fields: list[str]
    ) -> list[str]:
        if declared:
            if len(declared) > 2:
                self._gaps.append(
                    SpecGap(
                        model=model_name,
                        field=declared[2],
                        kind="missing_identity",
                        note=(
                            f"{len(declared)} identifier/key slots; only the first 2 "
                            "kept as identity (identity is 1-2 scalar fields)."
                        ),
                    )
                )
            return declared[:2]
        ladder_pick = pick_ladder_identity(scalar_fields)
        return [ladder_pick] if ladder_pick else []

    # ------------------------------------------------------------------ #
    # Enums, types, post passes
    # ------------------------------------------------------------------ #

    def _ensure_enum(self, enum_name: str) -> str:
        if enum_name in self._enums_used:
            return self._enums_used[enum_name]
        definition = self._enum_defs[enum_name]
        name = unique_name(sanitize_class_name(enum_name), self._taken_names)
        members: list[str] = []
        synonyms: dict[str, list[str]] = {}
        for text, permissible in definition.permissible_values.items():
            member = str(text)
            members.append(member)
            alternates: list[str] = []
            if permissible.title and str(permissible.title) != member:
                alternates.append(str(permissible.title))
            if permissible.meaning:
                alternates.append(str(permissible.meaning))
            if alternates:
                synonyms[member] = alternates
        self._enums_used[enum_name] = name
        self._enum_drafts.append(
            {
                "name": name,
                "members": members,
                "synonyms": synonyms,
                "include_other": True,
            }
        )
        return name

    def _scalar_for_type(self, type_name: str) -> str:
        current: str | None = type_name or "string"
        seen: set[str] = set()
        all_types = self._view.all_types()
        while current and current not in LINKML_TYPE_MAP and current not in seen:
            seen.add(current)
            definition = all_types.get(current)
            current = str(definition.typeof) if definition and definition.typeof else None
        return LINKML_TYPE_MAP.get(current or "", "str")

    def _apply_cardinality_bounds(self, models: list[dict[str, Any]]) -> None:
        by_name = {m["name"]: m for m in models}
        for target_name, documented_max in sorted(self._bounds.items()):
            model = by_name.get(target_name)
            if model is None:
                continue
            if model["kind"] != "entity":
                self._notes.append(
                    f"Cardinality bound on '{target_name}' skipped: "
                    f"{model['kind']}s take no max_instances."
                )
                continue
            # The DOCUMENTED maximum: the linter's repair_draft doubles it once.
            model["max_instances"] = documented_max
            model["docstring"] = _join_sentences(
                model["docstring"], cardinality_sentence(documented_max)
            )

    def _class_uri(self, class_name: str) -> str:
        try:
            return str(self._view.get_uri(self._concrete[class_name], expand=True))
        except Exception:
            return class_name


def _as_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _join_sentences(base: str, addition: str) -> str:
    if not base:
        return addition
    if not base.rstrip().endswith((".", "!", "?")):
        base = f"{base.rstrip()}."
    return f"{base.rstrip()} {addition}"
