"""
JSON Schema -> TemplateSpec draft compiler (stdlib only — no new dependency).

A pure compiler (design §5.3), tested in
``tests/unit/templategen/test_ontology_jsonschema.py``. **Zero LLM calls.**

Mapping summary:

- top-level schema -> root model; ``$defs``/``definitions`` reached via
  ``$ref`` -> models (unreferenced definitions are not emitted). Only object
  definitions become models: a ``$ref`` to an enum definition routes to an
  EnumSpec under the definition's name, a ``$ref`` to a scalar definition maps
  through the type/format tables, and top-level ``$ref`` alias chains between
  definitions are followed first.
- ``properties`` -> fields (``format: date``/``date-time`` mapped; property
  ``examples`` harvested).
- inline ``object`` properties (no ``$ref``) -> components.
- ``enum`` -> EnumSpec (field gets the ``enum`` normalizer).
- ``required`` -> identity **candidates only**: a required string property
  whose name sits on the identity ladder (id/name/number/...) wins; every
  other required field stays optional (the Optionality Law) with a note.
  No candidate -> component demotion + ``missing_identity`` gap
  (identity-less root -> synthesized ``document_reference``).
- ``maxItems``: 1 -> single optional; n>1 -> list + the documented
  ``max_instances = n`` on the target (the linter's ``repair_draft`` doubles
  exactly once) + "At most n per document." docstring sentence.
- ``allOf`` merged (refs resolved; a ``$ref`` cycle through ``allOf`` raises
  ``ValueError`` naming the cycle — design §9, malformed inputs fail with
  actionable errors); ``oneOf``/``anyOf`` of objects -> the branches' common
  fields + a generator note (no discriminated-union idiom is invented).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from docling_graph.templategen.naming import (
    normalize_edge_label,
    sanitize_class_name,
    sanitize_field_name,
)
from docling_graph.templategen.ontology import (
    cardinality_sentence,
    class_passes_filters,
    document_reference_field,
    identity_ladder_rank,
    placeholder_docstring,
    unique_name,
)
from docling_graph.templategen.spec import SpecGap

_JSON_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
}

_FORMAT_MAP: dict[str, str] = {
    "date": "date",
    "date-time": "datetime",
}


def spec_draft_from_jsonschema(
    path: str | Path,
    *,
    root: str | None = None,
    depth: int = 4,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[SpecGap]]:
    """Compile a JSON Schema file into a loose TemplateSpec draft + gaps.

    Args:
        path: JSON Schema file (a JSON object; draft-07 and 2020-12 layouts).
        root: Root model name override (defaults to the schema ``title``, then
            the file stem).
        depth: Maximum nesting depth from the root over ``$ref``/inline objects.
        include: Glob patterns over ``$defs`` names; non-matching pruned.
        exclude: Glob patterns over ``$defs`` names to drop (wins over include).

    Raises:
        ValueError: The file is not a JSON object.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"'{path}' must contain a JSON object at the top level.")
    return _JsonSchemaCompiler(
        schema, source=Path(path), root=root, depth=depth, include=include, exclude=exclude
    ).compile()


class _JsonSchemaCompiler:
    """Single-use compiler: one parsed schema -> one draft dict + gaps."""

    def __init__(
        self,
        schema: dict[str, Any],
        *,
        source: Path,
        root: str | None,
        depth: int,
        include: Sequence[str] | None,
        exclude: Sequence[str] | None,
    ) -> None:
        self._schema = schema
        self._source = source
        self._root_arg = root
        self._depth = depth
        self._include = include
        self._exclude = exclude
        self._defs: dict[str, Any] = schema.get("$defs") or schema.get("definitions") or {}
        self._notes: list[str] = []
        self._gaps: list[SpecGap] = []
        self._taken_names: set[str] = set()
        self._def_models: dict[str, str] = {}
        self._def_enums: dict[str, str] = {}
        self._enum_drafts: list[dict[str, Any]] = []
        self._bounds: dict[str, int] = {}
        self._models: list[dict[str, Any]] = []
        # (model_name, schema, depth, is_root, is_inline, source_ref)
        self._queue: deque[tuple[str, dict[str, Any], int, bool, bool, str]] = deque()

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def compile(self) -> tuple[dict[str, Any], list[SpecGap]]:
        root_name = unique_name(
            sanitize_class_name(
                self._root_arg or str(self._schema.get("title") or self._source.stem)
            ),
            self._taken_names,
        )
        self._queue.append((root_name, self._schema, 0, True, False, "#"))
        while self._queue:
            self._models.append(self._build_model(*self._queue.popleft()))

        self._assign_edge_roles()
        self._apply_cardinality_bounds()

        draft = {
            "module_docstring": str(self._schema.get("description") or "")
            or f"Template draft compiled from JSON Schema '{self._source.name}' "
            f"(root: {root_name}).",
            "root": root_name,
            "enums": self._enum_drafts,
            "models": self._models,
            "needs_root_list_dedup": [],
            "generator": {
                "format": "jsonschema",
                "source": str(self._source),
                "root": root_name,
                "notes": self._notes,
            },
        }
        return draft, self._gaps

    # ------------------------------------------------------------------ #
    # Schema normalization ($ref / allOf / oneOf-anyOf)
    # ------------------------------------------------------------------ #

    def _deref(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Inline a local ``$ref`` (used inside allOf/oneOf normalization)."""
        ref = schema.get("$ref")
        if not isinstance(ref, str):
            return schema
        target = self._ref_target_name(ref)
        if target is None or target not in self._defs:
            return {k: v for k, v in schema.items() if k != "$ref"}
        merged = dict(self._defs[target])
        merged.update({k: v for k, v in schema.items() if k != "$ref"})
        return merged

    @staticmethod
    def _ref_target_name(ref: str) -> str | None:
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                remainder = ref[len(prefix) :]
                if remainder and "/" not in remainder:
                    return remainder
        return None

    def _normalize_object(self, schema: dict[str, Any], context: str) -> dict[str, Any]:
        """Resolve ``allOf`` merges and object-branch ``oneOf``/``anyOf``."""
        schema = self._merge_allof(schema)
        branches = schema.get("oneOf") or schema.get("anyOf")
        if not isinstance(branches, list) or not branches:
            return schema
        resolved = [self._merge_allof(self._deref(b)) for b in branches if isinstance(b, dict)]
        if not resolved or not all(_is_object_schema(b) for b in resolved):
            return schema
        common = set(resolved[0].get("properties", {}))
        required = set(resolved[0].get("required", []))
        for branch in resolved[1:]:
            common &= set(branch.get("properties", {}))
            required &= set(branch.get("required", []))
        merged = {k: v for k, v in schema.items() if k not in ("oneOf", "anyOf")}
        merged["type"] = "object"
        merged["properties"] = {name: resolved[0]["properties"][name] for name in sorted(common)}
        merged["required"] = sorted(required)
        self._notes.append(
            f"'{context}' is a oneOf/anyOf of {len(resolved)} object variants; kept "
            f"the {len(common)} common field(s) (no discriminated-union idiom)."
        )
        return merged

    def _merge_allof(
        self, schema: dict[str, Any], active: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Merge ``allOf`` branches, resolving local ``$ref``\\ s.

        ``active`` carries the definition names currently being expanded on
        this recursion path; re-entering one is a ``$ref`` cycle and raises
        ``ValueError`` naming the cycle instead of recursing forever (design
        §9: malformed inputs fail with actionable errors).
        """
        subschemas = schema.get("allOf")
        if not isinstance(subschemas, list):
            return schema
        merged = {k: v for k, v in schema.items() if k != "allOf"}
        properties = dict(merged.get("properties", {}))
        required = list(merged.get("required", []))
        for subschema in subschemas:
            if not isinstance(subschema, dict):
                continue
            branch_active = active
            ref = subschema.get("$ref")
            if isinstance(ref, str):
                def_name = self._ref_target_name(ref)
                if def_name is not None and def_name in self._defs:
                    if def_name in active:
                        raise ValueError(
                            f"allOf/$ref cycle detected in '{self._source.name}': "
                            f"definition '{def_name}' is reached again while it is "
                            "still being expanded (definitions on the cycle: "
                            f"{', '.join(sorted(active))}). Break the cycle in the schema."
                        )
                    branch_active = active | {def_name}
            resolved = self._merge_allof(self._deref(subschema), branch_active)
            properties.update(resolved.get("properties", {}))
            required.extend(resolved.get("required", []))
            if not merged.get("description") and resolved.get("description"):
                merged["description"] = resolved["description"]
        merged["properties"] = properties
        merged["required"] = list(dict.fromkeys(required))
        if properties:
            merged.setdefault("type", "object")
        return merged

    # ------------------------------------------------------------------ #
    # Model building
    # ------------------------------------------------------------------ #

    def _build_model(
        self,
        model_name: str,
        schema: dict[str, Any],
        current_depth: int,
        is_root: bool,
        is_inline: bool,
        source_ref: str,
    ) -> dict[str, Any]:
        schema = self._normalize_object(schema, model_name)
        properties = schema.get("properties", {})
        required = [r for r in schema.get("required", []) if isinstance(r, str)]

        fields: list[dict[str, Any]] = []
        raw_by_field: dict[str, str] = {}
        used: set[str] = set()
        for prop_name in properties:
            subschema = properties[prop_name]
            if not isinstance(subschema, dict):
                self._notes.append(
                    f"Property '{model_name}.{prop_name}' is not an object schema; skipped."
                )
                continue
            field_draft = self._build_field(prop_name, subschema, model_name, current_depth, used)
            if field_draft is not None:
                fields.append(field_draft)
                raw_by_field[field_draft["name"]] = prop_name

        identity = (
            [] if is_inline else self._select_identity(model_name, fields, required, raw_by_field)
        )
        for required_name in required:
            field_name = _find_field(fields, raw_by_field, required_name)
            if field_name is not None and field_name not in identity:
                self._notes.append(
                    f"Required property '{required_name}' on '{model_name}' rendered "
                    "optional (the Optionality Law: required = identity, nothing else)."
                )

        kind = "root" if is_root else ("entity" if identity else "component")
        if is_root and not identity:
            fields.insert(0, document_reference_field())
            identity = ["document_reference"]
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    field="document_reference",
                    kind="missing_identity",
                    note=(
                        "No required identity-like property on the root; synthesized "
                        "'document_reference'. Replace with a real printed identifier."
                    ),
                )
            )
        elif not identity and not is_inline:
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    kind="missing_identity",
                    note=(
                        "No required id/name/number-named string property; demoted to "
                        "component (never invent ids)."
                    ),
                )
            )

        for field_draft in fields:
            if field_draft["name"] in identity:
                field_draft["role"] = "identity"
                field_draft["is_list"] = False
                field_draft.pop("_target", None)
                if len(field_draft.get("examples", [])) < 2:
                    self._gaps.append(
                        SpecGap(
                            model=model_name,
                            field=field_draft["name"],
                            kind="missing_examples",
                            note="Add 2-5 verbatim examples from a real document "
                            "(schema 'examples' are harvested when present).",
                        )
                    )

        docstring = str(schema.get("description") or "")
        if not docstring:
            docstring = placeholder_docstring(model_name)
            self._gaps.append(
                SpecGap(
                    model=model_name,
                    kind="missing_docstring",
                    note="No 'description' in the schema.",
                )
            )

        return {
            "name": model_name,
            "kind": kind,
            "docstring": docstring,
            "identity_fields": identity,
            "max_instances": None,
            "fields": fields,
            "canonical_home": None,
            "provenance": "ontology",
            "source_ref": f"{self._source.name}{source_ref}",
        }

    def _build_field(
        self,
        prop_name: str,
        subschema: dict[str, Any],
        model_name: str,
        current_depth: int,
        used: set[str],
    ) -> dict[str, Any] | None:
        base_name = sanitize_field_name(prop_name)
        field_name = base_name
        counter = 2
        while field_name in used:
            field_name = f"{base_name}_{counter}"
            counter += 1

        subschema = self._normalize_object(
            self._merge_allof(subschema), f"{model_name}.{prop_name}"
        )
        is_list = False
        max_items: int | None = None
        if _json_type(subschema) == "array":
            is_list = True
            raw_max = subschema.get("maxItems")
            max_items = int(raw_max) if isinstance(raw_max, int) else None
            if max_items == 1:
                is_list = False
            items = subschema.get("items")
            element = items if isinstance(items, dict) else {}
            element = self._normalize_object(
                self._merge_allof(element), f"{model_name}.{prop_name}[]"
            )
            description = str(subschema.get("description") or element.get("description") or "")
            examples = _examples(subschema) or _examples(element)
            subschema, resolved_type, target = self._resolve_element(
                prop_name, element, model_name, current_depth
            )
        else:
            description = str(subschema.get("description") or "")
            examples = _examples(subschema)
            subschema, resolved_type, target = self._resolve_element(
                prop_name, subschema, model_name, current_depth
            )

        if resolved_type is None:
            return None

        field_draft: dict[str, Any] = {
            "name": field_name,
            "type": resolved_type,
            "is_list": is_list,
            "description": description,
            "examples": examples,
            "role": "property",
        }
        if target is not None:
            field_draft["_target"] = target
            if max_items is not None and max_items > 1:
                self._bounds[target] = max(self._bounds.get(target, 0), max_items)
        else:
            if subschema.get("enum") is not None:
                field_draft["normalizer"] = "enum"
            if max_items is not None and max_items > 1:
                field_draft["description"] = _join_sentences(
                    description, f"At most {max_items} values per document."
                )

        used.add(field_name)
        return field_draft

    def _resolve_element(
        self,
        prop_name: str,
        subschema: dict[str, Any],
        model_name: str,
        current_depth: int,
    ) -> tuple[dict[str, Any], str | None, str | None]:
        """Resolve one scalar/enum/model element schema.

        Returns ``(schema, spec_type, target_model_name)`` — ``spec_type`` is
        ``None`` when the field must be dropped (pruned target), and
        ``target_model_name`` is set only for model-typed fields.
        """
        ref = subschema.get("$ref")
        if isinstance(ref, str):
            def_name = self._ref_target_name(ref)
            if def_name is None or def_name not in self._defs:
                self._notes.append(
                    f"'{model_name}.{prop_name}': unresolvable $ref '{ref}'; typed as str."
                )
                return subschema, "str", None
            def_name, definition = self._follow_def_aliases(def_name)
            resolved = self._merge_allof(definition, frozenset({def_name}))

            # Only object definitions become models: enum defs route to an
            # EnumSpec, scalar defs map through the type/format tables (an
            # empty model would silently drop the definition's constraint).
            enum_values = resolved.get("enum")
            if isinstance(enum_values, list) and enum_values:
                return resolved, self._ensure_def_enum(def_name, enum_values), None
            json_type = _json_type(resolved)
            if json_type == "string" and "properties" not in resolved:
                return resolved, _FORMAT_MAP.get(str(resolved.get("format")), "str"), None
            if json_type in ("integer", "number", "boolean"):
                return resolved, _JSON_TYPE_MAP[json_type], None

            target = self._ensure_def_model(def_name, current_depth + 1, model_name, prop_name)
            if target is None:
                return subschema, None, None
            return subschema, target, target

        enum_values = subschema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_name = self._ensure_enum(prop_name, enum_values, model_name)
            return subschema, enum_name, None

        if _is_object_schema(subschema):
            target = self._ensure_inline_model(prop_name, subschema, current_depth + 1, model_name)
            if target is None:
                return subschema, None, None
            return subschema, target, target

        json_type = _json_type(subschema)
        if json_type == "string":
            format_value = subschema.get("format")
            return subschema, _FORMAT_MAP.get(str(format_value), "str"), None
        return subschema, _JSON_TYPE_MAP.get(json_type or "", "str"), None

    def _follow_def_aliases(self, def_name: str) -> tuple[str, dict[str, Any]]:
        """Follow top-level ``$ref`` alias chains between definitions.

        ``{"$defs": {"Alias": {"$ref": "#/$defs/Currency"}}}`` resolves to
        ``("Currency", <its schema>)``. Cycle-safe: a chain that loops stops at
        the last new definition.
        """
        seen = {def_name}
        definition = self._defs.get(def_name)
        while isinstance(definition, dict):
            ref = definition.get("$ref")
            if not isinstance(ref, str):
                break
            next_name = self._ref_target_name(ref)
            if next_name is None or next_name not in self._defs or next_name in seen:
                break
            seen.add(next_name)
            def_name = next_name
            definition = self._defs[def_name]
        return def_name, definition if isinstance(definition, dict) else {}

    def _ensure_def_enum(self, def_name: str, values: list[Any]) -> str:
        """EnumSpec for an enum-carrying ``$defs`` entry, under the def's name."""
        if def_name in self._def_enums:
            return self._def_enums[def_name]
        members = [str(v) for v in values if v is not None]
        name = unique_name(sanitize_class_name(def_name), self._taken_names)
        self._def_enums[def_name] = name
        self._enum_drafts.append(
            {
                "name": name,
                "members": members,
                "synonyms": {},
                "include_other": True,
            }
        )
        return name

    def _ensure_def_model(
        self, def_name: str, target_depth: int, parent: str, prop_name: str
    ) -> str | None:
        if def_name in self._def_models:
            return self._def_models[def_name]
        if not class_passes_filters(def_name, self._include, self._exclude):
            self._notes.append(
                f"'{parent}.{prop_name}' dropped: '{def_name}' is pruned by include/exclude."
            )
            return None
        if target_depth > self._depth:
            self._notes.append(
                f"'{parent}.{prop_name}' dropped: '{def_name}' would sit at depth "
                f"{target_depth}, beyond --depth {self._depth}."
            )
            return None
        model_name = unique_name(sanitize_class_name(def_name), self._taken_names)
        self._def_models[def_name] = model_name
        definition = self._defs[def_name]
        if not isinstance(definition, dict):
            definition = {}
        self._queue.append(
            (
                model_name,
                definition,
                target_depth,
                False,
                False,
                f"#/$defs/{def_name}" if "$defs" in self._schema else f"#/definitions/{def_name}",
            )
        )
        return model_name

    def _ensure_inline_model(
        self, prop_name: str, schema: dict[str, Any], target_depth: int, parent: str
    ) -> str | None:
        if target_depth > self._depth:
            self._notes.append(
                f"'{parent}.{prop_name}' dropped: inline object would sit at depth "
                f"{target_depth}, beyond --depth {self._depth}."
            )
            return None
        base = sanitize_class_name(prop_name)
        if base in self._taken_names:
            base = sanitize_class_name(f"{parent} {prop_name}")
        model_name = unique_name(base, self._taken_names)
        self._queue.append(
            (model_name, schema, target_depth, False, True, f"#/{parent}.{prop_name}")
        )
        return model_name

    def _ensure_enum(self, prop_name: str, values: list[Any], parent: str) -> str:
        members = [str(v) for v in values if v is not None]
        base = sanitize_class_name(prop_name)
        if base in self._taken_names:
            base = sanitize_class_name(f"{parent} {prop_name}")
        name = unique_name(base, self._taken_names)
        self._enum_drafts.append(
            {
                "name": name,
                "members": members,
                "synonyms": {},
                # JSON Schema enums validate closed, but extraction still needs
                # the OTHER safety net: every generated enum normalizer must
                # coerce-and-warn, never raise (R17, the never-reject law).
                "include_other": True,
            }
        )
        return name

    def _select_identity(
        self,
        model_name: str,
        fields: list[dict[str, Any]],
        required: list[str],
        raw_by_field: dict[str, str],
    ) -> list[str]:
        candidates: list[tuple[int, str]] = []
        for field_draft in fields:
            raw = raw_by_field.get(field_draft["name"], field_draft["name"])
            if raw not in required:
                continue
            if field_draft["type"] != "str" or field_draft["is_list"]:
                continue
            if field_draft.get("_target") is not None:
                continue
            rank = identity_ladder_rank(field_draft["name"])
            if rank is not None:
                candidates.append((rank, field_draft["name"]))
        if not candidates:
            return []
        return [min(candidates)[1]]

    # ------------------------------------------------------------------ #
    # Post passes
    # ------------------------------------------------------------------ #

    def _assign_edge_roles(self) -> None:
        kinds = {m["name"]: m["kind"] for m in self._models}
        for model in self._models:
            for field_draft in model["fields"]:
                target = field_draft.pop("_target", None)
                if target is None:
                    continue
                if kinds.get(target) == "component":
                    continue  # components nest as plain property fields
                field_draft["role"] = "edge"
                field_draft["edge_label"] = normalize_edge_label(field_draft["name"], target=target)

    def _apply_cardinality_bounds(self) -> None:
        by_name = {m["name"]: m for m in self._models}
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


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _json_type(schema: dict[str, Any]) -> str | None:
    """The schema's JSON type, unwrapping nullable-union lists."""
    value = schema.get("type")
    if isinstance(value, list):
        non_null = [t for t in value if t != "null"]
        value = non_null[0] if non_null else None
    if value is None and "properties" in schema:
        return "object"
    return str(value) if value is not None else None


def _is_object_schema(schema: dict[str, Any]) -> bool:
    return _json_type(schema) == "object"


def _examples(schema: dict[str, Any]) -> list[str]:
    values = schema.get("examples")
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if v is not None][:5]


def _find_field(
    fields: list[dict[str, Any]], raw_by_field: dict[str, str], raw_name: str
) -> str | None:
    for field_draft in fields:
        if raw_by_field.get(field_draft["name"]) == raw_name:
            return str(field_draft["name"])
    return None


def _join_sentences(base: str, addition: str) -> str:
    if not base:
        return addition
    if not base.rstrip().endswith((".", "!", "?")):
        base = f"{base.rstrip()}."
    return f"{base.rstrip()} {addition}"
