"""
TemplateSpec — the intermediate representation (IR) for template generation.

The IR sits on the hard boundary between SPEC induction (LLM structured output
or deterministic ontology parsing) and deterministic rendering: everything the
renderer emits is derived from a validated ``TemplateSpec``. The intrinsic
validators below make whole families of rulebook violations *unrepresentable*
(see ``docs/fundamentals/schema-definition/``):

- ``closed_catalog`` without ``reference`` (relationships.md, closed catalogs
  are reference edges by definition);
- ``edge_label`` on non-edge fields, or edges without a label;
- list-valued / enum-typed / model-typed identity fields
  (field-definitions.md: identity is required, scalar, and concise);
- components carrying identity fields or cardinality bounds
  (entities-vs-components.md);
- entities with zero or more than two identity fields;
- dangling ``type`` / ``canonical_home`` references and duplicate names.

The SPEC round-trips through YAML (``to_yaml`` / ``from_yaml``) so users can
hand-edit one line and re-render instead of editing generated Python.
"""

from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScalarType = Literal["str", "int", "float", "bool", "date", "datetime"]
"""Scalar field types the renderer knows how to emit."""

SCALAR_TYPES: frozenset[str] = frozenset(("str", "int", "float", "bool", "date", "datetime"))
"""String forms of :data:`ScalarType` — any other ``FieldSpec.type`` value must
name a declared :class:`EnumSpec` or :class:`ModelSpec`."""

MAX_IDENTITY_FIELDS = 2
"""Identity is one scalar field, two at most (field-definitions.md)."""

MAX_FIELD_EXAMPLES = 5
"""Identity/field examples are capped at 5 (field-definitions.md)."""


class EnumSpec(BaseModel):
    """A controlled vocabulary rendered as a ``(str, Enum)`` class."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="PascalCase enum class name.")
    members: list[str] = Field(
        min_length=1,
        description="Printed member values, e.g. ['Invoice', 'Credit Note'].",
    )
    synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Member -> document phrases that map to it ('plate-plate' -> 'Plate-Plate').",
    )
    include_other: bool = Field(
        default=True,
        description="Append the OTHER safety-net member (validation.md enum guidance).",
    )

    @model_validator(mode="after")
    def _check_synonyms_reference_members(self) -> EnumSpec:
        unknown = set(self.synonyms) - set(self.members)
        if unknown:
            raise ValueError(
                f"EnumSpec '{self.name}': synonyms reference unknown members {sorted(unknown)}"
            )
        return self


class FieldSpec(BaseModel):
    """One field of a :class:`ModelSpec` (identity, property, or edge)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="snake_case field name (post naming.py sanitation).")
    type: str = Field(
        default="str",
        description="A scalar type name, or the name of a declared EnumSpec/ModelSpec.",
    )
    is_list: bool = False
    description: str = Field(
        default="",
        description="1-3 sentences: LOOK-FOR locator + one normalization rule.",
    )
    examples: list[str] = Field(
        default_factory=list,
        max_length=MAX_FIELD_EXAMPLES,
        description="Verbatim document/ontology-derived example strings.",
    )
    role: Literal["identity", "property", "edge"] = "property"
    # --- edge-only ---
    edge_label: str | None = Field(
        default=None,
        description="ALL_CAPS verb phrase (post naming.normalize_edge_label). Edges only.",
    )
    reference: bool = Field(
        default=False,
        description="Identity-only link -> json_schema_extra['graph_reference'].",
    )
    closed_catalog: bool = Field(
        default=False,
        description="Fixed catalog -> json_schema_extra['reference_closed_catalog'].",
    )
    # --- property-only ---
    normalizer: Literal["none", "enum", "currency", "numeric", "string_list"] = "none"
    unit: str | None = None
    evidence: list[str] = Field(
        default_factory=list,
        description="Induction-only source quotes justifying the field (dropped at render).",
    )

    @field_validator("examples", "evidence", mode="before")
    @classmethod
    def _coerce_scalar_items_to_str(cls, v: Any) -> Any:
        """Accept unquoted YAML scalars ('examples: [1, 2]') as verbatim strings."""
        if isinstance(v, list):
            return [str(item) if isinstance(item, int | float | bool) else item for item in v]
        return v

    @model_validator(mode="after")
    def _check_intrinsics(self) -> FieldSpec:
        if self.closed_catalog and not self.reference:
            raise ValueError(
                f"Field '{self.name}': closed_catalog=True requires reference=True "
                "(relationships.md: closed catalogs are reference edges)"
            )
        if self.role == "edge" and not self.edge_label:
            raise ValueError(f"Field '{self.name}': role='edge' requires an edge_label")
        if self.role != "edge" and self.edge_label is not None:
            raise ValueError(f"Field '{self.name}': edge_label is only allowed when role='edge'")
        if self.role != "edge" and (self.reference or self.closed_catalog):
            raise ValueError(f"Field '{self.name}': reference/closed_catalog markers are edge-only")
        if self.role == "identity":
            if self.is_list:
                raise ValueError(
                    f"Field '{self.name}': identity fields must be scalar, not lists "
                    "(field-definitions.md)"
                )
            if self.type not in SCALAR_TYPES:
                raise ValueError(
                    f"Field '{self.name}': identity fields must be scalar-typed, "
                    f"not enum/model-typed ('{self.type}')"
                )
            if self.normalizer != "none":
                raise ValueError(
                    f"Field '{self.name}': identity fields are copied verbatim and "
                    "take no normalizer"
                )
        if self.role == "edge" and self.normalizer != "none":
            raise ValueError(f"Field '{self.name}': normalizers are property-only")
        return self


class ModelSpec(BaseModel):
    """One Pydantic model of the template (root, entity, or component)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="PascalCase class name (post naming.py sanitation).")
    kind: Literal["entity", "component", "root"]
    docstring: str = Field(
        description="First ~240 chars carry IS / IS-NOT / cardinality / routing "
        "(best-practices.md)."
    )
    identity_fields: list[str] = Field(
        default_factory=list,
        description="Subset of field names -> graph_id_fields. Empty iff component.",
    )
    max_instances: int | None = Field(
        default=None,
        ge=1,
        description="-> graph_max_instances; only when the docstring states cardinality.",
    )
    fields: list[FieldSpec]
    canonical_home: str | None = Field(
        default=None,
        description="'ParentModel.field_name' — the ONE path where full detail lives.",
    )
    provenance: Literal["induced", "ontology", "gapfill", "user"] = "induced"
    source_ref: str | None = Field(
        default=None,
        description="Class IRI / document quote — kept in the YAML for audit.",
    )

    @model_validator(mode="after")
    def _check_intrinsics(self) -> ModelSpec:
        if not self.docstring.strip():
            raise ValueError(f"Model '{self.name}': docstring must be non-empty")

        field_names = [f.name for f in self.fields]
        duplicates = {n for n in field_names if field_names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Model '{self.name}': duplicate field names {sorted(duplicates)}")

        if self.kind == "component":
            if self.identity_fields:
                raise ValueError(
                    f"Model '{self.name}': components carry no identity_fields "
                    "(entities-vs-components.md)"
                )
            if self.max_instances is not None:
                raise ValueError(
                    f"Model '{self.name}': components take no max_instances "
                    "(cardinality bounds apply to discovered entities only)"
                )
        else:  # entity | root
            if not 1 <= len(self.identity_fields) <= MAX_IDENTITY_FIELDS:
                raise ValueError(
                    f"Model '{self.name}': {self.kind} requires 1-{MAX_IDENTITY_FIELDS} "
                    f"identity fields, got {len(self.identity_fields)}"
                )
            fields_by_name = {f.name: f for f in self.fields}
            for id_name in self.identity_fields:
                field = fields_by_name.get(id_name)
                if field is None:
                    raise ValueError(
                        f"Model '{self.name}': identity field '{id_name}' is not declared in fields"
                    )
                if field.role != "identity":
                    raise ValueError(
                        f"Model '{self.name}': identity field '{id_name}' must have "
                        f"role='identity', got '{field.role}'"
                    )

        # The reverse direction: a role='identity' field outside identity_fields is
        # either a stray on a component (unrepresentable identity) or an unlisted id.
        listed = set(self.identity_fields)
        stray = [f.name for f in self.fields if f.role == "identity" and f.name not in listed]
        if stray:
            raise ValueError(
                f"Model '{self.name}': fields {stray} have role='identity' but are not "
                "listed in identity_fields"
            )
        return self


class SpecGap(BaseModel):
    """A declared hole in the SPEC that gap-fill or the user can close."""

    model_config = ConfigDict(extra="forbid")

    model: str
    field: str | None = None
    kind: Literal[
        "missing_docstring",
        "missing_examples",
        "missing_identity",
        "ambiguous_kind",
        "missing_description",
        "missing_edge_label",
    ]
    note: str = ""


class TemplateSpec(BaseModel):
    """The full template IR: enums + models + root, YAML round-trippable."""

    model_config = ConfigDict(extra="forbid")

    module_docstring: str = Field(
        default="",
        description="Domain summary + key entities/relationships block.",
    )
    root: str = Field(description="Name of the root ModelSpec.")
    enums: list[EnumSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(min_length=1)
    needs_root_list_dedup: list[str] = Field(
        default_factory=list,
        description="Root list fields that get the dedup model_validator (validation.md).",
    )
    generator: dict[str, Any] = Field(
        default_factory=dict,
        description="Audit-only: generator version, source paths, timestamps.",
    )

    @model_validator(mode="after")
    def _check_intrinsics(self) -> TemplateSpec:
        self._check_unique_names()
        self._check_root()
        self._check_references_resolve()
        return self

    def _check_unique_names(self) -> None:
        model_names = [m.name for m in self.models]
        enum_names = [e.name for e in self.enums]
        all_names = model_names + enum_names
        duplicates = {n for n in all_names if all_names.count(n) > 1}
        if duplicates:
            raise ValueError(f"TemplateSpec: duplicate model/enum names {sorted(duplicates)}")

    def _check_root(self) -> None:
        roots = [m.name for m in self.models if m.kind == "root"]
        if len(roots) != 1:
            raise ValueError(
                f"TemplateSpec: exactly one model must have kind='root', got {roots or 'none'}"
            )
        if self.root != roots[0]:
            raise ValueError(
                f"TemplateSpec: root='{self.root}' does not name the root model '{roots[0]}'"
            )

    def _check_references_resolve(self) -> None:
        model_names = {m.name for m in self.models}
        enum_names = {e.name for e in self.enums}
        known_types = SCALAR_TYPES | model_names | enum_names

        for model in self.models:
            for field in model.fields:
                if field.type not in known_types:
                    raise ValueError(
                        f"Model '{model.name}': field '{field.name}' has unresolved "
                        f"type '{field.type}'"
                    )
                if field.role == "edge" and field.type not in model_names:
                    raise ValueError(
                        f"Model '{model.name}': edge field '{field.name}' must target a "
                        f"declared model, got '{field.type}'"
                    )
            if model.canonical_home is not None:
                self._check_canonical_home(model)

        root_model = next(m for m in self.models if m.kind == "root")
        root_fields = {f.name: f for f in root_model.fields}
        for field_name in self.needs_root_list_dedup:
            dedup_field = root_fields.get(field_name)
            if dedup_field is None:
                raise ValueError(
                    f"TemplateSpec: needs_root_list_dedup names unknown root field '{field_name}'"
                )
            if not dedup_field.is_list:
                raise ValueError(
                    f"TemplateSpec: needs_root_list_dedup field '{field_name}' is not a list"
                )

    def _check_canonical_home(self, model: ModelSpec) -> None:
        home = model.canonical_home or ""
        parent_name, _, field_name = home.partition(".")
        if not parent_name or not field_name:
            raise ValueError(
                f"Model '{model.name}': canonical_home '{home}' must be 'ParentModel.field_name'"
            )
        parent = next((m for m in self.models if m.name == parent_name), None)
        if parent is None:
            raise ValueError(
                f"Model '{model.name}': canonical_home parent '{parent_name}' is not a "
                "declared model"
            )
        if field_name not in {f.name for f in parent.fields}:
            raise ValueError(
                f"Model '{model.name}': canonical_home field '{parent_name}.{field_name}' "
                "does not exist"
            )

    def to_yaml(self) -> str:
        """Serialize the SPEC to YAML (the ``--spec-out`` format)."""
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> TemplateSpec:
        """Parse and validate a SPEC from YAML text (the ``from-spec`` entry point)."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("TemplateSpec YAML must contain a mapping at the top level")
        return cls.model_validate(data)
