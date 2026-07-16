"""
Deterministic renderer: ``TemplateSpec`` -> Python template module source.

Plain string assembly — no LLM, no Jinja. The renderer can only emit
constructs that comply with ``docs/fundamentals/schema-definition/``, so whole
rule families hold *by construction*:

- mandated file order (template-basics.md): module docstring -> imports ->
  ``edge()`` helper -> helpers -> enums -> components -> entities -> root LAST;
- the Optionality Law (field-definitions.md): identity fields are the only
  ``...``-required fields; every other field is ``Optional[...] = None`` or
  ``default_factory=list``;
- list edges always carry ``default_factory=list`` (relationships.md);
- validators come verbatim from the snippet library, which contains no
  ``raise`` (validation.md's never-reject law);
- enum fields default to ``OTHER`` (when the enum carries it) and get the
  ``mode="before"`` normalize validator (validation.md).

Models are topologically sorted within the mandated groups (edge targets
before their sources); cycles fall back to quoted forward references plus
``Model.model_rebuild()`` lines appended after the class definitions.

Compatibility note (documented decision): the repo floor is Python 3.10 while
``typing.Self`` needs 3.11+. The shipped canon templates never import ``Self``,
so the emitted typing import carries only the names the module actually uses;
when the root-list dedup validator (the one snippet returning ``Self``) is
emitted, ``Self`` is imported from ``typing_extensions`` — the exact idiom of
validation.md's "Deduplicate root-level list by key" section (and a pydantic
dependency, hence always installed).

Determinism: the same SPEC always renders byte-identical source. All iteration
follows SPEC list order; nothing depends on set/dict iteration order.
``FieldSpec.evidence`` is induction-only audit data and is never rendered.
"""

from __future__ import annotations

import json
import pprint
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping

from .naming import to_snake_case
from .snippets import (
    CURRENCY_VALIDATOR_TEMPLATE,
    EDGE_HELPER,
    ENUM_FIELD_VALIDATOR_TEMPLATE,
    ENUM_LIST_FIELD_VALIDATOR_TEMPLATE,
    IMPORT_BLOCK,
    LOGGER_SETUP,
    NORMALIZE_ENUM_HELPER,
    NUMERIC_VALIDATOR_TEMPLATE,
    OPTIONAL_IMPORT_DATETIME,
    OPTIONAL_IMPORT_ENUM,
    OPTIONAL_IMPORT_LOGGING,
    OPTIONAL_IMPORT_RE,
    ROOT_LIST_DEDUP_TEMPLATE,
    STR_METHOD_TEMPLATE,
    STRING_LIST_VALIDATOR_TEMPLATE,
)
from .spec import SCALAR_TYPES, EnumSpec, FieldSpec, ModelSpec, TemplateSpec
from .verify import synthesize_sample_plan

MAX_LINE = 100
"""Emitted lines aim for the repo's ruff line length (long string literals may
exceed it; ruff ignores E501 and the formatter never splits strings)."""

_TODO_EXAMPLES = "# TODO(docling-graph): add 2-5 verbatim examples"
_TODO_DESCRIPTION = "# TODO(docling-graph): add a 1-3 sentence LOOK-FOR description"

_SCALAR_ANNOTATIONS: dict[str, str] = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "date": "date",
    "datetime": "datetime",
}


@dataclass
class _Usage:
    """Import/helper usage tracked while emitting the module body."""

    uses_list: bool = False
    uses_optional: bool = False
    uses_datetime: bool = False
    uses_enum: bool = False
    uses_re: bool = False
    uses_logging: bool = False
    uses_self: bool = False
    uses_field_validator: bool = False
    uses_model_validator: bool = False


def render_template(spec: TemplateSpec) -> str:
    """Render a validated SPEC into a runnable Pydantic template module."""
    enums_by_name = {enum.name: enum for enum in spec.enums}
    models_by_name = {model.name: model for model in spec.models}
    ordered = _ordered_models(spec)
    usage = _Usage()

    if spec.enums:
        # Enum classes + the _normalize_enum helper (uses Type/Enum/re/logger).
        usage.uses_enum = True
        usage.uses_re = True
        usage.uses_logging = True

    defined: set[str] = set(enums_by_name)
    rebuild: list[str] = []
    model_blocks: list[str] = []
    for model in ordered:
        block, has_forward_ref = _render_model(
            model, spec, enums_by_name, models_by_name, defined, usage
        )
        model_blocks.append(block)
        defined.add(model.name)
        if has_forward_ref:
            rebuild.append(model.name)

    chunks: list[str] = []
    chunks.append(_banner("Docling Graph edge helper") + "\n" + EDGE_HELPER.rstrip("\n"))
    if spec.enums:
        chunks.append(_banner("Helpers") + "\n" + NORMALIZE_ENUM_HELPER.rstrip("\n"))
        enum_chunks = [_render_enum(enum) for enum in spec.enums]
        enum_chunks[0] = _banner("Enums") + "\n" + enum_chunks[0]
        chunks.extend(enum_chunks)

    components = [m for m in ordered if m.kind == "component"]
    entities = [m for m in ordered if m.kind == "entity"]
    sections = (
        ("Components", len(components)),
        ("Entities", len(entities)),
        ("Root document", 1),
    )
    index = 0
    for title, count in sections:
        for position in range(count):
            block = model_blocks[index]
            if position == 0:
                block = _banner(title) + "\n" + block
            chunks.append(block)
            index += 1

    if rebuild:
        chunks.append("\n".join(f"{name}.model_rebuild()" for name in rebuild))
    chunks.append(_render_footer(spec))

    header = _render_module_docstring(spec, ordered) + "\n\n" + _render_imports(usage)
    if usage.uses_logging:
        # One blank line between imports and the logger assignment (isort),
        # matching the insurance_terms.py canon preamble.
        header += "\n\n" + LOGGER_SETUP
    return header + "\n\n\n" + "\n\n\n".join(chunks) + "\n"


# ---------------------------------------------------------------------------
# Model ordering
# ---------------------------------------------------------------------------


def _ordered_models(spec: TemplateSpec) -> list[ModelSpec]:
    """Mandated group order (components -> entities -> root) with a topological
    sort inside each group: edge/nested targets come before their sources so
    most references resolve without forward refs. Cycles keep SPEC order and
    are later broken with quoted forward references."""
    model_names = {model.name for model in spec.models}
    components = [m for m in spec.models if m.kind == "component"]
    entities = [m for m in spec.models if m.kind == "entity"]
    root = next(m for m in spec.models if m.kind == "root")

    def order_group(group: list[ModelSpec]) -> list[ModelSpec]:
        remaining = list(group)
        ordered: list[ModelSpec] = []
        while remaining:
            remaining_names = {m.name for m in remaining}
            pick = next(
                (
                    m
                    for m in remaining
                    if not (_model_dependencies(m, model_names) & (remaining_names - {m.name}))
                ),
                remaining[0],  # cycle within the group: keep SPEC order
            )
            ordered.append(pick)
            remaining.remove(pick)
        return ordered

    return order_group(components) + order_group(entities) + [root]


def _model_dependencies(model: ModelSpec, model_names: set[str]) -> set[str]:
    """Names of models referenced by this model's field annotations."""
    return {f.type for f in model.fields if f.type in model_names}


# ---------------------------------------------------------------------------
# Module docstring + imports
# ---------------------------------------------------------------------------


def _first_sentence(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    sentence = collapsed.split(". ", 1)[0].rstrip(".")
    if len(sentence) > limit:
        sentence = sentence[: limit - 3].rstrip() + "..."
    return sentence + "."


def _provenance_line(generator: Mapping[str, Any]) -> str:
    """Audit line built ONLY from what ``spec.generator`` carries (no invented
    timestamps): version, source path(s), creation stamp, spec path."""
    line = "Generated by docling-graph template generation"
    version = generator.get("version")
    if version:
        line += f" v{version}"
    sources = generator.get("sources") or generator.get("source")
    if isinstance(sources, str):
        sources = [sources]
    if isinstance(sources, list) and sources:
        line += f" from {', '.join(str(s) for s in sources)}"
    stamp = generator.get("created") or generator.get("timestamp") or generator.get("date")
    if stamp:
        line += f" on {stamp}"
    line += "."
    spec_path = generator.get("spec") or generator.get("spec_out")
    if spec_path:
        line += f" Spec: {spec_path}."
    return line


def _escape_docstring_text(text: str) -> str:
    """Make free text safe inside an emitted triple-quoted docstring.

    Backslashes are doubled first ('\\U'/'\\x' sequences would otherwise be
    invalid escapes that fail ``ast.parse``), then embedded triple quotes are
    substituted.
    """
    return text.replace("\\", "\\\\").replace('"""', "'''")


def _render_module_docstring(spec: TemplateSpec, ordered: list[ModelSpec]) -> str:
    paragraphs: list[str] = []
    summary = " ".join(spec.module_docstring.split())
    if summary:
        # Word-boundary wrapping only: breaking inside a token would corrupt
        # hyphenated names and split escaped-backslash pairs.
        paragraphs.append(
            textwrap.fill(
                summary, width=MAX_LINE - 1, break_long_words=False, break_on_hyphens=False
            )
        )

    named = [m for m in ordered if m.kind == "root"] + [m for m in ordered if m.kind == "entity"]
    entity_lines = ["Key entities:"]
    for model in named:
        suffix = " (root)" if model.kind == "root" else ""
        entity_lines.append(f"- {model.name}{suffix}: {_first_sentence(model.docstring)}")
    paragraphs.append("\n".join(entity_lines))

    edge_lines = ["Key relationships:"]
    for model in ordered:
        for field in model.fields:
            if field.role != "edge":
                continue
            marker = " (by reference)" if field.reference else ""
            edge_lines.append(f"- {model.name} --{field.edge_label}--> {field.type}{marker}")
    if len(edge_lines) > 1:
        paragraphs.append("\n".join(edge_lines))

    # Provenance carries paths: never wrap inside a token (mid-path breaks at
    # hyphens would corrupt copy-pasted paths).
    paragraphs.append(
        textwrap.fill(
            _provenance_line(spec.generator),
            width=MAX_LINE - 1,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    body = _escape_docstring_text("\n\n".join(paragraphs))
    return f'"""\n{body}\n"""'


def _render_imports(usage: _Usage) -> str:
    stdlib: list[str] = []
    if usage.uses_logging:
        stdlib.append(OPTIONAL_IMPORT_LOGGING)
    if usage.uses_re:
        stdlib.append(OPTIONAL_IMPORT_RE)
    if usage.uses_datetime:
        stdlib.append(OPTIONAL_IMPORT_DATETIME)
    if usage.uses_enum:
        stdlib.append(OPTIONAL_IMPORT_ENUM)

    # Subset of the canonical typing/pydantic lines (template-basics.md),
    # filtered to what the emitted module actually uses. `Self` deliberately
    # never comes from `typing` (3.11+); see the module docstring.
    typing_line, pydantic_line = (line for line in IMPORT_BLOCK.strip().splitlines())
    typing_used = {
        "Any": True,  # edge() helper
        "List": usage.uses_list,
        "Optional": usage.uses_optional,
        "Type": usage.uses_enum,
    }
    typing_names = [name for name in _imported_names(typing_line) if typing_used.get(name, False)]
    stdlib.append(f"from typing import {', '.join(typing_names)}")

    pydantic_used = {
        "BaseModel": True,
        "ConfigDict": True,
        "Field": True,
        "field_validator": usage.uses_field_validator,
        "model_validator": usage.uses_model_validator,
    }
    pydantic_names = [
        name for name in _imported_names(pydantic_line) if pydantic_used.get(name, False)
    ]
    third_party = [f"from pydantic import {', '.join(pydantic_names)}"]
    if usage.uses_self:
        third_party.append("from typing_extensions import Self")

    return "\n".join(stdlib) + "\n\n" + "\n".join(third_party)


def _imported_names(import_line: str) -> list[str]:
    """Names of a ``from x import a, b`` line, in source order."""
    _, _, names = import_line.partition(" import ")
    return [name.strip() for name in names.split(",")]


def _banner(title: str) -> str:
    rule = "# " + "-" * 77
    return f"{rule}\n# {title}\n{rule}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def _enum_member_name(value: str, used: set[str]) -> str:
    base = to_snake_case(value).upper() or "MEMBER"
    if base[0].isdigit():
        base = f"VALUE_{base}"
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def _render_enum(enum: EnumSpec) -> str:
    doc = f"Controlled vocabulary for {to_snake_case(enum.name)} values."
    if enum.include_other:
        doc += " Unmapped values normalize to OTHER."
    lines = [f"class {enum.name}(str, Enum):", f'    """{doc}"""', ""]
    used: set[str] = set()
    for member in enum.members:
        name = _enum_member_name(member, used)
        lines.append(f"    {name} = {json.dumps(member, ensure_ascii=False)}")
    if enum.include_other and "OTHER" not in used:
        lines.append('    OTHER = "Other"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _render_model(
    model: ModelSpec,
    spec: TemplateSpec,
    enums: dict[str, EnumSpec],
    models: dict[str, ModelSpec],
    defined: set[str],
    usage: _Usage,
) -> tuple[str, bool]:
    """Render one class block. Returns (source, needs_model_rebuild)."""
    has_forward_ref = False
    blocks: list[str] = [_render_class_docstring(model.docstring), _render_model_config(model)]

    for field in _emission_field_order(model):
        field_src, forward = _render_field(field, model, enums, models, defined, usage)
        has_forward_ref = has_forward_ref or forward
        blocks.append(field_src)

    blocks.extend(_render_field_validators(model, enums, usage))
    if model.kind == "root":
        blocks.extend(_render_dedup_validators(model, spec, models, usage))
    blocks.append(_render_str_method(model))

    body = "\n\n".join(blocks)
    return f"class {model.name}(BaseModel):\n{body}", has_forward_ref


def _emission_field_order(model: ModelSpec) -> list[FieldSpec]:
    """Identity fields (in identity_fields order) -> properties -> edges."""
    by_name = {field.name: field for field in model.fields}
    identity = [by_name[name] for name in model.identity_fields]
    properties = [
        field for field in model.fields if field.role == "property" and field not in identity
    ]
    edges = [field for field in model.fields if field.role == "edge"]
    return identity + properties + edges


def _render_class_docstring(text: str) -> str:
    clean = _escape_docstring_text(" ".join(text.split()))
    if len(clean) + 10 <= MAX_LINE and not clean.endswith('"'):
        return f'    """{clean}"""'
    # Never break inside a word: a split escaped-backslash pair or a broken
    # hyphenated token would change the docstring's content.
    wrapped = textwrap.wrap(
        clean, width=MAX_LINE - 4, break_long_words=False, break_on_hyphens=False
    )
    body = "\n".join(f"    {line}" for line in wrapped)
    return f'    """\n{body}\n    """'


def _render_model_config(model: ModelSpec) -> str:
    if model.kind == "component":
        return "    model_config = ConfigDict(is_entity=False)"
    kwargs = [f"graph_id_fields={json.dumps(model.identity_fields)}"]
    if model.max_instances is not None:
        kwargs.append(f"graph_max_instances={model.max_instances}")
    inline = f"    model_config = ConfigDict({', '.join(kwargs)})"
    if len(inline) <= MAX_LINE:
        return inline
    body = "".join(f"        {kwarg},\n" for kwarg in kwargs)
    return f"    model_config = ConfigDict(\n{body}    )"


def _render_field(
    field: FieldSpec,
    model: ModelSpec,
    enums: dict[str, EnumSpec],
    models: dict[str, ModelSpec],
    defined: set[str],
    usage: _Usage,
) -> tuple[str, bool]:
    """Render one field block. Returns (source, uses_forward_ref)."""
    forward = False
    if field.type in _SCALAR_ANNOTATIONS:
        annotation = _SCALAR_ANNOTATIONS[field.type]
        if field.type in ("date", "datetime"):
            usage.uses_datetime = True
    elif field.type in enums:
        annotation = field.type
    else:
        forward = field.type not in defined
        annotation = f'"{field.type}"' if forward else field.type

    args: list[str] = []
    func = "Field"
    if field.role == "identity":
        args.append("...,")
    elif field.role == "edge":
        func = "edge"
        args.append(f"label={json.dumps(field.edge_label)},")
        if field.is_list:
            args.append("default_factory=list,")
        if field.reference:
            args.append("reference=True,")
        if field.closed_catalog:
            args.append("closed_catalog=True,")
    elif field.is_list:
        args.append("default_factory=list,")
    elif field.type in enums and enums[field.type].include_other:
        args.append(f"{field.type}.OTHER,")
    else:
        args.append("None,")

    if field.is_list:
        annotation = f"List[{annotation}]"
        usage.uses_list = True
    elif field.role != "identity" and not (field.type in enums and enums[field.type].include_other):
        annotation = f"Optional[{annotation}]"
        usage.uses_optional = True

    description = _effective_description(field, enums)
    if description:
        args.append(_string_kwarg("description", description))
    else:
        args.append(f'description="",  {_TODO_DESCRIPTION}')

    if field.examples:
        args.append(_examples_kwarg(field.examples))
    elif field.role == "identity":
        args.append(f"examples=[],  {_TODO_EXAMPLES}")

    arg_lines = "\n".join(f"        {arg}" for arg in "\n".join(args).splitlines())
    return f"    {field.name}: {annotation} = {func}(\n{arg_lines}\n    )", forward


def _effective_description(field: FieldSpec, enums: dict[str, EnumSpec]) -> str:
    description = " ".join(field.description.split())
    enum = enums.get(field.type)
    if enum is not None and enum.synonyms:
        mappings = [
            f"'{synonym}' -> '{member}'"
            for member in enum.members
            for synonym in enum.synonyms.get(member, [])
        ]
        synonyms_sentence = f"Synonyms: map {'; '.join(mappings)}."
        description = f"{description} {synonyms_sentence}".strip()
    return description


def _string_kwarg(key: str, text: str, indent: int = 8) -> str:
    """``key="...",`` — parenthesized implicit concatenation when too long."""
    literal = json.dumps(text, ensure_ascii=False)
    inline = f"{key}={literal},"
    if indent + len(inline) <= MAX_LINE:
        return inline
    inner_indent = " " * 4
    chunk_width = MAX_LINE - indent - 4 - 2  # inner indent + surrounding quotes
    chunks = "".join(
        f"{inner_indent}{json.dumps(chunk, ensure_ascii=False)}\n"
        for chunk in _chunk_words(text, chunk_width)
    )
    return f"{key}=(\n{chunks}),"


def _chunk_words(text: str, width: int) -> list[str]:
    """Split on word boundaries so the chunks concatenate back exactly."""
    words = text.split(" ")
    chunks: list[str] = []
    current = ""
    for index, word in enumerate(words):
        piece = word if index == len(words) - 1 else f"{word} "
        if current and len(current) + len(piece) > width:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


def _examples_kwarg(examples: list[str], indent: int = 8) -> str:
    inline = f"examples={json.dumps(examples, ensure_ascii=False)},"
    if indent + len(inline) <= MAX_LINE:
        return inline
    items = "".join(f"    {json.dumps(item, ensure_ascii=False)},\n" for item in examples)
    return f"examples=[\n{items}],"


def _render_field_validators(
    model: ModelSpec, enums: dict[str, EnumSpec], usage: _Usage
) -> list[str]:
    blocks: list[str] = []
    for field in _emission_field_order(model):
        if field.type in enums:
            template = (
                ENUM_LIST_FIELD_VALIDATOR_TEMPLATE
                if field.is_list
                else ENUM_FIELD_VALIDATOR_TEMPLATE
            )
            blocks.append(template.format(field=field.name, enum_name=field.type).rstrip("\n"))
            usage.uses_field_validator = True
            usage.uses_enum = True
            usage.uses_re = True
            usage.uses_logging = True
        if field.normalizer == "currency":
            blocks.append(CURRENCY_VALIDATOR_TEMPLATE.format(field=field.name).rstrip("\n"))
            usage.uses_field_validator = True
            usage.uses_logging = True
        elif field.normalizer == "numeric":
            blocks.append(NUMERIC_VALIDATOR_TEMPLATE.format(field=field.name).rstrip("\n"))
            usage.uses_field_validator = True
            usage.uses_re = True
            usage.uses_logging = True
        elif field.normalizer == "string_list":
            blocks.append(STRING_LIST_VALIDATOR_TEMPLATE.format(field=field.name).rstrip("\n"))
            usage.uses_field_validator = True
    return blocks


def _render_dedup_validators(
    root: ModelSpec, spec: TemplateSpec, models: dict[str, ModelSpec], usage: _Usage
) -> list[str]:
    fields_by_name = {field.name: field for field in root.fields}
    blocks: list[str] = []
    for field_name in spec.needs_root_list_dedup:
        field = fields_by_name[field_name]
        key_expr = _dedup_key_expr(field, models)
        blocks.append(
            ROOT_LIST_DEDUP_TEMPLATE.format(field=field_name, key_expr=key_expr).rstrip("\n")
        )
        usage.uses_model_validator = True
        usage.uses_self = True
    return blocks


def _dedup_key_expr(field: FieldSpec, models: dict[str, ModelSpec]) -> str:
    """Dedup key from the item shape: identity field, else first scalar
    non-list field, else the item's string form."""
    target = models.get(field.type)
    if target is None:
        return "str(item).strip().lower()"
    key_field: str | None = None
    if target.identity_fields:
        key_field = target.identity_fields[0]
    else:
        key_field = next(
            (
                f.name
                for f in target.fields
                if f.type in SCALAR_TYPES and not f.is_list and f.role == "property"
            ),
            None,
        )
    if key_field is None:
        return "str(item).strip().lower()"
    return f'str(getattr(item, "{key_field}", "") or "").strip().lower()'


def _render_str_method(model: ModelSpec) -> str:
    names = list(model.identity_fields)
    scalar_properties = [
        field.name
        for field in model.fields
        if field.role == "property"
        and field.type in SCALAR_TYPES
        and not field.is_list
        and field.name not in names
    ]
    if names:
        names.extend(scalar_properties[:1])
    else:
        names.extend(scalar_properties[:2])
    parts = ", ".join(f"self.{name}" for name in names)
    return STR_METHOD_TEMPLATE.format(parts=parts).rstrip("\n")


# ---------------------------------------------------------------------------
# Verification footer
# ---------------------------------------------------------------------------


def _render_footer(spec: TemplateSpec) -> str:
    """Comment block showing the V6 graph-shape smoke test with the exact
    sample the verification gate synthesizes, so users can re-run it by hand
    (the best-practices.md verification ritual)."""
    plan = synthesize_sample_plan(spec)
    payload_lines = pprint.pformat(plan.payload, width=72, sort_dicts=False).splitlines()
    lines = [
        _banner("Verification"),
        "# Graph-shape smoke test mirroring the templategen verification gate (V6).",
        "# Re-run it by hand after editing this template:",
        "#",
        "#     from docling_graph.core.converters.graph_converter import GraphConverter",
        "#",
        f"#     sample = {spec.root}.model_validate(",
        *(f"#         {line}" for line in payload_lines),
        "#     )",
        "#     graph, _metadata = GraphConverter().pydantic_list_to_graph([sample])",
        '#     assert not graph.graph.get("empty_identity_nodes")',
        '#     print(graph.number_of_nodes(), "nodes /", graph.number_of_edges(), "edges")',
    ]
    return "\n".join(lines)
