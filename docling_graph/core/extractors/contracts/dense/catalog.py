"""
Dense extraction catalog: built from the Pydantic template.

Provides path specs, id_fields, parent_path, and projected fill schemas for Phase 1 (skeleton) and Phase 2 (fill).
No dependency on other contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from docling_graph.logging_utils import get_component_logger

logger = get_component_logger("DenseExtraction", __name__)


def _unwrap_model_from_annotation(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        model = _unwrap_model_from_annotation(arg)
        if model is not None:
            return model
    return None


def _get_id_fields(model: type[BaseModel]) -> list[str]:
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return []
    raw = cfg.get("graph_id_fields", [])
    return [f for f in raw if isinstance(f, str)]


def _is_entity(model: type[BaseModel]) -> bool:
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return True
    if cfg.get("is_entity") is False:
        return False
    return len(_get_id_fields(model)) > 0 or cfg.get("is_entity") is not False


def _is_component(model: type[BaseModel]) -> bool:
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return False
    return cfg.get("is_entity") is False


def _is_reference_field(field_info: Any, target_model: type[BaseModel] | None) -> bool:
    """True for entity-link fields declared reference-only in the template.

    A field marked ``json_schema_extra={"graph_reference": True}`` (the
    ``edge(..., reference=True)`` helper pattern) carries id-only references to
    entities whose full detail lives elsewhere. Such fields are filled BY THE
    PARENT's own fill call (projected down to the target's graph_id_fields)
    instead of being skeleton-discovered: per-parent membership then survives
    (skeleton dedup collapses same-id children across parents) and there is no
    parent reference to drift. The marker is honored only when the target
    actually has identity fields — a reference to an identity-less model is
    meaningless and falls back to normal handling.
    """
    if target_model is None or not _get_id_fields(target_model):
        return False
    extra = getattr(field_info, "json_schema_extra", None)
    return isinstance(extra, dict) and extra.get("graph_reference") is True


def _field_aliases(field_name: str, field_info: Any) -> list[str]:
    values: list[str] = []
    alias = getattr(field_info, "alias", None)
    if isinstance(alias, str) and alias != field_name:
        values.append(alias)
    validation_alias = getattr(field_info, "validation_alias", None)
    choices = getattr(validation_alias, "choices", None)
    if isinstance(choices, tuple | list):
        for choice in choices:
            if isinstance(choice, str) and choice != field_name:
                values.append(choice)
    elif isinstance(validation_alias, str) and validation_alias != field_name:
        values.append(validation_alias)
    return sorted(set(values))


def _schema_hints_for_model(model: type[BaseModel], id_fields: list[str]) -> tuple[str, str]:
    try:
        schema = model.model_json_schema()
    except Exception:
        return "", ""
    desc = (schema.get("description") or (model.__doc__ or "") or "").strip()
    desc = desc[:400].strip() if desc else ""
    parts: list[str] = []
    props = schema.get("properties") or {}
    for f in id_fields[:4]:
        field_schema = props.get(f)
        if not isinstance(field_schema, dict):
            continue
        ex = field_schema.get("examples")
        if ex and isinstance(ex, list):
            samples = [str(x)[:50] for x in ex[:3]]
            parts.append(f"{f}: {', '.join(repr(s) for s in samples)}")
    example_hint = (" e.g. " + "; ".join(parts)) if parts else ""
    return desc, example_hint


@dataclass
class NodeSpec:
    """Specification of a node type at a given catalog path."""

    path: str
    node_type: str
    id_fields: list[str] = field(default_factory=list)
    kind: str = "entity"
    parent_path: str = ""
    field_name: str = ""
    is_list: bool = False
    description: str = ""
    example_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "node_type": self.node_type,
            "id_fields": self.id_fields,
            "kind": self.kind,
            "parent_path": self.parent_path,
            "field_name": self.field_name,
            "is_list": self.is_list,
            "description": self.description,
            "example_hint": self.example_hint,
        }


@dataclass
class NodeCatalog:
    """Catalog of node specs derived from the template."""

    nodes: list[NodeSpec] = field(default_factory=list)
    field_aliases: dict[str, str] = field(default_factory=dict)

    def paths(self) -> list[str]:
        return [n.path for n in self.nodes]


def build_node_catalog(
    template: type[BaseModel], *, include_references: bool = False
) -> NodeCatalog:
    """Build NodeCatalog from a Pydantic template. Entity paths only: components
    are filled inline with their parent, and reference fields
    (``json_schema_extra={"graph_reference": True}``) are filled id-only by the
    parent — neither is skeleton-discovered.

    ``include_references=True`` restores paths for reference fields (and their
    subtrees). The provenance binder uses this: nodes that exist in the graph
    only through references still need to be walked and grounded.
    """
    nodes: list[NodeSpec] = []
    field_aliases: dict[str, str] = {}

    def add_node(
        path: str,
        model: type[BaseModel],
        parent_path: str,
        field_name: str,
        is_list: bool,
    ) -> None:
        id_fields = _get_id_fields(model)
        node_type = getattr(model, "__name__", "Unknown")
        kind = "component" if _is_component(model) else "entity"
        description, example_hint = _schema_hints_for_model(model, id_fields)
        nodes.append(
            NodeSpec(
                path=path,
                node_type=node_type,
                id_fields=id_fields,
                kind=kind,
                parent_path=parent_path,
                field_name=field_name,
                is_list=is_list,
                description=description,
                example_hint=example_hint,
            )
        )

    def walk(
        path_prefix: str,
        model: type[BaseModel],
        parent_entity_path: str,
        from_root: bool,
        ancestry: tuple[type[BaseModel], ...] = (),
    ) -> None:
        if from_root:
            add_node("", model, "", "", False)
        ancestry = (*ancestry, model)

        for fname, field_info in model.model_fields.items():
            for alias_name in _field_aliases(fname, field_info):
                field_aliases.setdefault(alias_name, fname)
            segment = f".{fname}" if path_prefix else fname
            path = f"{path_prefix}{segment}" if path_prefix else fname
            target_model = _unwrap_model_from_annotation(field_info.annotation)
            origin = get_origin(field_info.annotation)
            if target_model is None:
                continue

            if _is_reference_field(field_info, target_model) and not include_references:
                # Reference fields get no path (and no descendant paths): the
                # parent's projected fill schema re-includes them id-only, and
                # the GraphConverter resolves the id-only instances onto the
                # canonical nodes via the NodeIDRegistry.
                continue

            if target_model in ancestry:
                # Recursive nesting (self-loops, mutual cycles): the model is
                # already on this walk's ancestry, so descending again would
                # recurse forever. The recurrence is pruned from discovery —
                # the model keeps its non-recursive path(s), and recursively
                # nested instances resolve through the converter like any
                # non-discovered occurrence.
                continue

            if _is_entity(target_model) and not _is_component(target_model):
                if origin is list:
                    list_path = f"{path}[]"
                    add_node(list_path, target_model, parent_entity_path, fname, True)
                    walk(list_path, target_model, list_path, from_root=False, ancestry=ancestry)
                else:
                    add_node(path, target_model, parent_entity_path, fname, False)
                    walk(path, target_model, path, from_root=False, ancestry=ancestry)
            else:
                # Components (is_entity=False) never become catalog paths — even
                # with an edge_label. They are value objects without identity,
                # which Phase 1 cannot reliably discover (ids={}), so hoisting
                # them starved the parent's fill schema of exactly these fields
                # (conditions/franchises stayed empty while direct mode filled
                # them inline). They stay in the parent's projected fill schema
                # and the GraphConverter embeds them inline regardless. The walk
                # still recurses so entities nested below a component keep paths
                # parented to the nearest entity ancestor.
                if origin is list:
                    walk(
                        f"{path}[]",
                        target_model,
                        parent_entity_path,
                        from_root=False,
                        ancestry=ancestry,
                    )
                else:
                    walk(path, target_model, parent_entity_path, from_root=False, ancestry=ancestry)

    walk("", template, "", from_root=True)
    return NodeCatalog(nodes=nodes, field_aliases=field_aliases)


def get_model_for_path(template: type[BaseModel], path: str) -> type[BaseModel] | None:
    """Return the Pydantic model class for a catalog path."""
    path_to_model: dict[str, type[BaseModel]] = {}

    def _walk(prefix: str, model: type[BaseModel], ancestry: tuple[type[BaseModel], ...]) -> None:
        path_to_model[prefix or ""] = model
        ancestry = (*ancestry, model)
        for fname, field_info in model.model_fields.items():
            seg = f".{fname}" if prefix else fname
            p = f"{prefix}{seg}" if prefix else fname
            target = _unwrap_model_from_annotation(field_info.annotation)
            if target is None or target in ancestry:  # recursion pruned like build_node_catalog
                continue
            orig = get_origin(field_info.annotation)
            if orig is list:
                lp = f"{p}[]"
                path_to_model[lp] = target
                _walk(lp, target, ancestry)
            else:
                path_to_model[p] = target
                _walk(p, target, ancestry)

    _walk("", template, ())
    return path_to_model.get(path)


def _reference_projection(
    field_info: Any, target_model: type[BaseModel], is_list: bool
) -> dict[str, Any]:
    """Id-fields-only schema for a reference field, self-contained (no $refs)."""
    id_fields = _get_id_fields(target_model)
    try:
        target_props = target_model.model_json_schema().get("properties") or {}
    except Exception:
        target_props = {}
    item_props: dict[str, Any] = {}
    for id_field in id_fields:
        source = target_props.get(id_field)
        item_props[id_field] = (
            {k: v for k, v in source.items() if k in ("type", "description", "examples")}
            if isinstance(source, dict)
            else {"type": "string"}
        )
    item_schema: dict[str, Any] = {
        "type": "object",
        "description": (
            f"Identity-only reference to a {target_model.__name__}: output ONLY "
            f"{', '.join(repr(f) for f in id_fields)}, matching the name used where the "
            f"{target_model.__name__} is described in full."
        ),
        "properties": item_props,
        "required": list(id_fields),
    }
    field_description = getattr(field_info, "description", None)
    if is_list:
        out: dict[str, Any] = {"type": "array", "items": item_schema}
    else:
        out = dict(item_schema)
    if isinstance(field_description, str) and field_description:
        out["description"] = field_description
    return out


def build_projected_fill_schema(
    template: type[BaseModel], spec: NodeSpec, catalog: NodeCatalog
) -> str:
    """Return JSON schema for filling one path: the model schema minus nested
    child path fields, with reference fields projected down to identity only."""
    model = get_model_for_path(template, spec.path)
    if model is None:
        return "{}"
    schema = model.model_json_schema()
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return json.dumps(schema, indent=2)
    child_fields = {
        child.field_name
        for child in catalog.nodes
        if child.parent_path == spec.path and child.field_name
    }
    keep_props = {k: v for k, v in props.items() if k not in child_fields}
    for fname, field_info in model.model_fields.items():
        if fname not in keep_props:
            continue
        target_model = _unwrap_model_from_annotation(field_info.annotation)
        if not _is_reference_field(field_info, target_model):
            continue
        assert target_model is not None  # guaranteed by _is_reference_field
        keep_props[fname] = _reference_projection(
            field_info, target_model, is_list=get_origin(field_info.annotation) is list
        )
    schema = dict(schema)
    schema["properties"] = keep_props
    if isinstance(schema.get("required"), list):
        schema["required"] = [k for k in schema["required"] if k in keep_props]
    return json.dumps(schema, indent=2)


def path_has_reference_fields(template: type[BaseModel], spec: NodeSpec) -> bool:
    """True when the path's fill schema carries id-only reference projections.

    Reference-list fields (per-instance membership lists) are the fields most
    prone to first-instance dumping when several sibling parents share one fill
    call, so callers use this to drop the fill batch size to one for such paths.
    """
    model = get_model_for_path(template, spec.path)
    if model is None:
        return False
    for field_info in model.model_fields.values():
        target_model = _unwrap_model_from_annotation(field_info.annotation)
        if _is_reference_field(field_info, target_model):
            return True
    return False


# Character budget per path description in the skeleton semantic guide. Phase 1
# decides identity and classification, so schema authors should front-load the
# discriminating sentence of each class docstring within this budget.
_GUIDE_DESCRIPTION_CHARS = 240


def build_skeleton_semantic_guide(catalog: NodeCatalog) -> str:
    """Per-path semantic guide for Phase 1: path, node type, id_fields, plus the
    class docstring (truncated) and identity examples.

    This is where the template author's classification guidance ("an Option is
    not an Offre", granularity rules, canonical naming) reaches the skeleton —
    without it the model assigns paths blind and misfiles instances. Non-identity
    property names are still deliberately absent: Phase 1 must not extract values.
    """
    lines: list[str] = []
    truncated_paths: list[str] = []
    for spec in catalog.nodes:
        path_label = '""' if spec.path == "" else spec.path
        ids_label = ", ".join(spec.id_fields) if spec.id_fields else "none (use ids={})"
        line = f"- {path_label} ({spec.node_type}) ids=[{ids_label}]"
        description = " ".join((spec.description or "").split())
        if description:
            shown = description[:_GUIDE_DESCRIPTION_CHARS]
            # A visible marker so it is obvious in the guide (and debug
            # artifacts) that steering past this point never reached Phase 1.
            if len(description) > _GUIDE_DESCRIPTION_CHARS:
                shown = shown.rstrip() + " […]"
                truncated_paths.append(path_label)
            line += f" — {shown}"
        if spec.example_hint:
            line += f" ({spec.example_hint.strip()})"
        lines.append(line)
    if truncated_paths:
        # Author-facing signal: discovery-time steering (path routing,
        # cardinality, negative cues) MUST live in the first
        # ``_GUIDE_DESCRIPTION_CHARS`` chars of the class docstring — anything
        # past the cut only reaches the fill phase.
        logger.warning(
            "Semantic guide: class docstring truncated at %d chars for %s — "
            "discovery-time steering after the cut is not shown to Phase 1; "
            "move it into the first sentence.",
            _GUIDE_DESCRIPTION_CHARS,
            ", ".join(truncated_paths),
        )
    return "\n".join(lines)


def skeleton_output_schema(allowed_paths: list[str]) -> dict[str, Any]:
    """JSON schema for Phase 1 skeleton LLM output.

    Compact handle-based format: each node has an integer handle ``i`` and
    references its parent by the parent's handle ``p`` within the same
    response. No repeated parent objects, no ancestry arrays.
    """
    return {
        "type": "object",
        "description": "Skeleton: entity instances with integer handles. No property values.",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {
                            "type": "integer",
                            "description": "Handle: sequential integer, unique in this response.",
                        },
                        "path": {"type": "string", "enum": allowed_paths},
                        "ids": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Short identifier labels copied verbatim from the document.",
                        },
                        "p": {
                            "type": ["integer", "null"],
                            "description": "Handle (i) of the parent node in this response; null for the root.",
                        },
                        "c": {
                            "type": ["integer", "null"],
                            "description": "Number N of the '--- CHUNK N ---' marker this entity was found under; omit when unsure.",
                        },
                    },
                    "required": ["i", "path", "ids"],
                },
            }
        },
        "required": ["nodes"],
    }


def bottom_up_path_order(catalog: NodeCatalog) -> list[str]:
    """Return catalog paths in bottom-up order (deepest first) for fill pass."""

    def depth(p: str) -> int:
        return (p.count(".") + 1) if p else 0

    return sorted(catalog.paths(), key=depth, reverse=True)
