"""
Dense extraction catalog: built from the Pydantic template.

Provides path specs, id_fields, parent_path, and projected fill schemas for Phase 1 (skeleton) and Phase 2 (fill).
No dependency on delta or staged contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel


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


def _field_aliases(field_name: str, field_info: Any) -> list[str]:
    values: list[str] = []
    alias = getattr(field_info, "alias", None)
    if isinstance(alias, str) and alias != field_name:
        values.append(alias)
    validation_alias = getattr(field_info, "validation_alias", None)
    choices = getattr(validation_alias, "choices", None)
    if isinstance(choices, (tuple, list)):
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


def build_node_catalog(template: type[BaseModel]) -> NodeCatalog:
    """Build NodeCatalog from a Pydantic template. Entities and edge-labeled components only."""
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
    ) -> None:
        if from_root:
            add_node("", model, "", "", False)

        for fname, field_info in model.model_fields.items():
            for alias_name in _field_aliases(fname, field_info):
                field_aliases.setdefault(alias_name, fname)
            segment = f".{fname}" if path_prefix else fname
            path = f"{path_prefix}{segment}" if path_prefix else fname
            extra = getattr(field_info, "json_schema_extra", None) or {}
            raw_edge_label = extra.get("edge_label") if isinstance(extra, dict) else None
            edge_label = str(raw_edge_label) if raw_edge_label is not None else None
            target_model = _unwrap_model_from_annotation(field_info.annotation)
            origin = get_origin(field_info.annotation)
            if target_model is None:
                continue

            is_entity_child = _is_entity(target_model)
            is_component_child = _is_component(target_model)
            if is_entity_child or (is_component_child and edge_label):
                if origin is list:
                    list_path = f"{path}[]"
                    add_node(list_path, target_model, parent_entity_path, fname, True)
                    next_entity_path = list_path if is_entity_child else parent_entity_path
                    walk(list_path, target_model, next_entity_path, from_root=False)
                else:
                    add_node(path, target_model, parent_entity_path, fname, False)
                    next_entity_path = path if is_entity_child else parent_entity_path
                    walk(path, target_model, next_entity_path, from_root=False)
            else:
                if origin is list:
                    walk(f"{path}[]", target_model, parent_entity_path, from_root=False)
                else:
                    walk(path, target_model, parent_entity_path, from_root=False)

    walk("", template, "", from_root=True)
    return NodeCatalog(nodes=nodes, field_aliases=field_aliases)


def get_model_for_path(template: type[BaseModel], path: str) -> type[BaseModel] | None:
    """Return the Pydantic model class for a catalog path."""
    path_to_model: dict[str, type[BaseModel]] = {}

    def _walk(prefix: str, model: type[BaseModel]) -> None:
        path_to_model[prefix or ""] = model
        for fname, field_info in model.model_fields.items():
            seg = f".{fname}" if prefix else fname
            p = f"{prefix}{seg}" if prefix else fname
            target = _unwrap_model_from_annotation(field_info.annotation)
            if target is None:
                continue
            orig = get_origin(field_info.annotation)
            if orig is list:
                lp = f"{p}[]"
                path_to_model[lp] = target
                _walk(lp, target)
            else:
                path_to_model[p] = target
                _walk(p, target)

    _walk("", template)
    return path_to_model.get(path)


def build_projected_fill_schema(
    template: type[BaseModel], spec: NodeSpec, catalog: NodeCatalog
) -> str:
    """Return JSON schema for filling one path: model schema minus nested child path fields."""
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
    schema = dict(schema)
    schema["properties"] = keep_props
    if isinstance(schema.get("required"), list):
        schema["required"] = [k for k in schema["required"] if k in keep_props]
    return json.dumps(schema, indent=2)


def build_skeleton_semantic_guide(catalog: NodeCatalog) -> str:
    """Skeleton-only semantic guide: path, node type, id_fields. No other property names or descriptions."""
    lines: list[str] = []
    for spec in catalog.nodes:
        path_label = '""' if spec.path == "" else spec.path
        ids_label = ", ".join(spec.id_fields) if spec.id_fields else "none (use ids={})"
        lines.append(f"- {path_label} ({spec.node_type}) ids=[{ids_label}]")
    return "\n".join(lines)


def skeleton_output_schema(allowed_paths: list[str]) -> dict[str, Any]:
    """JSON schema for Phase 1 skeleton LLM output: nodes with path, ids, parent, optional ancestry."""
    parent_ref = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "ids": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }
    return {
        "type": "object",
        "description": "Skeleton: node instances with path, ids, parent, and ancestry. No properties.",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "enum": allowed_paths},
                        "ids": {"type": "object", "additionalProperties": {"type": "string"}},
                        "parent": {
                            "oneOf": [
                                {"type": "null"},
                                parent_ref,
                            ]
                        },
                        "ancestry": {
                            "type": "array",
                            "description": "Full lineage from root to immediate parent (each element: path, ids).",
                            "items": parent_ref,
                        },
                    },
                    "required": ["path", "ids"],
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
