"""
Node and edge catalog built from a Pydantic extraction template.

Used by the catalog staged strategy: Pass 0 builds this catalog;
Pass 1 discovers IDs per path; Pass 2 fills nodes; Pass 3 extracts/assembles edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .....llm_clients.schema_utils import build_compact_semantic_guide


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
    """True if model is an entity (graph_id_fields); False if component (is_entity=False)."""
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return True  # default to entity if no config
    if cfg.get("is_entity") is False:
        return False
    return len(_get_id_fields(model)) > 0 or cfg.get("is_entity") is not False


def _is_component(model: type[BaseModel]) -> bool:
    """True if model is explicitly a component (is_entity=False)."""
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return False
    return cfg.get("is_entity") is False


def _schema_hints_for_model(model: type[BaseModel], id_fields: list[str]) -> tuple[str, str]:
    """
    Extract model-level description and short example hints from Pydantic schema.
    Used to guide the LLM in the ID pass. Returns (description, example_hint).
    """
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


def _field_aliases(field_name: str, field_info: Any) -> list[str]:
    """Collect declared alias names for one Pydantic field."""
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


@dataclass
class NodeSpec:
    """Specification of a node type at a given catalog path."""

    path: str
    node_type: str
    id_fields: list[str] = field(default_factory=list)
    kind: str = "entity"  # "entity" | "component"
    parent_path: str = ""  # nearest ancestor entity path ("" for root children)
    field_name: str = ""  # field on parent that holds this node
    is_list: bool = False  # True if path ends with []
    description: str = ""  # from Pydantic model/schema (guides LLM)
    example_hint: str = ""  # short "e.g. field: 'x', 'y'" from schema examples

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
class EdgeSpec:
    """Specification of an edge from schema (edge_label + source/target paths)."""

    edge_label: str
    source_path: str
    target_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_label": self.edge_label,
            "source_path": self.source_path,
            "target_path": self.target_path,
        }


@dataclass
class NodeCatalog:
    """Catalog of all node specs (paths) discoverable from the template."""

    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    field_aliases: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "field_aliases": dict(self.field_aliases),
        }

    def paths(self) -> list[str]:
        return [n.path for n in self.nodes]


def build_node_catalog(template: type[BaseModel]) -> NodeCatalog:
    """
    Build NodeCatalog and EdgeSpec list from a Pydantic template.

    Entity-aware rules:
    - **Entities** (model_config.graph_id_fields or not is_entity=False): included at any depth.
    - **Components** (model_config.is_entity=False): included only when the field has edge_label.
    - Each NodeSpec has kind, parent_path (nearest ancestor entity), field_name, is_list for merge.
    """
    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
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

        for field_name, field_info in model.model_fields.items():
            for alias_name in _field_aliases(field_name, field_info):
                field_aliases.setdefault(alias_name, field_name)
            segment = f".{field_name}" if path_prefix else field_name
            path = f"{path_prefix}{segment}" if path_prefix else field_name
            extra = field_info.json_schema_extra or {}
            raw_edge_label = extra.get("edge_label") if isinstance(extra, dict) else None
            edge_label: str | None = str(raw_edge_label) if raw_edge_label is not None else None
            target_model = _unwrap_model_from_annotation(field_info.annotation)
            origin = get_origin(field_info.annotation)
            if target_model is None:
                continue

            is_entity_child = _is_entity(target_model)
            is_component_child = _is_component(target_model)

            # Include: all entities; components only if edge_label
            if is_entity_child or (is_component_child and edge_label):
                if origin is list:
                    list_path = f"{path}[]"
                    add_node(list_path, target_model, parent_entity_path, field_name, True)
                    if edge_label:
                        edges.append(
                            EdgeSpec(
                                edge_label=edge_label,
                                source_path=path_prefix or "",
                                target_path=list_path,
                            )
                        )
                    next_entity_path = list_path if is_entity_child else parent_entity_path
                    walk(list_path, target_model, next_entity_path, from_root=False)
                else:
                    add_node(path, target_model, parent_entity_path, field_name, False)
                    if edge_label:
                        edges.append(
                            EdgeSpec(
                                edge_label=edge_label,
                                source_path=path_prefix or "",
                                target_path=path,
                            )
                        )
                    next_entity_path = path if is_entity_child else parent_entity_path
                    walk(path, target_model, next_entity_path, from_root=False)
            else:
                # Still traverse for nested entities/edge-labeled components
                if origin is list:
                    list_path = f"{path}[]"
                    if edge_label:
                        edges.append(
                            EdgeSpec(
                                edge_label=edge_label,
                                source_path=path_prefix or "",
                                target_path=list_path,
                            )
                        )
                    walk(list_path, target_model, parent_entity_path, from_root=False)
                else:
                    if edge_label:
                        edges.append(
                            EdgeSpec(
                                edge_label=edge_label,
                                source_path=path_prefix or "",
                                target_path=path,
                            )
                        )
                    walk(path, target_model, parent_entity_path, from_root=False)

    walk("", template, "", from_root=True)
    return NodeCatalog(nodes=nodes, edges=edges, field_aliases=field_aliases)


def get_model_for_path(template: type[BaseModel], path: str) -> type[BaseModel] | None:
    path_to_model: dict[str, type[BaseModel]] = {}

    def _walk(prefix: str, model: type[BaseModel], from_root: bool = True) -> None:
        path_to_model[prefix or ""] = model
        for field_name, field_info in model.model_fields.items():
            seg = f".{field_name}" if prefix else field_name
            p = f"{prefix}{seg}" if prefix else field_name
            target = _unwrap_model_from_annotation(field_info.annotation)
            if target is None:
                continue
            orig = get_origin(field_info.annotation)
            if orig is list:
                lp = f"{p}[]"
                path_to_model[lp] = target
                _walk(lp, target, from_root=False)
            else:
                path_to_model[p] = target
                _walk(p, target, from_root=False)

    _walk("", template, from_root=True)
    return path_to_model.get(path)


def write_catalog_artifact(catalog: NodeCatalog, debug_dir: str) -> str:
    """
    Write node_catalog.json into debug_dir. Creates directory if needed.
    Returns the path of the written file.
    """
    import os

    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, "node_catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog.to_dict(), f, indent=2)
    return path


def build_discovery_schema(catalog: NodeCatalog, allowed_paths: list[str] | None = None) -> str:
    """
    Build JSON schema for ID-pass node skeleton output.
    Output shape: {"nodes": [ {"path": "...", "ids": {...}, "parent": null | {"path": "...", "ids": {...}} } ]}
    """
    if allowed_paths is None:
        allowed_paths = catalog.paths()
    schema: dict[str, Any] = {
        "type": "object",
        "description": "Discovery skeleton: list of node instances with path, ids, and parent reference.",
        "properties": {
            "nodes": {
                "type": "array",
                "description": "Each instance: path, ids (actual identifier values), parent (path+ids or null).",
                "minItems": 1,
                "items": {"$ref": "#/$defs/node_instance"},
            }
        },
        "required": ["nodes"],
        "$defs": {
            "node_instance": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": allowed_paths,
                        "description": 'Catalog path. ROOT is "" (empty string).',
                    },
                    "ids": {
                        "type": "object",
                        "description": "Identifier values for this path. Keys must match that path's id_fields in catalog.",
                        "additionalProperties": {"type": "string"},
                    },
                    "parent": {
                        "type": ["object", "null"],
                        "description": "For non-root: path and ids of the containing instance. Root must have null.",
                        "properties": {
                            "path": {"type": "string", "enum": allowed_paths},
                            "ids": {"type": "object", "additionalProperties": {"type": "string"}},
                        },
                        "required": ["path", "ids"],
                    },
                },
                "required": ["path", "ids", "parent"],
            }
        },
    }
    return json.dumps(schema, indent=2)


def get_discovery_prompt(
    markdown_content: str,
    catalog: NodeCatalog,
    primary_paths: list[str] | None = None,
    allowed_paths: list[str] | None = None,
    *,
    compact: bool = False,
    include_schema_in_user: bool = True,
    structured_output: bool = True,
) -> dict[str, str]:
    """Prompt for node-skeleton discovery. Asks for path, ids, and parent using exact catalog paths."""
    if allowed_paths is None:
        allowed_paths = catalog.paths()
    schema_json = build_discovery_schema(catalog, allowed_paths)
    specs_to_list = catalog.nodes
    if primary_paths is not None:
        path_set = set(primary_paths)
        specs_to_list = [s for s in catalog.nodes if s.path in path_set]
    path_lines = []
    no_id_paths: list[str] = []
    for spec in specs_to_list:
        path_label = '""' if spec.path == "" else spec.path
        if spec.id_fields:
            ids_label = ", ".join(spec.id_fields)
        else:
            ids_label = "none (ids must be {})"
            no_id_paths.append(path_label)
        if spec.path == "":
            parent_part = " (ROOT; parent must be null)"
        elif spec.parent_path == "":
            parent_part = ' (child of ROOT; parent.path="", parent.ids=<root ids>)'
        else:
            parent_part = f" (parent path={spec.parent_path})"
        path_lines.append(f"- {path_label} :: {spec.node_type} ids=[{ids_label}]{parent_part}")
    paths_block = "\n".join(path_lines)
    allowed_block = ", ".join(('""' if p == "" else p) for p in allowed_paths)

    list_path_rule = (
        "For list paths (nested paths under a parent): for each parent instance you list, "
        "output every child instance that belongs to that parent. "
        "If the same child (same path and ids) belongs to several parents, output one node per "
        "(parent, child) pair — same path and ids, parent set to each parent in turn."
    )
    if compact:
        system_prompt = (
            "ID pass only. Return strict JSON object with key 'nodes'. "
            "Each node must be {path, ids, parent}. "
            "Use exact path strings from allowed paths. "
            "ROOT path is ''. ROOT parent is null. "
            "Every non-root node must have parent {path, ids}. "
            "ids keys must exactly match id_fields for the path. "
            "For list paths: output one node per (parent, child) pair; same child under multiple parents → output once per parent. "
            "No markdown, no comments."
        )
        if any(_is_list_under_list(s.path, s) for s in catalog.nodes):
            system_prompt += "\nExample: " + _build_id_pass_example_from_catalog(catalog)
    else:
        system_prompt = (
            "You are a precise extraction assistant running an ID-pass node-skeleton discovery.\n\n"
            "List every entity/component instance in the document. For each instance output ONLY: "
            "path (one of the catalog paths), ids (actual values for that path id fields), parent "
            "(non-root: {path, ids} of container; root: null).\n"
            "IMPORTANT: path must be the exact catalog path string. Path is NOT the class name.\n\n"
            f"Paths: {allowed_block}\n\n"
            f"Catalog:\n{paths_block}\n\n"
            'Rules: ROOT is "". Root parent must be null. '
            "Every NON-ROOT node must have parent as an object {path, ids} (never null). "
            "ids keys must match id_fields for the path. "
            "If a path has no id_fields, ids must be {} (empty object). "
            "Use exact nested path strings from the catalog (list paths end with []). "
            f"{list_path_rule} "
            "Return valid minified JSON only.\n"
            f"Example: {_build_id_pass_example_from_catalog(catalog)}"
        )
    if no_id_paths:
        system_prompt += (
            "\nPaths with no id_fields (ids must be {}): " + ", ".join(no_id_paths) + "."
        )

    user_prompt = (
        "List every node instance in this document. "
        "For each parent instance, list all child instances that belong to it. "
        "For nested list paths, same child under multiple parents: output one node per parent (same path and ids, parent set to each parent).\n\n=== DOCUMENT ===\n"
    )
    user_prompt += f"{markdown_content}\n=== END DOCUMENT ===\n\n"
    user_prompt += f"=== ALLOWED PATHS ===\n{allowed_block}\n=== END ===\n\n"
    user_prompt += f"=== CATALOG ===\n{paths_block}\n=== END ===\n\n"
    if include_schema_in_user and not structured_output:
        user_prompt += f"=== SCHEMA ===\n{schema_json}\n=== END ===\n\n"
    if structured_output:
        try:
            semantic_guide = build_compact_semantic_guide(json.loads(schema_json))
            user_prompt += f"=== SEMANTIC FIELD GUIDANCE ===\n{semantic_guide}\n=== END ===\n\n"
        except json.JSONDecodeError:
            pass
    user_prompt += (
        'Return JSON: {"nodes": [{"path": "...", "ids": {"id_field": "value"} OR {}, '
        '"parent": null or {"path": "...", "ids": {"id_field": "value"}}}, ...]}.'
    )
    return {"system": system_prompt, "user": user_prompt}


def _build_id_pass_example_from_catalog(catalog: NodeCatalog) -> str:
    """Build a minimal, domain-agnostic example JSON snippet from catalog (root + one child).
    If the catalog has a list path under another list, include same child under two parents to show shared-child pattern.
    """
    root_spec = next((s for s in catalog.nodes if s.path == ""), None)
    root_ids: dict[str, str] = {}
    if root_spec and root_spec.id_fields:
        root_ids = dict.fromkeys(root_spec.id_fields, "value")
    root_node: dict[str, Any] = {"path": "", "ids": root_ids, "parent": None}
    nodes: list[dict[str, Any]] = [root_node]

    # Optional: list path under a parent list (same child under multiple parents)
    grandchild_spec = next(
        (
            s
            for s in catalog.nodes
            if s.path != "" and s.parent_path != "" and s.id_fields and s.path.endswith("[]")
        ),
        None,
    )
    if grandchild_spec:
        parent_spec = next(
            (s for s in catalog.nodes if s.path == grandchild_spec.parent_path and s.id_fields),
            None,
        )
        if parent_spec:
            parent_ids_1 = dict.fromkeys(parent_spec.id_fields, "P1")
            parent_ids_2 = dict.fromkeys(parent_spec.id_fields, "P2")
            grandchild_ids = dict.fromkeys(grandchild_spec.id_fields, "shared")
            nodes.extend(
                [
                    {
                        "path": parent_spec.path,
                        "ids": parent_ids_1,
                        "parent": {"path": "", "ids": root_ids},
                    },
                    {
                        "path": parent_spec.path,
                        "ids": parent_ids_2,
                        "parent": {"path": "", "ids": root_ids},
                    },
                    {
                        "path": grandchild_spec.path,
                        "ids": grandchild_ids,
                        "parent": {"path": parent_spec.path, "ids": parent_ids_1},
                    },
                    {
                        "path": grandchild_spec.path,
                        "ids": grandchild_ids,
                        "parent": {"path": parent_spec.path, "ids": parent_ids_2},
                    },
                ]
            )
            return json.dumps({"nodes": nodes}, separators=(",", ":"))

    # Fallback: root + one direct child of root
    child_spec = next(
        (s for s in catalog.nodes if s.path != "" and s.parent_path == "" and s.id_fields),
        None,
    )
    if child_spec is not None:
        child_ids: dict[str, str] = dict.fromkeys(child_spec.id_fields, "value")
        child_node: dict[str, Any] = {
            "path": child_spec.path,
            "ids": child_ids,
            "parent": {"path": "", "ids": root_ids},
        }
        nodes.append(child_node)
    return json.dumps({"nodes": nodes}, separators=(",", ":"))


def _get_spec_by_path(catalog: NodeCatalog) -> dict[str, NodeSpec]:
    """Return path -> NodeSpec for lookup."""
    return {spec.path: spec for spec in catalog.nodes}


def _parent_closure(catalog: NodeCatalog, paths: list[str]) -> set[str]:
    """Return paths plus all ancestor paths (parent_path chain) so parent references are valid."""
    spec_by_path = _get_spec_by_path(catalog)
    result: set[str] = set(paths)
    changed = True
    while changed:
        changed = False
        for p in list(result):
            spec = spec_by_path.get(p)
            if spec is not None and spec.parent_path not in result:
                result.add(spec.parent_path)
                changed = True
    return result


def get_allowed_paths_for_primary_paths(
    catalog: NodeCatalog, primary_paths: list[str]
) -> list[str]:
    """Return primary_paths plus parent closure, sorted. For targeted re-ask prompts."""
    return sorted(_parent_closure(catalog, primary_paths))


def get_id_pass_shards(catalog: NodeCatalog, shard_size: int) -> list[tuple[list[str], list[str]]]:
    """
    Split catalog paths into shards for smaller ID-pass LLM calls.
    Returns list of (primary_paths, allowed_paths). allowed_paths = primary_paths + parent closure.
    """
    if shard_size <= 0:
        all_paths = catalog.paths()
        allowed_list = sorted(_parent_closure(catalog, all_paths))
        return [(all_paths, allowed_list)]
    shards: list[tuple[list[str], list[str]]] = []
    nodes = list(catalog.nodes)
    for i in range(0, len(nodes), shard_size):
        chunk = nodes[i : i + shard_size]
        primary_paths = [spec.path for spec in chunk]
        allowed_list = sorted(_parent_closure(catalog, primary_paths))
        shards.append((primary_paths, allowed_list))
    return shards


def get_identity_paths(catalog: NodeCatalog) -> list[str]:
    """Return paths used for minimal ID pass (root + identity entities)."""
    paths: list[str] = []
    for spec in catalog.nodes:
        if spec.path == "":
            paths.append(spec.path)
            continue
        if spec.kind == "entity" and bool(spec.id_fields):
            paths.append(spec.path)
    return paths


def get_id_pass_shards_v2(
    catalog: NodeCatalog,
    shard_size: int,
    *,
    identity_only: bool = True,
    root_first: bool = True,
) -> list[tuple[list[str], list[str]]]:
    """Deterministic, parent-complete shard planning for ID pass."""
    all_paths = catalog.paths()
    base_paths = get_identity_paths(catalog) if identity_only else list(all_paths)
    if "" not in base_paths and "" in all_paths:
        base_paths.insert(0, "")

    spec_by_path = _get_spec_by_path(catalog)
    base_paths = sorted(set(base_paths), key=lambda p: (p.count("."), p))

    if root_first:
        seed = [
            p
            for p in base_paths
            if p == "" or (spec_by_path.get(p) and spec_by_path[p].parent_path == "")
        ]
        seed_set = set(seed)
        ordered = seed + [p for p in base_paths if p not in seed_set]
    else:
        ordered = base_paths

    if shard_size <= 0 or len(ordered) <= shard_size:
        primary = sorted(_parent_closure(catalog, ordered), key=lambda p: (p.count("."), p))
        return [(primary, primary)]

    shards: list[tuple[list[str], list[str]]] = []
    for i in range(0, len(ordered), shard_size):
        chunk = ordered[i : i + shard_size]
        primary = sorted(_parent_closure(catalog, chunk), key=lambda p: (p.count("."), p))
        shards.append((primary, primary))
    return shards


def _is_list_under_list(path: str, spec: NodeSpec) -> bool:
    """True if path is a list path whose parent is also a list path (same child can belong to multiple parents)."""
    return bool(path.endswith("[]") and spec.parent_path.endswith("[]"))


def _canonical_parent_key(
    parent: dict[str, Any],
    parent_spec: NodeSpec | None,
) -> tuple[Any, ...] | None:
    """Stable key for parent for use in dedup when list-under-list. Matches _canonical_lookup_key style."""
    if parent_spec is None:
        return None
    parent_path = parent.get("path", "")
    if parent_spec.id_fields:
        parent_ids = parent.get("ids") or {}
        from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

        normalized = tuple(
            sorted(
                (f, canonicalize_identity_for_dedup(f, parent_ids.get(f)))
                for f in parent_spec.id_fields
            )
        )
        return (parent_path, normalized)
    return (parent_path, (parent.get("__instance_key") or "",))


def merge_and_dedupe_flat_nodes(
    list_of_flat_nodes: list[list[dict[str, Any]]], catalog: NodeCatalog
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge shard results and dedupe by (path, ids tuple), or (path, ids, parent_key) for list-under-list paths.
    Returns (flat_nodes, per_path_counts)."""
    from docling_graph.core.utils.description_merger import merge_descriptions
    from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

    spec_by_path = _get_spec_by_path(catalog)
    seen: set[tuple[Any, ...]] = set()
    merged: list[dict[str, Any]] = []
    key_to_index: dict[tuple[Any, ...], int] = {}
    path_ordinals: dict[str, int] = {}
    for shard_idx, flat in enumerate(list_of_flat_nodes):
        # Re-key no-id instances per shard to avoid cross-shard collisions and keep parent refs coherent.
        remap_no_id_keys: dict[tuple[str, str], str] = {}
        prepared_nodes: list[dict[str, Any]] = []
        for node in flat:
            if not isinstance(node, dict):
                continue  # type: ignore[unreachable]
            current = dict(node)
            path = current.get("path", "")
            spec = spec_by_path.get(path)
            if spec is None:
                continue
            if not spec.id_fields:
                ordinal = path_ordinals.get(path, 0)
                path_ordinals[path] = ordinal + 1
                prev_key = current.get("__instance_key")
                new_key = f"{path}@shard{shard_idx}#{ordinal}"
                if isinstance(prev_key, str) and prev_key:
                    remap_no_id_keys[(path, prev_key)] = new_key
                current["__instance_key"] = new_key
            prepared_nodes.append(current)

        for node in prepared_nodes:
            path = node.get("path", "")
            if path not in spec_by_path:
                continue
            spec = spec_by_path[path]
            parent = node.get("parent")
            if isinstance(parent, dict):
                parent_path = parent.get("path", "")
                parent_spec = spec_by_path.get(parent_path)
                old_parent_key = parent.get("__instance_key")
                if (
                    parent_spec is not None
                    and not parent_spec.id_fields
                    and isinstance(old_parent_key, str)
                    and old_parent_key
                ):
                    remapped = remap_no_id_keys.get((parent_path, old_parent_key))
                    if remapped is not None:
                        parent = dict(parent)
                        parent["__instance_key"] = remapped
                        node["parent"] = parent

            ids = node.get("ids") or {}
            if not spec.id_fields:
                # Paths without ID fields cannot be safely deduped by ids (always empty),
                # so keep all instances.
                merged.append(node)
                continue
            # Canonicalize all identity values for dedup key so "run_1"/"run1"/"Run-1" collapse
            normalized_ids = {
                f: canonicalize_identity_for_dedup(f, ids.get(f)) for f in spec.id_fields
            }
            ids_tuple = tuple(sorted(normalized_ids.items()))
            # List-under-list: same (path, ids) can appear under multiple parents; keep one descriptor per (parent, child)
            list_under_list = _is_list_under_list(path, spec)
            parent_spec = spec_by_path.get(spec.parent_path) if spec.parent_path else None
            if list_under_list and isinstance(parent, dict) and parent_spec is not None:
                parent_key = _canonical_parent_key(parent, parent_spec)
                key = (path, ids_tuple, parent_key) if parent_key is not None else (path, ids_tuple)
            else:
                key = (path, ids_tuple)
            if key not in seen:
                seen.add(key)
                merged.append(node)
                key_to_index[key] = len(merged) - 1
            else:
                # Merge descriptions into the existing node
                existing_idx = key_to_index[key]
                existing_node = merged[existing_idx]
                for desc_field in ("description", "summary"):
                    existing_val = existing_node.get(desc_field)
                    new_val = node.get(desc_field)
                    if isinstance(existing_val, str) and isinstance(new_val, str) and new_val:
                        merged[existing_idx] = dict(existing_node)
                        merged[existing_idx][desc_field] = merge_descriptions(
                            existing_val, new_val, max_length=4096
                        )
                        existing_node = merged[existing_idx]
                    elif isinstance(new_val, str) and new_val and not existing_val:
                        merged[existing_idx] = dict(existing_node)
                        merged[existing_idx][desc_field] = new_val
                        existing_node = merged[existing_idx]
    per_path_counts = {spec.path: 0 for spec in catalog.nodes}
    for node in merged:
        p = node.get("path", "")
        if p in per_path_counts:
            per_path_counts[p] += 1
    return merged, per_path_counts


def _normalize_skeleton_path(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    p = raw.strip()
    if p in {"(root)", "root", "(ROOT)", "ROOT"}:
        return ""
    if p.startswith("(root)"):
        p = p[len("(root)") :].strip()
    return p


def _normalize_skeleton_ids(
    ids_raw: Any, spec: NodeSpec, err_prefix: str
) -> tuple[dict[str, str] | None, list[str]]:
    local_errors: list[str] = []
    if not isinstance(ids_raw, dict):
        return None, [f"{err_prefix}: ids must be an object"]
    ids_norm: dict[str, str] = {}
    expected_fields = list(spec.id_fields)
    expected = set(expected_fields)
    got = {k for k in ids_raw.keys() if isinstance(k, str)}
    missing = [f for f in expected_fields if f not in ids_raw]
    extra = sorted(k for k in got if k not in expected)
    if missing:
        local_errors.append(f"{err_prefix}: missing id field(s): {missing}")
    if extra:
        local_errors.append(f"{err_prefix}: unexpected id field(s): {extra}")
    for f in expected_fields:
        if f not in ids_raw:
            continue
        v = ids_raw.get(f)
        if v is None:
            local_errors.append(f"{err_prefix}: id field '{f}' cannot be null")
            continue
        s = str(v).strip()
        if not s:
            local_errors.append(f"{err_prefix}: id field '{f}' cannot be empty")
            continue
        ids_norm[f] = s
    return (ids_norm if not local_errors else None), local_errors


def _validate_one_skeleton_node(
    i: int,
    item: Any,
    allowed_paths: set[str],
    spec_by_path: dict[str, NodeSpec],
    seen_keys: set[tuple[str, tuple[tuple[str, str], ...]]],
    path_ordinals: dict[str, int],
) -> tuple[list[str], dict[str, Any] | None, bool]:
    """Validate one raw node. Returns (errors, flat_node_or_none, is_root)."""
    if not isinstance(item, dict):
        return [f"Item {i}: not an object"], None, False
    path = _normalize_skeleton_path(item.get("path"))
    if path is None or path not in allowed_paths:
        return (
            [f"Item {i}: missing or invalid path (must be one of {sorted(allowed_paths)})"],
            None,
            False,
        )
    spec = spec_by_path.get(path)
    if not spec:
        return [f"Item {i}: unknown path '{path}'"], None, False
    ids, id_errs = _normalize_skeleton_ids(item.get("ids"), spec, f"Item {i} (path '{path}')")
    if id_errs:
        return id_errs, None, False
    assert ids is not None
    parent_raw = item.get("parent")
    if path == "":
        if parent_raw is not None and parent_raw is not False:
            return [f"Item {i}: root must have parent null"], None, False
        parent_desc = None
        is_root = True
    else:
        if not isinstance(parent_raw, dict):
            return (
                [f"Item {i} (path '{path}'): non-root must have parent {{path, ids}}"],
                None,
                False,
            )
        p_path = _normalize_skeleton_path(parent_raw.get("path"))
        if p_path is None or p_path not in allowed_paths:
            return (
                [f"Item {i} (path '{path}'): parent.path must be one of {sorted(allowed_paths)}"],
                None,
                False,
            )
        if p_path != spec.parent_path:
            return (
                [
                    f"Item {i} (path '{path}'): parent.path '{p_path}' must equal catalog parent_path '{spec.parent_path}'"
                ],
                None,
                False,
            )
        parent_spec = spec_by_path.get(p_path)
        if not parent_spec:
            return [f"Item {i} (path '{path}'): unknown parent path '{p_path}'"], None, False
        p_ids, p_id_errs = _normalize_skeleton_ids(
            parent_raw.get("ids"), parent_spec, f"Item {i} (path '{path}') parent"
        )
        if p_id_errs:
            return p_id_errs, None, False
        assert p_ids is not None
        parent_desc = {"path": p_path, "ids": p_ids}
        is_root = False
    if spec.id_fields and not _is_list_under_list(path, spec):
        key = (path, tuple(sorted(ids.items())))
        if key in seen_keys:
            return [], None, is_root
        seen_keys.add(key)
    node_out: dict[str, Any] = {"path": path, "ids": ids, "parent": parent_desc}
    if not spec.id_fields:
        ordinal = path_ordinals.get(path, 0)
        path_ordinals[path] = ordinal + 1
        node_out["__instance_key"] = f"{path}#{ordinal}"
    return [], node_out, is_root


def _validate_skeleton_parent_refs(
    flat_nodes: list[dict[str, Any]],
    spec_by_path: dict[str, NodeSpec],
) -> list[str]:
    """Check that every non-root node's parent exists in flat_nodes. Returns list of errors."""
    errors: list[str] = []
    existing_parent_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    parent_candidates: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    for node in flat_nodes:
        key = (node["path"], tuple(sorted((node.get("ids") or {}).items())))
        existing_parent_keys.add(key)
        parent_candidates.setdefault(key, []).append(node)
    for i, node in enumerate(flat_nodes):
        if node["path"] == "":
            continue
        parent = node.get("parent")
        if not isinstance(parent, dict):
            continue
        parent_key = (parent.get("path", ""), tuple(sorted((parent.get("ids") or {}).items())))
        if parent_key not in existing_parent_keys:
            errors.append(
                f"Item {i} (path '{node['path']}'): parent reference not found in nodes "
                f"(parent.path='{parent.get('path', '')}', parent.ids={parent.get('ids', {})})"
            )
            continue
        parent_spec = spec_by_path.get(parent.get("path", ""))
        if parent_spec is not None and not parent_spec.id_fields:
            candidates = parent_candidates.get(parent_key, [])
            if len(candidates) != 1:
                errors.append(
                    f"Item {i} (path '{node['path']}'): ambiguous parent reference for "
                    f"path '{parent.get('path', '')}' without id_fields ({len(candidates)} candidates)"
                )
                continue
            parent_instance_key = candidates[0].get("__instance_key")
            if isinstance(parent_instance_key, str):
                parent["__instance_key"] = parent_instance_key
    return errors


def validate_id_pass_skeleton_response(
    data: dict[str, Any],
    catalog: NodeCatalog,
    allowed_paths_override: set[str] | None = None,
) -> tuple[bool, list[str], list[dict[str, Any]], dict[str, int]]:
    """
    Validate node-skeleton ID-pass response (path, ids, parent).
    Build flat_nodes with real ids and parent references.
    Returns (success, errors, flat_nodes, per_path_counts).
    """
    errors: list[str] = []
    flat_nodes: list[dict[str, Any]] = []
    per_path_counts: dict[str, int] = {spec.path: 0 for spec in catalog.nodes}
    spec_by_path = _get_spec_by_path(catalog)
    allowed_paths = (
        allowed_paths_override if allowed_paths_override is not None else set(catalog.paths())
    )
    seen_keys: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    path_ordinals: dict[str, int] = {}

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list):
        return False, ["Missing or invalid 'nodes' (must be an array)"], flat_nodes, per_path_counts

    has_root = False
    for i, item in enumerate(raw_nodes):
        item_errors, node_out, is_root = _validate_one_skeleton_node(
            i, item, allowed_paths, spec_by_path, seen_keys, path_ordinals
        )
        if item_errors:
            errors.extend(item_errors)
            continue
        if is_root:
            has_root = True
        if node_out is not None:
            flat_nodes.append(node_out)
            per_path_counts[node_out["path"]] = per_path_counts.get(node_out["path"], 0) + 1

    if errors:
        return False, errors, flat_nodes, per_path_counts

    parent_errors = _validate_skeleton_parent_refs(flat_nodes, spec_by_path)
    if parent_errors:
        return False, parent_errors, flat_nodes, per_path_counts

    if not has_root:
        return (
            False,
            ['Missing root instance: expected one node with path "" and parent null'],
            flat_nodes,
            per_path_counts,
        )

    return True, [], flat_nodes, per_path_counts


def flat_nodes_to_path_lists(
    flat_nodes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group flat nodes list by path for fill pass.
    Each value is a list of descriptors (path, ids, parent) in document order.
    """
    path_to_list: dict[str, list[dict[str, Any]]] = {}
    for desc in flat_nodes:
        path = desc["path"]
        if path not in path_to_list:
            path_to_list[path] = []
        path_to_list[path].append(desc)
    return path_to_list


def write_id_pass_artifact(
    id_pass_data: dict[str, Any], per_path_counts: dict[str, int], debug_dir: str
) -> str:
    """Write id_pass.json (raw response + per_path_counts) into debug_dir. Returns path."""
    import os

    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, "id_pass.json")
    payload = {"nodes": id_pass_data.get("nodes", []), "per_path_counts": per_path_counts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
