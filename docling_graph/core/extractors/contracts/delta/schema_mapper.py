"""Template-to-graph mapping helpers for delta extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .....llm_clients.schema_utils import build_compact_semantic_guide
from .catalog import DeltaNodeCatalog, build_delta_node_catalog, merge_delta_filled_into_root


def _field_aliases(model: type[BaseModel]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for field_name, field_info in model.model_fields.items():
        alias_values: list[str] = []
        alias = getattr(field_info, "alias", None)
        if isinstance(alias, str) and alias != field_name:
            alias_values.append(alias)
        validation_alias = getattr(field_info, "validation_alias", None)
        choices = getattr(validation_alias, "choices", None)
        if isinstance(choices, tuple | list):
            for choice in choices:
                if isinstance(choice, str) and choice != field_name:
                    alias_values.append(choice)
        elif isinstance(validation_alias, str) and validation_alias != field_name:
            alias_values.append(validation_alias)
        if alias_values:
            aliases[field_name] = sorted(set(alias_values))
    return aliases


def build_delta_semantic_guide(template: type[BaseModel], schema_dict: dict[str, Any]) -> str:
    """Build compact schema guidance plus alias hints."""

    base = build_compact_semantic_guide(schema_dict)
    alias_map = _field_aliases(template)
    if not alias_map:
        return base

    alias_lines = ["Alias hints (accept these source labels as synonyms):"]
    for field_name, aliases in alias_map.items():
        alias_lines.append(f"- {field_name}: {', '.join(aliases)}")
    return base + "\n\n" + "\n".join(alias_lines)


def build_catalog_prompt_block(catalog: DeltaNodeCatalog) -> str:
    """Format path-level extraction hints for delta prompts."""
    catalog_paths = set(catalog.paths())

    lines: list[str] = []
    for spec in catalog.nodes:
        path_label = '""' if spec.path == "" else spec.path
        ids_label = ", ".join(spec.id_fields) if spec.id_fields else "none (use ids={})"
        desc = (spec.description or "").strip()
        if len(desc) > 220:
            desc = desc[:220].rstrip() + "..."
        example_hint = (spec.example_hint or "").strip()
        line = f"- {path_label} ({spec.node_type}, {spec.kind}) ids=[{ids_label}]"
        if spec.id_fields:
            line += " :: id keys are mandatory and must match exactly."
        if desc:
            line += f" :: {desc}"
        if example_hint:
            line += f"{example_hint}"
        if (
            spec.kind == "entity"
            and spec.id_fields
            and "section" not in example_hint.lower()
            and "chapter" not in example_hint.lower()
        ):
            line += " Use only identity values from the document (e.g. from tables or named items); do not use section or chapter titles."
        identity_example_values = getattr(spec, "identity_example_values", None)
        if identity_example_values:
            id_label = ", ".join(spec.id_fields) if spec.id_fields else "id"
            examples_str = ", ".join(str(v) for v in identity_example_values[:6])
            line += f" Valid {id_label}: only values from the document (e.g. {examples_str}). Do not use section or chapter titles."
            line += " Emit only when this batch contains the table/structure that defines these identities; otherwise omit."
        if spec.path.endswith("[]") and spec.kind == "entity":
            has_child_list = any(
                p != spec.path and p.startswith(spec.path + ".") and "[]" in p
                for p in catalog_paths
            )
            if has_child_list:
                line += " Put only this path's fields; use the child list path for nested entities."
        if spec.path == "" and spec.kind == "entity":
            line += " Include all required root-level fields from the schema (e.g. reference_document, title) when present in the document."
        lines.append(line)
    return "\n".join(lines)


def project_graph_to_template_root(
    merged_graph: dict[str, Any],
    template: type[BaseModel],
) -> tuple[dict[str, Any], dict[str, int | list[Any]]]:
    """Rebuild template-shaped root object from merged flat IR nodes."""

    catalog = build_delta_node_catalog(template)
    spec_by_path = {spec.path: spec for spec in catalog.nodes}
    path_descriptors: dict[str, list[dict[str, Any]]] = {}
    path_filled: dict[str, list[dict[str, Any]]] = {}

    for node in merged_graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        path = str(node.get("path") or "")
        ids = node.get("ids") if isinstance(node.get("ids"), dict) else {}
        parent = node.get("parent") if isinstance(node.get("parent"), dict) else None
        if parent is None and path != "":
            parent = {"path": "", "ids": {}}
        descriptor = {"path": path, "ids": ids, "parent": parent}
        if "__instance_key" in node:
            descriptor["__instance_key"] = node.get("__instance_key")
        if isinstance(parent, dict) and "__instance_key" in parent:
            parent_ref = descriptor.setdefault("parent", dict(parent))
            if isinstance(parent_ref, dict):
                parent_ref["__instance_key"] = parent.get("__instance_key")
        path_descriptors.setdefault(path, []).append(descriptor)

        raw_properties = node.get("properties")
        properties: dict[str, Any] = raw_properties if isinstance(raw_properties, dict) else {}
        filled = dict(properties)
        spec = spec_by_path.get(path)
        # Prefer ids over properties for identity fields so correct ids (e.g. offer name)
        # are not overwritten by wrong properties (e.g. guarantee block in nom).
        if spec is not None and spec.kind == "entity" and ids:
            id_fields = getattr(spec, "id_fields", None) or ()
            for key in id_fields:
                if key not in ids:
                    continue
                value = ids[key]
                if value is None:
                    continue
                if isinstance(value, str) and value.strip():
                    filled[key] = value.strip()
                elif isinstance(value, int | float | bool):
                    filled[key] = value
                else:
                    filled[key] = value
            for key, value in ids.items():
                if key not in filled and value is not None:
                    filled[key] = value
        if spec is not None and getattr(spec, "id_fields", None):
            for key in spec.id_fields:
                if key not in filled:
                    continue
                val = filled[key]
                if isinstance(val, list | dict):
                    filled[key] = ""
                elif isinstance(val, int | float | bool):
                    filled[key] = str(val)
                elif not isinstance(val, str):
                    filled[key] = str(val) if val is not None else ""
        path_filled.setdefault(path, []).append(filled)

    merge_stats: dict[str, int | list[Any]] = {}
    merged_root = merge_delta_filled_into_root(
        path_filled=path_filled,
        path_descriptors=path_descriptors,
        catalog=catalog,
        stats=merge_stats,
        salvage_orphans=True,
    )
    return merged_root, merge_stats
