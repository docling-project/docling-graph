"""Delta-owned catalog and projection assembly utilities."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel

logger = logging.getLogger(__name__)
LOCAL_ID_FIELD_HINTS: tuple[str, ...] = ("line_number", "index", "position", "item_number")


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


def _is_component(model: type[BaseModel]) -> bool:
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return False
    return cfg.get("is_entity") is False


def _is_entity(model: type[BaseModel]) -> bool:
    cfg = getattr(model, "model_config", {}) or {}
    if not isinstance(cfg, dict):
        return True
    if cfg.get("is_entity") is False:
        return False
    return len(_get_id_fields(model)) > 0 or cfg.get("is_entity") is not False


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


def _model_property_fields(model: type[BaseModel]) -> list[str]:
    fields: list[str] = []
    for field_name, field_info in model.model_fields.items():
        target = _unwrap_model_from_annotation(field_info.annotation)
        if target is None:
            fields.append(field_name)
    return fields


def _field_description(field_info: Any) -> str:
    description = getattr(field_info, "description", None)
    if isinstance(description, str):
        return description.strip()
    return ""


def _collect_scalar_examples(field_info: Any) -> list[str]:
    examples: list[str] = []
    raw_examples = getattr(field_info, "examples", None)
    if isinstance(raw_examples, list | tuple):
        for item in raw_examples:
            if isinstance(item, str) and item.strip():
                examples.append(item.strip())
            elif isinstance(item, int | float | bool):
                examples.append(str(item))

    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict):
        example_value = extra.get("example")
        if isinstance(example_value, str) and example_value.strip():
            examples.append(example_value.strip())
        elif isinstance(example_value, int | float | bool):
            examples.append(str(example_value))
        extra_examples = extra.get("examples")
        if isinstance(extra_examples, list | tuple):
            for item in extra_examples:
                if isinstance(item, str) and item.strip():
                    examples.append(item.strip())
                elif isinstance(item, int | float | bool):
                    examples.append(str(item))
    return examples


def _identity_values_from_dict(d: Any, id_fields: list[str]) -> list[str]:
    if not isinstance(d, dict) or not id_fields:
        return []
    values: list[str] = []
    for key in id_fields:
        v = d.get(key)
        if v is not None and isinstance(v, str) and v.strip():
            values.append(v.strip())
        elif v is not None and not isinstance(v, dict | list):
            values.append(str(v).strip())
    return values


def _identity_example_values_from_field(field_info: Any, id_fields: list[str]) -> list[str]:
    """Return list of identity-field values from Field examples (list-of-dict); for allowlist filtering."""
    if not id_fields:
        return []
    collected: list[str] = []
    raw_examples = getattr(field_info, "examples", None)
    if isinstance(raw_examples, list | tuple):
        for item in raw_examples:
            if isinstance(item, list | tuple):
                for sub in item:
                    collected.extend(_identity_values_from_dict(sub, id_fields))
            elif isinstance(item, dict):
                collected.extend(_identity_values_from_dict(item, id_fields))

    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict):
        for key in ("example", "examples"):
            val = extra.get(key)
            if isinstance(val, list | tuple):
                for sub in val:
                    collected.extend(_identity_values_from_dict(sub, id_fields))
            elif isinstance(val, dict):
                collected.extend(_identity_values_from_dict(val, id_fields))

    return list(dict.fromkeys(collected))[:12]


def _identity_examples_from_field(field_info: Any, id_fields: list[str]) -> str:
    """Extract identity-field values from Field examples when they are list-of-dict (any list-entity field)."""
    unique = _identity_example_values_from_field(field_info, id_fields)[:6]
    if not unique:
        return ""
    id_label = ", ".join(id_fields)
    return f" Valid {id_label}: only values from the document structure (e.g. {', '.join(unique)}). Do not use section or chapter titles."


def _field_example_hint(field_info: Any, id_fields: list[str] | None = None) -> str:
    scalar = _collect_scalar_examples(field_info)
    if id_fields:
        identity_hint = _identity_examples_from_field(field_info, id_fields)
        if identity_hint:
            return identity_hint
    if not scalar:
        return ""
    unique_examples = list(dict.fromkeys(scalar))[:3]
    return f" :: examples={unique_examples}"


@dataclass
class DeltaNodeSpec:
    path: str
    node_type: str
    id_fields: list[str] = field(default_factory=list)
    kind: str = "entity"
    parent_path: str = ""
    field_name: str = ""
    is_list: bool = False
    property_fields: list[str] = field(default_factory=list)
    description: str = ""
    example_hint: str = ""
    identity_example_values: list[str] = field(default_factory=list)


@dataclass
class DeltaNodeCatalog:
    nodes: list[DeltaNodeSpec] = field(default_factory=list)
    field_aliases: dict[str, str] = field(default_factory=dict)

    def paths(self) -> list[str]:
        return [n.path for n in self.nodes]


def build_delta_node_catalog(template: type[BaseModel]) -> DeltaNodeCatalog:
    """Build a Delta-specific catalog from template schema."""
    nodes: list[DeltaNodeSpec] = []
    field_aliases: dict[str, str] = {}

    def add_node(
        path: str,
        model: type[BaseModel],
        parent_path: str,
        field_name: str,
        is_list: bool,
        field_description: str = "",
        field_example_hint: str = "",
        identity_example_values: list[str] | None = None,
    ) -> None:
        model_doc = (model.__doc__ or "").strip()
        description = field_description or model_doc
        nodes.append(
            DeltaNodeSpec(
                path=path,
                node_type=getattr(model, "__name__", "Unknown"),
                id_fields=_get_id_fields(model),
                kind=("component" if _is_component(model) else "entity"),
                parent_path=parent_path,
                field_name=field_name,
                is_list=is_list,
                property_fields=_model_property_fields(model),
                description=description[:300],
                example_hint=field_example_hint,
                identity_example_values=identity_example_values or [],
            )
        )

    def walk(
        path_prefix: str, model: type[BaseModel], parent_entity_path: str, from_root: bool
    ) -> None:
        if from_root:
            add_node("", model, "", "", False)

        for field_name, field_info in model.model_fields.items():
            for alias_name in _field_aliases(field_name, field_info):
                field_aliases.setdefault(alias_name, field_name)

            segment = f".{field_name}" if path_prefix else field_name
            path = f"{path_prefix}{segment}" if path_prefix else field_name
            target_model = _unwrap_model_from_annotation(field_info.annotation)
            origin = get_origin(field_info.annotation)
            if target_model is None:
                continue

            is_entity_child = _is_entity(target_model)
            is_component_child = _is_component(target_model)
            include_child = is_entity_child or is_component_child

            if include_child:
                field_description = _field_description(field_info)
                id_fields_for_hint = _get_id_fields(target_model) if is_entity_child else None
                field_example_hint = _field_example_hint(field_info, id_fields_for_hint)
                identity_vals = (
                    _identity_example_values_from_field(field_info, id_fields_for_hint)
                    if id_fields_for_hint
                    else []
                )
                if origin is list:
                    list_path = f"{path}[]"
                    add_node(
                        list_path,
                        target_model,
                        parent_entity_path,
                        field_name,
                        True,
                        field_description=field_description,
                        field_example_hint=field_example_hint,
                        identity_example_values=identity_vals,
                    )
                    next_entity_path = list_path if is_entity_child else parent_entity_path
                    walk(list_path, target_model, next_entity_path, from_root=False)
                else:
                    add_node(
                        path,
                        target_model,
                        parent_entity_path,
                        field_name,
                        False,
                        field_description=field_description,
                        field_example_hint=field_example_hint,
                        identity_example_values=identity_vals,
                    )
                    next_entity_path = path if is_entity_child else parent_entity_path
                    walk(path, target_model, next_entity_path, from_root=False)
            else:
                if origin is list:
                    walk(f"{path}[]", target_model, parent_entity_path, from_root=False)
                else:
                    walk(path, target_model, parent_entity_path, from_root=False)

    walk("", template, "", from_root=True)
    return DeltaNodeCatalog(nodes=nodes, field_aliases=field_aliases)


def _id_tuple(
    spec: DeltaNodeSpec, ids: dict[str, Any], instance_key: str | None = None
) -> tuple[Any, ...]:
    if not spec.id_fields:
        return (instance_key or "",)
    return tuple(ids.get(f) for f in spec.id_fields)


def _canonicalize_id_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = " ".join(value.strip().split()).casefold()
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return str(value)


def merge_delta_filled_into_root(  # noqa: C901
    path_filled: dict[str, list[Any]],
    path_descriptors: dict[str, list[dict[str, Any]]],
    catalog: DeltaNodeCatalog,
    *,
    stats: dict[str, int | list[Any]] | None = None,
    salvage_orphans: bool = True,
    orphan_field_name: str = "__orphans__",
) -> dict[str, Any]:
    """Attach filled nodes to parent descriptors and build root object."""
    root: dict[str, Any] = {}
    merge_counters: dict[str, int] = {
        "descriptor_length_mismatch": 0,
        "non_dict_filled_objects": 0,
        "missing_parent_descriptor": 0,
        "parent_lookup_miss": 0,
        "attached_list_items": 0,
        "attached_scalar_items": 0,
        "orphan_attached": 0,
        "orphan_dropped": 0,
        "parent_lookup_repaired_local_id": 0,
        "parent_lookup_repaired_single_candidate": 0,
        "parent_lookup_repaired_positional": 0,
        "parent_lookup_repaired_canonical_id": 0,
        "parent_lookup_repaired_best_effort": 0,
    }
    parent_lookup_miss_examples: list[dict[str, Any]] = []
    missing_parent_examples: list[dict[str, Any]] = []
    spec_by_path = {spec.path: spec for spec in catalog.nodes}
    lookup: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    lookup_by_path: dict[str, list[dict[str, Any]]] = {}
    lookup_entries_by_path: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {}

    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        if len(filled_list) != len(descriptors):
            merge_counters["descriptor_length_mismatch"] += 1
        for i, obj in enumerate(filled_list):
            if not isinstance(obj, dict):
                merge_counters["non_dict_filled_objects"] += 1
                continue
            desc = descriptors[i] if i < len(descriptors) else {}
            ids = desc.get("ids") or {}
            instance_key = desc.get("__instance_key") if isinstance(desc, dict) else None
            key = (path, _id_tuple(spec, ids, instance_key=instance_key))
            lookup[key] = obj
            lookup_by_path.setdefault(path, []).append(obj)
            lookup_entries_by_path.setdefault(path, []).append((key[1], obj))

    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        if not filled_list:
            continue
        if path == "":
            if filled_list and isinstance(filled_list[0], dict):
                root.update(filled_list[0])
            continue
        parent_path = spec.parent_path
        field_name = spec.field_name
        is_list = spec.is_list
        if not field_name:
            continue
        if parent_path == "":
            if is_list:
                root[field_name] = filled_list
            else:
                root[field_name] = filled_list[0] if filled_list else None
            continue
        parent_spec = spec_by_path.get(parent_path)
        if not parent_spec:
            continue
        for i, obj in enumerate(filled_list):
            if not isinstance(obj, dict):
                merge_counters["non_dict_filled_objects"] += 1
                continue
            desc = descriptors[i] if i < len(descriptors) else {}
            parent = desc.get("parent")
            if not parent or not isinstance(parent, dict):
                merge_counters["missing_parent_descriptor"] += 1
                if len(missing_parent_examples) < 20:
                    missing_parent_examples.append({"path": path, "parent_path": parent_path})
                if salvage_orphans:
                    root.setdefault(orphan_field_name, []).append(
                        {"path": path, "parent_path": parent_path, "data": obj}
                    )
                    merge_counters["orphan_attached"] += 1
                else:
                    merge_counters["orphan_dropped"] += 1
                    logger.warning(
                        "[DeltaProjection] Dropped orphan node: path=%s parent_path=%s reason=missing_parent_descriptor",
                        path,
                        parent_path,
                    )
                continue
            parent_ids = parent.get("ids") or {}
            parent_instance_key = parent.get("__instance_key") if isinstance(parent, dict) else None
            parent_key = (
                parent_path,
                _id_tuple(parent_spec, parent_ids, instance_key=parent_instance_key),
            )
            parent_obj = lookup.get(parent_key)
            if parent_obj is None:
                if parent_ids:
                    for fid, raw_val in parent_ids.items():
                        sval = str(raw_val)
                        if fid not in LOCAL_ID_FIELD_HINTS or not sval.isdigit():
                            continue
                        ival = int(sval)
                        repaired = None
                        for delta in (1, -1):
                            candidate = dict(parent_ids)
                            candidate[fid] = str(ival + delta)
                            candidate_key = (
                                parent_path,
                                _id_tuple(parent_spec, candidate, instance_key=parent_instance_key),
                            )
                            repaired = lookup.get(candidate_key)
                            if repaired is not None:
                                break
                        if repaired is not None:
                            parent_obj = repaired
                            merge_counters["parent_lookup_repaired_local_id"] += 1
                            break
                if parent_obj is None:
                    if parent_ids and parent_spec.id_fields:
                        canonical_candidates: list[dict[str, Any]] = []
                        for candidate_tuple, candidate_obj in lookup_entries_by_path.get(
                            parent_path, []
                        ):
                            candidate_ok = True
                            for idx, field_name in enumerate(parent_spec.id_fields):
                                parent_val = parent_ids.get(field_name)
                                candidate_val = (
                                    candidate_tuple[idx] if idx < len(candidate_tuple) else None
                                )
                                if parent_val in (None, "") or candidate_val in (None, ""):
                                    continue
                                if _canonicalize_id_value(parent_val) != _canonicalize_id_value(
                                    candidate_val
                                ):
                                    candidate_ok = False
                                    break
                            if candidate_ok:
                                canonical_candidates.append(candidate_obj)
                        if len(canonical_candidates) == 1:
                            parent_obj = canonical_candidates[0]
                            merge_counters["parent_lookup_repaired_canonical_id"] += 1
                if parent_obj is None:
                    candidates = lookup_by_path.get(parent_path, [])
                    if len(candidates) == 1:
                        parent_obj = candidates[0]
                        merge_counters["parent_lookup_repaired_single_candidate"] += 1
                if parent_obj is None and not parent_ids:
                    parent_candidates = path_filled.get(parent_path, [])
                    if 0 <= i < len(parent_candidates):
                        positional_parent = parent_candidates[i]
                        if isinstance(positional_parent, dict):
                            parent_obj = positional_parent
                            merge_counters["parent_lookup_repaired_positional"] += 1
                if parent_obj is None:
                    candidates = lookup_by_path.get(parent_path, [])
                    if candidates:
                        parent_obj = candidates[0]
                        merge_counters["parent_lookup_repaired_best_effort"] += 1
            if parent_obj is None:
                merge_counters["parent_lookup_miss"] += 1
                if len(parent_lookup_miss_examples) < 20:
                    parent_lookup_miss_examples.append(
                        {"path": path, "parent_path": parent_path, "parent_ids": dict(parent_ids)}
                    )
                if salvage_orphans:
                    root.setdefault(orphan_field_name, []).append(
                        {"path": path, "parent_path": parent_path, "data": obj}
                    )
                    merge_counters["orphan_attached"] += 1
                else:
                    merge_counters["orphan_dropped"] += 1
                    logger.warning(
                        "[DeltaProjection] Dropped orphan node: path=%s parent_path=%s reason=parent_lookup_miss",
                        path,
                        parent_path,
                    )
                continue
            if is_list:
                existing = parent_obj.get(field_name)
                if not isinstance(existing, list):
                    parent_obj[field_name] = []
                parent_obj[field_name].append(obj)
                merge_counters["attached_list_items"] += 1
            else:
                parent_obj[field_name] = obj
                merge_counters["attached_scalar_items"] += 1
    root_level_count = 0
    for spec in catalog.nodes:
        if spec.parent_path != "":
            continue
        filled_list = path_filled.get(spec.path, [])
        if spec.is_list:
            root_level_count += len(filled_list)
        elif filled_list:
            root_level_count += 1
    merge_counters["attached_node_count"] = (
        1
        + root_level_count
        + merge_counters["attached_list_items"]
        + merge_counters["attached_scalar_items"]
    )
    if stats is not None:
        stats.update(merge_counters)
        stats["parent_lookup_miss_examples"] = parent_lookup_miss_examples
        stats["missing_parent_examples"] = missing_parent_examples
    if merge_counters["parent_lookup_miss"] > 0:
        logger.warning(
            "[DeltaProjection] Parent lookup misses=%s (salvage_orphans=%s)",
            merge_counters["parent_lookup_miss"],
            salvage_orphans,
        )
    if merge_counters["missing_parent_descriptor"] > 0:
        logger.warning(
            "[DeltaProjection] Missing parent descriptors=%s (salvage_orphans=%s)",
            merge_counters["missing_parent_descriptor"],
            salvage_orphans,
        )
    return root
