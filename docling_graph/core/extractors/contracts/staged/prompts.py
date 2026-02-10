"""
Prompt, planning, and quality helpers for the staged extraction contract.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

PLACEHOLDER_VALUE_PATTERNS = (
    r"\.\.\.$",
    r"^sample\b",
    r"\bexample\b",
    r"^value$",
    r"^unknown$",
    r"^n/?a$",
    r"^tbd$",
    r"^placeholder$",
    r"^null$",
    r"^none$",
    r"^-$",
)

PLACEHOLDER_ID_PATTERNS = (
    r"-SAMPLE-",
    r"-EXAMPLE-",
    r"-TEST-",
    r"-000$",
    r"-001$",
)


@dataclass(frozen=True)
class TemplateGraphMetadata:
    root_identity_fields: list[str]
    root_edge_fields: dict[str, str]
    root_entity_identity_fields: dict[str, list[str]]
    # Path-keyed identity for nested entity lists, e.g. {"studies": [...], "studies.experiments": [...]}
    nested_entity_identity_fields: dict[str, list[str]]
    # For each root list field: nested edge targets inside list items: (child_field_name, target_identity_fields)
    nested_edge_targets: dict[str, list[tuple[str, list[str]]]]


@dataclass(frozen=True)
class ExtractionFieldPlan:
    skeleton_fields: list[str]
    groups: list[list[str]]
    critical_fields: list[str]


@dataclass(frozen=True)
class QualityIssue:
    field_path: tuple[str | int, ...]
    reason: Literal["missing", "placeholder", "empty", "identity_missing", "edge_missing"]
    severity: Literal["critical", "warning"]

    @property
    def root_field(self) -> str:
        for token in self.field_path:
            if isinstance(token, str):
                return token
        return ""


@dataclass
class QualityReport:
    issues: list[QualityIssue]

    def root_fields(self) -> list[str]:
        ordered = OrderedDict[str, None]()
        for issue in self.issues:
            root = issue.root_field
            if root:
                ordered[root] = None
        return list(ordered.keys())

    def nested_path_obligations(
        self,
        metadata: TemplateGraphMetadata,
    ) -> list[tuple[str, list[str]]]:
        """
        From issues with path length > 1 at known nested edge targets, normalize to
        path patterns (e.g. "line_items[].item") with target identity fields.
        """
        if not metadata.nested_edge_targets:
            return []
        obligations: dict[str, list[str]] = {}
        for issue in self.issues:
            path = issue.field_path
            if len(path) < 2:
                continue
            root = issue.root_field
            targets = metadata.nested_edge_targets.get(root)
            if not targets:
                continue
            child_names = {t[0] for t in targets}
            pattern_parts: list[str] = []
            for token in path:
                if isinstance(token, str):
                    pattern_parts.append(token)
                else:
                    pattern_parts.append("[]")
            pattern = ".".join(pattern_parts).replace(".[]", "[]")
            last_str = next((t for t in reversed(path) if isinstance(t, str)), None)
            if last_str not in child_names:
                continue
            target_ids = next(ids for name, ids in targets if name == last_str)
            if pattern not in obligations:
                obligations[pattern] = target_ids
        return list(obligations.items())

    def improved_over(self, previous: QualityReport | None) -> bool:
        if previous is None:
            return True
        prev_critical = sum(1 for i in previous.issues if i.severity == "critical")
        curr_critical = sum(1 for i in self.issues if i.severity == "critical")
        if curr_critical < prev_critical:
            return True
        if curr_critical == prev_critical:
            return len(self.issues) < len(previous.issues)
        return False


def _load_schema(schema_json: str) -> dict[str, Any]:
    parsed = json.loads(schema_json)
    if not isinstance(parsed, dict):
        raise ValueError("Schema JSON must represent an object.")
    return parsed


def _root_properties(schema: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties", {})
    return props if isinstance(props, dict) else {}


def _root_required(schema: dict[str, Any]) -> list[str]:
    required = schema.get("required", [])
    return [str(x) for x in required] if isinstance(required, list) else []


def _is_object_like(defn: Any) -> bool:
    if not isinstance(defn, dict):
        return False
    if isinstance(defn.get("$ref"), str):
        return True
    field_type = defn.get("type")
    if field_type in {"object", "array"}:
        return True
    for option in defn.get("anyOf", []):
        if isinstance(option, dict) and (
            option.get("type") in {"object", "array"} or option.get("$ref")
        ):
            return True
    return False


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


def _collect_nested_entity_identity_paths(
    model: type[BaseModel],
    path_prefix: str,
) -> dict[str, list[str]]:
    """Recursively build path -> identity fields for all entity lists (root and nested)."""
    result: dict[str, list[str]] = {}
    for field_name, field_info in model.model_fields.items():
        nested = _unwrap_model_from_annotation(field_info.annotation)
        if nested is None:
            continue
        nested_cfg = getattr(nested, "model_config", {}) or {}
        nested_ids = (
            nested_cfg.get("graph_id_fields", []) if isinstance(nested_cfg, dict) else []
        )
        ids = [f for f in nested_ids if isinstance(f, str)]
        path = f"{path_prefix}.{field_name}" if path_prefix else field_name
        if ids:
            result[path] = ids
        result.update(_collect_nested_entity_identity_paths(nested, path))
    return result


def _single_object_edge_targets(item_model: type[BaseModel]) -> list[tuple[str, list[str]]]:
    """
    For a model that is the item type of a list (e.g. LineItem), return (field_name, target_identity_fields)
    for each field that is a single-object edge (edge_label + non-list model with graph_id_fields).
    """
    result: list[tuple[str, list[str]]] = []
    for field_name, field_info in item_model.model_fields.items():
        extra = field_info.json_schema_extra or {}
        if not isinstance(extra, dict) or not extra.get("edge_label"):
            continue
        target = _unwrap_model_from_annotation(field_info.annotation)
        if target is None:
            continue
        origin = get_origin(field_info.annotation)
        if origin is list:
            continue
        target_cfg = getattr(target, "model_config", {}) or {}
        target_ids = (
            target_cfg.get("graph_id_fields", []) if isinstance(target_cfg, dict) else []
        )
        ids = [f for f in target_ids if isinstance(f, str)]
        if ids:
            result.append((field_name, ids))
    return result


def _collect_nested_edge_targets(
    template: type[BaseModel],
    prop_names: set[str],
) -> dict[str, list[tuple[str, list[str]]]]:
    """
    For each root field that is a list of entities, collect nested single-object edge targets
    (child_field_name, target_identity_fields) so we can materialize or prompt for them.
    """
    result: dict[str, list[tuple[str, list[str]]]] = {}
    for field_name, field_info in template.model_fields.items():
        if field_name not in prop_names:
            continue
        origin = get_origin(field_info.annotation)
        if origin is not list:
            continue
        item_model = _unwrap_model_from_annotation(field_info.annotation)
        if item_model is None:
            continue
        targets = _single_object_edge_targets(item_model)
        if targets:
            result[field_name] = targets
    return result


def get_template_graph_metadata(
    template: type[BaseModel] | None,
    schema_json: str,
) -> TemplateGraphMetadata:
    schema = _load_schema(schema_json)
    props = _root_properties(schema)
    prop_names = set(props.keys())
    if template is None:
        return TemplateGraphMetadata([], {}, {}, {}, {})

    model_config = getattr(template, "model_config", {}) or {}
    graph_id_fields = model_config.get("graph_id_fields", []) if isinstance(model_config, dict) else []
    root_identity_fields = [f for f in graph_id_fields if isinstance(f, str) and f in prop_names]

    edge_fields: dict[str, str] = {}
    entity_identity_by_root: dict[str, list[str]] = {}
    for field_name, field_info in template.model_fields.items():
        if field_name not in prop_names:
            continue
        extra = field_info.json_schema_extra
        if isinstance(extra, dict) and isinstance(extra.get("edge_label"), str):
            edge_fields[field_name] = extra["edge_label"]

        nested_model = _unwrap_model_from_annotation(field_info.annotation)
        if nested_model is None:
            continue
        nested_cfg = getattr(nested_model, "model_config", {}) or {}
        nested_ids = (
            nested_cfg.get("graph_id_fields", []) if isinstance(nested_cfg, dict) else []
        )
        ids = [f for f in nested_ids if isinstance(f, str)]
        if ids:
            entity_identity_by_root[field_name] = ids

    nested_paths = _collect_nested_entity_identity_paths(template, "")
    nested_edge_targets = _collect_nested_edge_targets(template, prop_names)

    return TemplateGraphMetadata(
        root_identity_fields=root_identity_fields,
        root_edge_fields=edge_fields,
        root_entity_identity_fields=entity_identity_by_root,
        nested_entity_identity_fields=nested_paths,
        nested_edge_targets=nested_edge_targets,
    )


def plan_extraction_passes(
    schema_json: str,
    template_metadata: TemplateGraphMetadata | None = None,
    max_fields_per_group: int = 6,
    max_skeleton_fields: int = 10,
) -> ExtractionFieldPlan:
    schema = _load_schema(schema_json)
    props = _root_properties(schema)
    required = _root_required(schema)
    if not props:
        return ExtractionFieldPlan([], [], [])

    metadata = template_metadata or TemplateGraphMetadata([], {}, {}, {}, {})
    ordered = OrderedDict[str, None]()
    for field in required:
        if field in props:
            ordered[field] = None
    for field in metadata.root_identity_fields:
        if field in props:
            ordered[field] = None
    for field in metadata.root_edge_fields:
        if field in props:
            ordered[field] = None

    id_like_tokens = ("id", "number", "type", "title", "name", "date", "currency")
    for field in props:
        lowered = field.lower()
        if any(token in lowered for token in id_like_tokens):
            ordered[field] = None

    skeleton_fields = list(ordered.keys())[:max_skeleton_fields]

    remaining = [field for field in props.keys() if field not in skeleton_fields]
    scalar_fields = [field for field in remaining if not _is_object_like(props[field])]
    object_fields = [field for field in remaining if _is_object_like(props[field])]

    groups: list[list[str]] = []
    for i in range(0, len(scalar_fields), max_fields_per_group):
        groups.append(scalar_fields[i : i + max_fields_per_group])
    for field in object_fields:
        groups.append([field])

    critical = OrderedDict[str, None]()
    for field in required:
        if field in props:
            critical[field] = None
    for field in metadata.root_identity_fields:
        if field in props:
            critical[field] = None
    for field in metadata.root_edge_fields:
        if field in props:
            critical[field] = None

    return ExtractionFieldPlan(
        skeleton_fields=skeleton_fields,
        groups=[group for group in groups if group],
        critical_fields=list(critical.keys()),
    )


def get_root_field_groups(schema_json: str, max_fields_per_group: int = 6) -> list[list[str]]:
    """Backward-compatible root grouping over all fields."""
    schema = _load_schema(schema_json)
    props = _root_properties(schema)
    if not props:
        return []
    scalar_fields = [name for name, defn in props.items() if not _is_object_like(defn)]
    object_fields = [name for name, defn in props.items() if _is_object_like(defn)]
    groups: list[list[str]] = []
    for i in range(0, len(scalar_fields), max_fields_per_group):
        groups.append(scalar_fields[i : i + max_fields_per_group])
    groups.extend([[field] for field in object_fields])
    return [group for group in groups if group]


def get_skeleton_fields(schema_json: str, max_fields: int = 10) -> list[str]:
    """Backward-compatible skeleton selection without template metadata."""
    plan = plan_extraction_passes(
        schema_json=schema_json,
        template_metadata=None,
        max_fields_per_group=6,
        max_skeleton_fields=max_fields,
    )
    return plan.skeleton_fields


def build_root_subschema(schema_json: str, selected_fields: list[str]) -> str:
    schema = _load_schema(schema_json)
    props = _root_properties(schema)
    required = _root_required(schema)
    selected = [field for field in selected_fields if field in props]
    reduced_schema = dict(schema)
    reduced_schema["properties"] = {field: props[field] for field in selected}
    reduced_schema["required"] = [field for field in required if field in selected]
    return json.dumps(reduced_schema, indent=2)


def _base_instructions() -> str:
    return (
        "1. Extract ONLY values evidenced in the provided text.\n"
        "2. Return ONLY valid JSON with keys from the provided schema.\n"
        "3. Do not invent values, IDs, dates, or numeric series.\n"
        "4. Omit fields that are not evidenced in text.\n"
        "5. Keep identifiers consistent across all extraction passes.\n"
    )


def _render_prior_context(prior_extraction: dict[str, Any] | None) -> str:
    if not prior_extraction:
        return ""
    rendered = json.dumps(prior_extraction, ensure_ascii=True, default=str)
    if len(rendered) > 4000:
        rendered = rendered[:4000] + "...(truncated)"
    return f"\n=== PRIOR EXTRACTION ===\n{rendered}\n=== END PRIOR ===\n"


def _render_nested_edge_hints(
    nested_edge_hints: dict[str, list[tuple[str, list[str]]]] | None,
    field_names: list[str],
) -> str:
    """One-line instruction per root field that has nested edge targets."""
    if not nested_edge_hints or not field_names:
        return ""
    lines: list[str] = []
    for root_field in field_names:
        targets = nested_edge_hints.get(root_field)
        if not targets:
            continue
        for nested_field, identity_fields in targets:
            ids_str = ", ".join(identity_fields)
            lines.append(
                f"For each element of `{root_field}`, also populate the nested object "
                f"`{nested_field}` with at least: {ids_str} "
                "(use the element's matching scalar if present, e.g. item_code → item.item_code)."
            )
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def get_skeleton_prompt(
    markdown_content: str,
    schema_json: str,
    anchor_fields: list[str] | None = None,
    prior_extraction: dict[str, Any] | None = None,
    nested_edge_hints: dict[str, list[tuple[str, list[str]]]] | None = None,
) -> dict[str, str]:
    anchor_text = ", ".join(anchor_fields or [])
    nested_text = _render_nested_edge_hints(nested_edge_hints, anchor_fields or [])
    system_prompt = (
        "You are a precise extraction assistant running a skeleton pass.\n\n"
        "Focus on stable anchors and high-confidence top-level values.\n"
        f"Anchor fields: {anchor_text or 'schema-required anchors'}\n"
        f"{nested_text}"
        f"Instructions:\n{_base_instructions()}\n"
        "This pass is intentionally conservative: skip uncertain fields.\n"
        "Important: response MUST be valid JSON."
    )
    user_prompt = (
        "Extract a conservative skeleton from this document.\n\n"
        "=== DOCUMENT ===\n"
        f"{markdown_content}\n"
        "=== END DOCUMENT ===\n"
        f"{_render_prior_context(prior_extraction)}\n"
        "=== TARGET SCHEMA ===\n"
        f"{schema_json}\n"
        "=== END SCHEMA ===\n\n"
        "Return only JSON."
    )
    return {"system": system_prompt, "user": user_prompt}


def get_group_prompt(
    markdown_content: str,
    schema_json: str,
    group_name: str,
    focus_fields: list[str],
    prior_extraction: dict[str, Any] | None = None,
    critical_fields: list[str] | None = None,
    nested_edge_hints: dict[str, list[tuple[str, list[str]]]] | None = None,
) -> dict[str, str]:
    focus_text = ", ".join(focus_fields) if focus_fields else "selected fields"
    critical_text = ", ".join(critical_fields or [])
    nested_text = _render_nested_edge_hints(nested_edge_hints, focus_fields)
    system_prompt = (
        "You are a precise extraction assistant running a focused group pass.\n\n"
        f"Current focus: {focus_text}\n"
        f"Critical fields (if present in schema): {critical_text or 'none'}\n"
        f"{nested_text}"
        f"Instructions:\n{_base_instructions()}\n"
        "Prioritize completeness for the focused fields while remaining evidence-bound.\n"
        "Important: response MUST be valid JSON."
    )
    user_prompt = (
        f"Extract values for group '{group_name}'.\n\n"
        "=== DOCUMENT ===\n"
        f"{markdown_content}\n"
        "=== END DOCUMENT ===\n"
        f"{_render_prior_context(prior_extraction)}\n"
        "=== TARGET SCHEMA ===\n"
        f"{schema_json}\n"
        "=== END SCHEMA ===\n\n"
        "Return only JSON."
    )
    return {"system": system_prompt, "user": user_prompt}


def get_repair_prompt(
    markdown_content: str,
    schema_json: str,
    failed_fields: list[str],
    prior_extraction: dict[str, Any] | None = None,
    issue_summary: str | None = None,
    nested_obligations: list[tuple[str, list[str]]] | None = None,
) -> dict[str, str]:
    fields_text = ", ".join(failed_fields) if failed_fields else "missing fields"
    checklist_parts: list[str] = []
    if nested_obligations:
        checklist_parts.append(
            "You must populate the following nested structures (currently missing or empty):"
        )
        for pattern, identity_fields in nested_obligations:
            ids_str = ", ".join(identity_fields)
            checklist_parts.append(f"- `{pattern}`: object with at least {ids_str} (and name if available).")
        checklist_parts.append("")
    checklist_text = "\n".join(checklist_parts) if checklist_parts else ""
    system_prompt = (
        "You are a precise extraction assistant running a targeted repair pass.\n\n"
        f"Repair targets: {fields_text}\n"
        f"Issue summary: {issue_summary or 'quality issues detected'}\n"
        f"{checklist_text}"
        f"Instructions:\n{_base_instructions()}\n"
        "Repair only the target fields. Keep output minimal and faithful.\n"
        "Important: response MUST be valid JSON."
    )
    user_prompt = (
        "Repair missing/low-quality fields from this document.\n\n"
        "=== DOCUMENT ===\n"
        f"{markdown_content}\n"
        "=== END DOCUMENT ===\n"
        f"{_render_prior_context(prior_extraction)}\n"
        "=== TARGET SCHEMA ===\n"
        f"{schema_json}\n"
        "=== END SCHEMA ===\n\n"
        "Return only JSON."
    )
    return {"system": system_prompt, "user": user_prompt}


def get_consolidation_prompt(
    markdown_content: str,
    schema_json: str,
    current_extraction: dict[str, Any],
    target_fields: list[str],
    issue_summary: str,
    identity_hint: str = "",
) -> dict[str, str]:
    """Build prompt for LLM-backed conflict resolution / consolidation."""
    excerpt = markdown_content[:12000] + "..." if len(markdown_content) > 12000 else markdown_content
    current_str = json.dumps(current_extraction, indent=2, ensure_ascii=True, default=str)
    if len(current_str) > 8000:
        current_str = current_str[:8000] + "\n...(truncated)"
    fields_text = ", ".join(target_fields) if target_fields else "all fields"
    system_prompt = (
        "You are a precise extraction assistant performing conflict resolution.\n\n"
        "You are given the current extracted JSON and the source document. "
        "Resolve remaining issues (missing required fields, placeholder values, or conflicts) "
        "using evidence from the document.\n\n"
        f"Target fields: {fields_text}\n"
        f"{identity_hint}\n"
        f"Instructions:\n{_base_instructions()}\n"
        "Return ONLY valid JSON with resolved fields. Response MUST be valid JSON."
    )
    user_prompt = (
        "Resolve extraction issues using the document and current extraction below.\n\n"
        f"Issue summary: {issue_summary}\n\n"
        "=== DOCUMENT (excerpt) ===\n"
        f"{excerpt}\n"
        "=== END DOCUMENT ===\n\n"
        "=== CURRENT EXTRACTION ===\n"
        f"{current_str}\n"
        "=== END CURRENT ===\n\n"
        "=== TARGET SCHEMA ===\n"
        f"{schema_json}\n"
        "=== END SCHEMA ===\n\n"
        "Return only JSON with resolved values for the target fields."
    )
    return {"system": system_prompt, "user": user_prompt}


def _contains_placeholder_text(value: str) -> bool:
    compact = value.strip().lower()
    for pattern in PLACEHOLDER_VALUE_PATTERNS:
        if re.search(pattern, compact):
            return True
    return False


def _contains_placeholder_id(value: str) -> bool:
    for pattern in PLACEHOLDER_ID_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return True
    return False


def _walk_quality_issues(
    value: Any,
    path: tuple[str | int, ...],
    issues: list[QualityIssue],
    max_depth: int,
) -> None:
    if max_depth < 0:
        return
    if value in (None, "", [], {}):
        issues.append(QualityIssue(path, reason="empty", severity="warning"))
        return
    if isinstance(value, str):
        if _contains_placeholder_text(value):
            issues.append(QualityIssue(path, reason="placeholder", severity="warning"))
        if _contains_placeholder_id(value):
            issues.append(QualityIssue(path, reason="placeholder", severity="critical"))
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _walk_quality_issues(item, (*path, idx), issues, max_depth - 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_quality_issues(item, (*path, key), issues, max_depth - 1)


def assess_quality(
    candidate_data: dict[str, Any],
    schema_json: str,
    critical_fields: list[str] | None = None,
    max_depth: int = 3,
) -> QualityReport:
    schema = _load_schema(schema_json)
    props = _root_properties(schema)
    required = _root_required(schema)
    critical = set(critical_fields or []) | set(required)
    issues: list[QualityIssue] = []

    for field in required:
        value = candidate_data.get(field)
        if value in (None, "", [], {}):
            issues.append(QualityIssue((field,), reason="missing", severity="critical"))

    for field in critical:
        if field not in props:
            continue
        if field not in candidate_data:
            issues.append(QualityIssue((field,), reason="identity_missing", severity="critical"))

    for field, value in candidate_data.items():
        if field not in props:
            continue
        _walk_quality_issues(value, (field,), issues, max_depth=max_depth)

    # Flag edge fields with empty values as critical.
    for field, prop_def in props.items():
        if isinstance(prop_def, dict):
            edge_label = prop_def.get("edge_label")
            if edge_label and candidate_data.get(field) in (None, "", [], {}):
                issues.append(QualityIssue((field,), reason="edge_missing", severity="critical"))

    # Deduplicate by (path, reason).
    seen: set[tuple[tuple[str | int, ...], str]] = set()
    deduped: list[QualityIssue] = []
    for issue in issues:
        key = (issue.field_path, issue.reason)
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    return QualityReport(deduped)


def detect_quality_issues(
    candidate_data: dict[str, Any],
    schema_json: str,
    critical_fields: list[str] | None = None,
    max_depth: int = 3,
) -> list[str]:
    """
    Backward-compatible helper returning root fields with quality issues.
    """
    report = assess_quality(
        candidate_data=candidate_data,
        schema_json=schema_json,
        critical_fields=critical_fields,
        max_depth=max_depth,
    )
    return report.root_fields()

