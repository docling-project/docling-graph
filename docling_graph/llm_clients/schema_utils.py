"""JSON-schema utilities for structured output and prompt guidance."""

from __future__ import annotations

import copy
from typing import Any, Literal

DEFAULT_RESPONSE_SCHEMA_NAME = "extraction_result"


def normalize_schema_for_response_format(
    schema: dict[str, Any],
    *,
    top_level: Literal["object", "array"] = "object",
    name: str = DEFAULT_RESPONSE_SCHEMA_NAME,
) -> dict[str, Any]:
    """Prepare a provider-safe JSON schema payload for LiteLLM."""
    normalized = copy.deepcopy(schema)
    for key in ("title", "examples"):
        normalized.pop(key, None)

    if top_level == "array":
        if normalized.get("type") != "array":
            normalized = {"type": "array", "items": normalized}
    else:
        if normalized.get("type") not in ("object", None):
            if "properties" in normalized or "required" in normalized:
                normalized.setdefault("type", "object")
            else:
                normalized = {
                    "type": "object",
                    "properties": normalized.get("properties", {}),
                    "required": normalized.get("required", []),
                }

    if "type" not in normalized and "properties" in normalized:
        normalized["type"] = "object"

    return {"name": name, "schema": normalized, "strict": True}


def build_compact_semantic_guide(
    schema: dict[str, Any],
    *,
    max_chars_per_field: int = 250,
    max_total_chars: int = 4000,
    max_depth: int = 3,
) -> str:
    """Build compact field guidance from schema metadata."""
    parts: list[str] = []
    total = 0
    _defs = schema.get("$defs")
    defs: dict[str, Any] = _defs if isinstance(_defs, dict) else {}

    def _resolve_ref(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return node
        if ref.startswith("#/$defs/"):
            key = ref.split("/")[-1]
            resolved = defs.get(key)
            if isinstance(resolved, dict):
                return resolved
        return node

    def _line_for(path: str, prop: dict[str, Any], required: bool) -> str:
        line_parts = [f"- **{path}**"]
        if required:
            line_parts.append("(required)")
        desc = (prop.get("description") or "").strip()[:max_chars_per_field]
        if desc:
            line_parts.append(f": {desc}")
        ex = prop.get("examples")
        if ex and isinstance(ex, list) and ex:
            line_parts.append(f" e.g. {str(ex[0])[:80]}")
        enum_vals = prop.get("enum")
        if enum_vals and isinstance(enum_vals, list):
            joined = ", ".join(str(v) for v in enum_vals[:10])
            line_parts.append(f" One of: {joined}")
        if prop.get("type") == "array":
            line_parts.append(" Array field.")
        if path.endswith(("_id", ".id")) or ".id_" in path:
            line_parts.append(" Identity field; preserve consistency across references.")
        return " ".join(line_parts)

    def _walk(node: dict[str, Any], prefix: str, depth: int) -> None:
        nonlocal total
        if depth > max_depth:
            return
        resolved = _resolve_ref(node)
        _props = resolved.get("properties")
        props: dict[str, Any] = _props if isinstance(_props, dict) else {}
        required_fields = set(resolved.get("required") or [])
        for key, raw_prop in props.items():
            if not isinstance(raw_prop, dict):
                continue
            prop = _resolve_ref(raw_prop)
            path = f"{prefix}.{key}" if prefix else key
            line = _line_for(path, prop, key in required_fields)
            if total + len(line) + 1 > max_total_chars:
                return
            parts.append(line)
            total += len(line) + 1
            if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
                _walk(_resolve_ref(prop["items"]), f"{path}[]", depth + 1)
            elif prop.get("type") == "object" or isinstance(prop.get("properties"), dict):
                _walk(prop, path, depth + 1)

    _walk(schema, "", 1)

    if not parts:
        return "Extract all fields that match the schema enforced by the API. Omit missing values."

    return "Field guidance:\n" + "\n".join(parts)
