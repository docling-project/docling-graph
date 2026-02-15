"""IR normalization for delta batch graph outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .catalog import DeltaNodeCatalog
from .helpers import DedupPolicy


@dataclass
class DeltaIrNormalizerConfig:
    """Options for pre-merge delta IR normalization."""

    validate_paths: bool = True
    canonicalize_ids: bool = True
    strip_nested_properties: bool = True
    attach_provenance: bool = True


def _canonicalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _normalize_ids(
    ids_raw: Any,
    *,
    identity_fields: tuple[str, ...],
    canonicalize: bool,
) -> tuple[dict[str, str], dict[str, int]]:
    stats = {"id_key_mismatch": 0, "id_missing_required": 0}
    if not isinstance(ids_raw, dict):
        ids_raw = {}
    out: dict[str, str] = {}
    if not identity_fields:
        return out, stats
    for key in identity_fields:
        value = ids_raw.get(key)
        if value is None:
            stats["id_missing_required"] += 1
            continue
        txt = _canonicalize_text(value) if canonicalize else str(value)
        if not txt:
            stats["id_missing_required"] += 1
            continue
        out[key] = txt
    extra_keys = [k for k in ids_raw if k not in identity_fields]
    if extra_keys:
        stats["id_key_mismatch"] += 1
    return out, stats


def _backfill_missing_identity_ids(
    *,
    ids_raw: Any,
    identity_fields: tuple[str, ...],
    remapped_properties: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    if not isinstance(ids_raw, dict):
        ids_raw = {}
    if not identity_fields:
        return dict(ids_raw), 0

    repaired = 0
    out = dict(ids_raw)
    for key in identity_fields:
        current = out.get(key)
        if _is_non_empty(current):
            continue
        candidate = remapped_properties.get(key)
        if _is_non_empty(candidate):
            out[key] = candidate
            repaired += 1
    return out, repaired


def _strip_nested_props(properties_raw: Any) -> tuple[dict[str, Any], int]:
    if not isinstance(properties_raw, dict):
        return {}, 0
    clean: dict[str, Any] = {}
    dropped = 0
    for key, value in properties_raw.items():
        if isinstance(value, dict):
            dropped += 1
            continue
        if isinstance(value, list):
            flat: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    dropped += 1
                    continue
                if isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            dropped += 1
                            continue
                        flat.append(_coerce_scalar(sub))
                    continue
                flat.append(_coerce_scalar(item))
            clean[key] = flat
            continue
        clean[key] = _coerce_scalar(value)
    return clean, dropped


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _canonicalize_text(value)
    if not text:
        return text

    currency_match = re.match(r"^[A-Z]{3}\s+([-+]?\d[\d,]*\.?\d*)$", text)
    if currency_match:
        try:
            return float(currency_match.group(1).replace(",", ""))
        except Exception:
            return text

    percent_match = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*%$", text)
    if percent_match:
        try:
            return float(percent_match.group(1))
        except Exception:
            return text

    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    numeric_match = re.match(r"^[-+]?\d+(?:\.\d+)?$", text)
    if numeric_match:
        try:
            return float(text) if "." in text else int(text)
        except Exception:
            return text
    return text


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | dict | tuple | set):
        return len(value) > 0
    return True


def _unknown_field_candidate(path_raw: str) -> str:
    if not path_raw:
        return ""
    candidate = path_raw.strip().replace("/", ".").replace("[", ".").replace("]", ".")
    candidate = candidate.strip(".")
    if not candidate:
        return ""
    segment = candidate.split(".")[-1].strip()
    return _normalize_name(segment)


def _normalize_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _tokenize_name(value: str) -> set[str]:
    normalized = _normalize_name(value)
    if not normalized:
        return set()
    return {part for part in normalized.split("_") if part}


def _best_field_match(candidate: str, allowed_fields: set[str]) -> str | None:
    if not candidate or not allowed_fields:
        return None
    direct = _normalize_name(candidate)
    by_normalized = {_normalize_name(field): field for field in allowed_fields}
    if direct in by_normalized:
        return by_normalized[direct]

    cand_tokens = _tokenize_name(candidate)
    if not cand_tokens:
        return None

    best_field: str | None = None
    best_score = 0.0
    tie = False
    for field in allowed_fields:
        field_tokens = _tokenize_name(field)
        if not field_tokens:
            continue
        overlap = len(cand_tokens & field_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(cand_tokens), len(field_tokens))
        if score > best_score:
            best_score = score
            best_field = field
            tie = False
        elif score == best_score:
            tie = True

    if tie:
        return None
    # Conservative threshold to avoid accidental remaps.
    if best_score >= 0.5:
        return best_field
    return None


def _extract_salvage_value(raw_node: dict[str, Any]) -> Any:
    properties = raw_node.get("properties")
    if isinstance(properties, dict):
        for val in properties.values():
            if _is_non_empty(val):
                return _coerce_scalar(val)
    ids = raw_node.get("ids")
    if isinstance(ids, dict):
        if _is_non_empty(ids.get("value")):
            return _coerce_scalar(ids.get("value"))
        for val in ids.values():
            if _is_non_empty(val):
                return _coerce_scalar(val)
    return None


def _scalar_candidates_from_node(raw_node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ids = raw_node.get("ids")
    if isinstance(ids, dict):
        for key, value in ids.items():
            if _is_non_empty(value):
                out[_normalize_name(str(key))] = _coerce_scalar(value)
    props = raw_node.get("properties")
    if isinstance(props, dict):
        for key, value in props.items():
            if _is_non_empty(value) and not isinstance(value, dict | list):
                out[_normalize_name(str(key))] = _coerce_scalar(value)
    return out


def _collect_catalog_path_prefixes(
    *,
    path_raw: str,
    allowed_paths: set[str],
    field_aliases: dict[str, str],
) -> list[str]:
    candidate = path_raw.strip().replace("/", ".").replace("[", ".").replace("]", ".")
    candidate = candidate.strip(".")
    if not candidate:
        return []
    parts = [p for p in candidate.split(".") if p]
    normalized_parts: list[str] = []
    for part in parts:
        base = part.replace("[]", "")
        normalized_parts.append(field_aliases.get(base, base))

    prefixes: list[str] = []
    for end in range(len(normalized_parts), 0, -1):
        prefix = ".".join(normalized_parts[:end])
        prefix = re.sub(r"\.(\d+)(?=\.|$)", "[]", prefix)
        prefix = re.sub(r"\[(\d+)\]", "[]", prefix)
        if prefix in allowed_paths and prefix not in prefixes:
            prefixes.append(prefix)
        with_list = f"{prefix}[]"
        if with_list in allowed_paths and with_list not in prefixes:
            prefixes.append(with_list)
    if "" in allowed_paths and "" not in prefixes:
        prefixes.append("")
    return prefixes


def _remap_properties_to_catalog_fields(
    *,
    raw_props: Any,
    allowed_fields: set[str],
    field_aliases: dict[str, str],
) -> tuple[dict[str, Any], int]:
    if not isinstance(raw_props, dict):
        return {}, 0
    out: dict[str, Any] = {}
    repaired = 0
    for key, value in raw_props.items():
        key_txt = str(key)
        candidate_key = field_aliases.get(key_txt, key_txt)
        mapped = (
            candidate_key
            if candidate_key in allowed_fields
            else _best_field_match(candidate_key, allowed_fields)
        )
        if mapped is None:
            mapped = _best_field_match(key_txt, allowed_fields)
        if mapped is not None and not isinstance(value, dict | list):
            out[mapped] = _coerce_scalar(value)
            if mapped != key_txt:
                repaired += 1
            continue

        # Nested salvage: flatten scalar leaves using their own keys.
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict | list):
                    continue
                nested_txt = str(nested_key)
                nested_candidate = field_aliases.get(nested_txt, nested_txt)
                nested_mapped = (
                    nested_candidate
                    if nested_candidate in allowed_fields
                    else _best_field_match(nested_candidate, allowed_fields)
                )
                if nested_mapped is None:
                    nested_mapped = _best_field_match(nested_txt, allowed_fields)
                if nested_mapped is None:
                    continue
                out[nested_mapped] = _coerce_scalar(nested_value)
                if nested_mapped != nested_txt:
                    repaired += 1
            continue

        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                for nested_key, nested_value in item.items():
                    if isinstance(nested_value, dict | list):
                        continue
                    nested_txt = str(nested_key)
                    nested_candidate = field_aliases.get(nested_txt, nested_txt)
                    nested_mapped = (
                        nested_candidate
                        if nested_candidate in allowed_fields
                        else _best_field_match(nested_candidate, allowed_fields)
                    )
                    if nested_mapped is None:
                        nested_mapped = _best_field_match(nested_txt, allowed_fields)
                    if nested_mapped is None:
                        continue
                    out[nested_mapped] = _coerce_scalar(nested_value)
                    if nested_mapped != nested_txt:
                        repaired += 1
            continue

        if mapped is None:
            continue
    return out, repaired


def _canonicalize_path(
    path: str, allowed_paths: set[str], field_aliases: dict[str, str] | None = None
) -> tuple[str, bool]:
    """Map common weak-model path variants to catalog paths."""
    original = path
    candidate = path.strip()
    candidate = candidate.replace("/", ".")

    # Common root aliases emitted by weaker models.
    if "" in allowed_paths and candidate in {"document", "root"}:
        return "", True
    if "" in allowed_paths and candidate.startswith(("document.", "root.")):
        if candidate.startswith("document."):
            candidate = candidate[len("document.") :]
        elif candidate.startswith("root."):
            candidate = candidate[len("root.") :]

    if "" in allowed_paths and candidate and candidate[0].isupper() and "." not in candidate:
        return "", True
    if "" in allowed_paths and "." in candidate:
        first_segment, remainder = candidate.split(".", 1)
        if first_segment and first_segment[0].isupper() and remainder:
            candidate = remainder

    # Normalize indexed list syntax:
    # - line_items.1.item -> line_items[].item
    # - line_items[1].item -> line_items[].item
    candidate = re.sub(r"\[(\d+)\]", "[]", candidate)
    candidate = re.sub(r"\.(\d+)(?=\.|$)", "[]", candidate)

    # Canonicalize field-segment aliases (supports AliasChoices / validation_alias).
    if field_aliases:
        normalized_parts: list[str] = []
        for part in [p for p in candidate.split(".") if p]:
            suffix = "[]" if part.endswith("[]") else ""
            base = part[:-2] if suffix else part
            mapped = field_aliases.get(base, base)
            normalized_parts.append(f"{mapped}{suffix}")
        candidate = ".".join(normalized_parts)

    # Normalize list-like terminal segments if catalog expects [] form.
    if (
        candidate not in allowed_paths
        and "[]" not in candidate
        and f"{candidate}[]" in allowed_paths
    ):
        candidate = f"{candidate}[]"

    # Normalize missing [] markers on intermediate list segments by following
    # catalog prefixes (e.g. studies.experiments -> studies[].experiments[]).
    if candidate not in allowed_paths and candidate:
        parts = [p for p in candidate.split(".") if p]
        rebuilt: list[str] = []
        for _idx, part in enumerate(parts):
            if part.endswith("[]"):
                rebuilt.append(part)
                continue
            with_marker = f"{part}[]"
            probe_parts = [*rebuilt, with_marker]
            probe_prefix = ".".join(probe_parts)
            is_valid_prefix = any(
                ap == probe_prefix or ap.startswith(f"{probe_prefix}.") for ap in allowed_paths
            )
            if is_valid_prefix:
                rebuilt.append(with_marker)
            else:
                rebuilt.append(part)
        normalized_candidate = ".".join(rebuilt)
        if normalized_candidate in allowed_paths:
            candidate = normalized_candidate

    # Handle accidental double separators from transformations.
    candidate = candidate.replace("..", ".").strip(".")
    return candidate, candidate != original


def _list_segment_from_path(path: str) -> str | None:
    """Return the last list segment name from a canonical path (without []), if any."""
    if not path:
        return None
    for part in reversed([p for p in path.split(".") if p]):
        if part.endswith("[]"):
            seg = part[:-2].strip()
            if seg:
                return seg
    return None


def _extract_index_for_list_segment(path_raw: str, list_segment: str) -> str | None:
    """Extract list index from raw path for a given segment.

    Examples:
    - document.line_items.3.item -> index=3 for segment line_items
    - document.line_items[3].item -> index=3 for segment line_items
    """
    if not path_raw or not list_segment:
        return None

    pattern = re.compile(rf"(?:^|\.|]){re.escape(list_segment)}(?:\[(\d+)\]|\.([0-9]+))(?=\.|$)")
    matches = list(pattern.finditer(path_raw))
    if not matches:
        return None
    last = matches[-1]
    return last.group(1) or last.group(2)


def _infer_ids_from_index(
    *,
    path_raw: str,
    canonical_path: str,
    ids_raw: Any,
    identity_fields: tuple[str, ...],
) -> tuple[dict[str, Any], bool]:
    """Infer missing identity field from indexed raw path when possible.

    This primarily repairs parent linkage for patterns such as:
    - child path: document.line_items.3.item
    - canonical parent path: line_items[]
    - inferred parent ids: {"line_number": "3"}
    """
    ids = dict(ids_raw) if isinstance(ids_raw, dict) else {}
    if not identity_fields:
        return ids, False
    if all(ids.get(k) not in (None, "") for k in identity_fields):
        return ids, False

    list_segment = _list_segment_from_path(canonical_path)
    if not list_segment:
        return ids, False

    inferred_index = _extract_index_for_list_segment(path_raw, list_segment)
    if inferred_index is None:
        return ids, False

    for field_name in identity_fields:
        if ids.get(field_name) in (None, ""):
            ids[field_name] = str(inferred_index)
            return ids, True
    return ids, False


def normalize_delta_ir_batch_results(  # noqa: C901
    *,
    batch_results: list[dict[str, Any]],
    batch_plan: list[list[tuple[int, str, int]]],
    chunk_metadata: list[dict[str, Any]] | None,
    catalog: DeltaNodeCatalog,
    dedup_policy: dict[str, DedupPolicy],
    config: DeltaIrNormalizerConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize batch IR outputs before global merge."""

    allowed_paths = {spec.path for spec in catalog.nodes}
    {spec.path: spec for spec in catalog.nodes}
    property_fields_by_path = {spec.path: set(spec.property_fields) for spec in catalog.nodes}
    field_aliases = dict(getattr(catalog, "field_aliases", {}) or {})
    parent_by_path = {spec.path: spec.parent_path for spec in catalog.nodes}
    stats: dict[str, Any] = {
        "unknown_path_dropped": 0,
        "path_alias_repaired": 0,
        "id_key_mismatch": 0,
        "id_missing_required": 0,
        "id_backfilled_from_properties": 0,
        "node_id_inferred": 0,
        "parent_id_inferred": 0,
        "nested_property_dropped": 0,
        "parent_ref_repaired": 0,
        "parent_ref_missing": 0,
        "unknown_path_salvaged": 0,
        "salvaged_properties": 0,
        "property_field_repaired": 0,
        "unknown_path_examples": [],
        "id_missing_required_by_path": {},
    }
    normalized_results: list[dict[str, Any]] = []

    for batch_index, graph in enumerate(batch_results):
        chunk_indexes = [chunk_index for chunk_index, _, _ in (batch_plan[batch_index] or [])]
        page_numbers: list[int] = []
        if chunk_metadata:
            for chunk_index in chunk_indexes:
                if 0 <= chunk_index < len(chunk_metadata):
                    pages = chunk_metadata[chunk_index].get("page_numbers")
                    if isinstance(pages, list):
                        page_numbers.extend([int(p) for p in pages if isinstance(p, int | float)])
        page_numbers = sorted(set(page_numbers))
        provenance = {
            "batch_index": batch_index,
            "chunk_indexes": chunk_indexes,
            "page_numbers": page_numbers,
        }

        out_nodes: list[dict[str, Any]] = []
        out_relationships: list[dict[str, Any]] = []
        salvaged_props_by_path: dict[str, dict[str, Any]] = {}

        for node_idx, raw_node in enumerate(graph.get("nodes", [])):
            if not isinstance(raw_node, dict):
                continue
            path_raw = str(raw_node.get("path") or "")
            path, path_repaired = _canonicalize_path(
                path_raw, allowed_paths, field_aliases=field_aliases
            )
            if path_repaired:
                stats["path_alias_repaired"] += 1
            if config.validate_paths and path not in allowed_paths:
                salvaged_any = False
                candidate_value = _extract_salvage_value(raw_node)
                scalar_candidates = _scalar_candidates_from_node(raw_node)
                if _is_non_empty(candidate_value):
                    scalar_candidates.setdefault(
                        _normalize_name(_unknown_field_candidate(path_raw)), candidate_value
                    )
                candidate_paths = _collect_catalog_path_prefixes(
                    path_raw=path_raw,
                    allowed_paths=allowed_paths,
                    field_aliases=field_aliases,
                )
                for candidate_path in candidate_paths:
                    allowed_fields = property_fields_by_path.get(candidate_path, set())
                    if not allowed_fields:
                        continue
                    path_salvaged = salvaged_props_by_path.setdefault(candidate_path, {})
                    for key, value in scalar_candidates.items():
                        if not _is_non_empty(value):
                            continue
                        mapped = (
                            key if key in allowed_fields else _best_field_match(key, allowed_fields)
                        )
                        if mapped and mapped not in path_salvaged:
                            path_salvaged[mapped] = value
                            stats["salvaged_properties"] += 1
                            salvaged_any = True
                    if salvaged_any:
                        break
                if salvaged_any:
                    stats["unknown_path_salvaged"] += 1
                stats["unknown_path_dropped"] += 1
                examples = stats.get("unknown_path_examples")
                if isinstance(examples, list) and len(examples) < 20:
                    examples.append(path_raw)
                continue
            policy = dedup_policy.get(path)
            allowed_fields = property_fields_by_path.get(path, set())
            remapped_props, repaired_fields = _remap_properties_to_catalog_fields(
                raw_props=raw_node.get("properties"),
                allowed_fields=allowed_fields,
                field_aliases=field_aliases,
            )
            stats["property_field_repaired"] += repaired_fields
            ids_raw, node_id_inferred = _infer_ids_from_index(
                path_raw=path_raw,
                canonical_path=path,
                ids_raw=raw_node.get("ids"),
                identity_fields=policy.identity_fields if policy else (),
            )
            if node_id_inferred:
                stats["node_id_inferred"] += 1
            if policy is not None and len(policy.identity_fields) == 1:
                ids_raw, backfilled_count = _backfill_missing_identity_ids(
                    ids_raw=ids_raw,
                    identity_fields=policy.identity_fields,
                    remapped_properties=remapped_props,
                )
                stats["id_backfilled_from_properties"] += backfilled_count
            ids, id_stats = _normalize_ids(
                ids_raw,
                identity_fields=policy.identity_fields if policy else (),
                canonicalize=config.canonicalize_ids,
            )
            stats["id_key_mismatch"] += id_stats["id_key_mismatch"]
            stats["id_missing_required"] += id_stats["id_missing_required"]
            if id_stats["id_missing_required"] > 0:
                missing_by_path = stats.get("id_missing_required_by_path")
                if isinstance(missing_by_path, dict):
                    missing_by_path[path] = int(missing_by_path.get(path, 0)) + int(
                        id_stats["id_missing_required"]
                    )

            parent_raw = raw_node.get("parent")
            expected_parent_path = parent_by_path.get(path, "")
            if path == "":
                parent = None
            else:
                parent = (
                    parent_raw
                    if isinstance(parent_raw, dict)
                    else {"path": expected_parent_path, "ids": {}}
                )
                if not isinstance(parent_raw, dict):
                    stats["parent_ref_missing"] += 1
                    stats["parent_ref_repaired"] += 1
                parent_path_raw = str(parent.get("path") or "")
                parent_path, parent_repaired = _canonicalize_path(
                    parent_path_raw, allowed_paths, field_aliases=field_aliases
                )
                if parent_repaired:
                    stats["path_alias_repaired"] += 1
                parent["path"] = parent_path
                if config.validate_paths and parent_path not in allowed_paths:
                    parent["path"] = expected_parent_path
                    stats["parent_ref_repaired"] += 1
                elif parent_path != expected_parent_path:
                    parent["path"] = expected_parent_path
                    stats["parent_ref_repaired"] += 1
                parent_policy = dedup_policy.get(parent["path"])
                parent_path_for_inference = parent_path_raw
                if not parent_path_for_inference and isinstance(parent_raw, dict):
                    parent_path_for_inference = str(parent_raw.get("path") or "")
                if not parent_path_for_inference:
                    parent_path_for_inference = path_raw
                parent_ids_raw, parent_id_inferred = _infer_ids_from_index(
                    path_raw=parent_path_for_inference,
                    canonical_path=parent["path"],
                    ids_raw=parent.get("ids"),
                    identity_fields=parent_policy.identity_fields if parent_policy else (),
                )
                if not parent_id_inferred and parent_path_for_inference != path_raw and path_raw:
                    parent_ids_raw, parent_id_inferred = _infer_ids_from_index(
                        path_raw=path_raw,
                        canonical_path=parent["path"],
                        ids_raw=parent.get("ids"),
                        identity_fields=parent_policy.identity_fields if parent_policy else (),
                    )
                if parent_id_inferred:
                    stats["parent_id_inferred"] += 1
                parent_ids, parent_stats = _normalize_ids(
                    parent_ids_raw,
                    identity_fields=parent_policy.identity_fields if parent_policy else (),
                    canonicalize=config.canonicalize_ids,
                )
                stats["id_key_mismatch"] += parent_stats["id_key_mismatch"]
                stats["id_missing_required"] += parent_stats["id_missing_required"]
                if parent_stats["id_missing_required"] > 0:
                    missing_by_path = stats.get("id_missing_required_by_path")
                    if isinstance(missing_by_path, dict):
                        ppath = parent.get("path") or expected_parent_path
                        missing_by_path[ppath] = int(missing_by_path.get(ppath, 0)) + int(
                            parent_stats["id_missing_required"]
                        )
                parent["ids"] = parent_ids

            if config.strip_nested_properties:
                clean_props, dropped = _strip_nested_props(raw_node.get("properties"))
                stats["nested_property_dropped"] += dropped
            else:
                _raw_props = raw_node.get("properties")
                clean_props = _raw_props if isinstance(_raw_props, dict) else {}
            if allowed_fields:
                clean_props = {k: v for k, v in clean_props.items() if k in allowed_fields}
            for key, value in remapped_props.items():
                if not _is_non_empty(clean_props.get(key)):
                    clean_props[key] = value

            normalized_node: dict[str, Any] = {
                "path": path,
                "node_type": raw_node.get("node_type"),
                "ids": ids,
                "parent": parent,
                "properties": clean_props,
                "__delta_node_uid": f"b{batch_index}:n{node_idx}",
            }
            if not ids:
                normalized_node["__instance_key"] = normalized_node["__delta_node_uid"]
            if config.attach_provenance:
                normalized_node["provenance"] = provenance
            out_nodes.append(normalized_node)

        for target_path, path_props in salvaged_props_by_path.items():
            target_node = next((n for n in out_nodes if n.get("path") == target_path), None)
            if not isinstance(target_node, dict):
                continue
            target_fields = property_fields_by_path.get(target_path, set())
            target_props = target_node.get("properties")
            if not isinstance(target_props, dict):
                target_props = {}
                target_node["properties"] = target_props
            for key, value in path_props.items():
                if key in target_fields and not _is_non_empty(target_props.get(key)):
                    target_props[key] = value

        for raw_rel in graph.get("relationships", []):
            if not isinstance(raw_rel, dict):
                continue
            source_path_raw = str(raw_rel.get("source_path") or "")
            target_path_raw = str(raw_rel.get("target_path") or "")
            source_path, source_repaired = _canonicalize_path(
                source_path_raw, allowed_paths, field_aliases=field_aliases
            )
            target_path, target_repaired = _canonicalize_path(
                target_path_raw, allowed_paths, field_aliases=field_aliases
            )
            if source_repaired:
                stats["path_alias_repaired"] += 1
            if target_repaired:
                stats["path_alias_repaired"] += 1
            if config.validate_paths and (
                source_path not in allowed_paths or target_path not in allowed_paths
            ):
                stats["unknown_path_dropped"] += 1
                examples = stats.get("unknown_path_examples")
                if isinstance(examples, list) and len(examples) < 20:
                    examples.extend([source_path_raw, target_path_raw])
                continue
            source_policy = dedup_policy.get(source_path)
            target_policy = dedup_policy.get(target_path)
            source_ids_raw, source_id_inferred = _infer_ids_from_index(
                path_raw=source_path_raw,
                canonical_path=source_path,
                ids_raw=raw_rel.get("source_ids"),
                identity_fields=source_policy.identity_fields if source_policy else (),
            )
            if source_id_inferred:
                stats["node_id_inferred"] += 1
            target_ids_raw, target_id_inferred = _infer_ids_from_index(
                path_raw=target_path_raw,
                canonical_path=target_path,
                ids_raw=raw_rel.get("target_ids"),
                identity_fields=target_policy.identity_fields if target_policy else (),
            )
            if target_id_inferred:
                stats["node_id_inferred"] += 1
            source_ids, source_stats = _normalize_ids(
                source_ids_raw,
                identity_fields=source_policy.identity_fields if source_policy else (),
                canonicalize=config.canonicalize_ids,
            )
            target_ids, target_stats = _normalize_ids(
                target_ids_raw,
                identity_fields=target_policy.identity_fields if target_policy else (),
                canonicalize=config.canonicalize_ids,
            )
            stats["id_key_mismatch"] += (
                source_stats["id_key_mismatch"] + target_stats["id_key_mismatch"]
            )
            stats["id_missing_required"] += (
                source_stats["id_missing_required"] + target_stats["id_missing_required"]
            )

            if config.strip_nested_properties:
                clean_rel_props, dropped_rel = _strip_nested_props(raw_rel.get("properties"))
                stats["nested_property_dropped"] += dropped_rel
            else:
                _raw_rel_props = raw_rel.get("properties")
                clean_rel_props = _raw_rel_props if isinstance(_raw_rel_props, dict) else {}

            normalized_rel: dict[str, Any] = {
                "edge_label": str(raw_rel.get("edge_label") or ""),
                "source_path": source_path,
                "source_ids": source_ids,
                "target_path": target_path,
                "target_ids": target_ids,
                "properties": clean_rel_props,
            }
            if config.attach_provenance:
                normalized_rel["provenance"] = provenance
            out_relationships.append(normalized_rel)

        normalized_results.append({"nodes": out_nodes, "relationships": out_relationships})

    return normalized_results, stats
