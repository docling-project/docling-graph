"""Helper functions for the delta extraction contract."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .catalog import DeltaNodeCatalog

logger = logging.getLogger(__name__)

# LLMs sometimes echo the batch context into node properties; strip those values before projection.
_BATCH_ECHO_PATTERN = re.compile(
    r"^(?:Delta extraction batch\s+\d+/\d+\.?|\[Batch\s+\d+/\d+[^\]]*\])$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DedupPolicy:
    """Per-path dedup policy derived from template catalog."""

    path: str
    node_type: str
    identity_fields: tuple[str, ...]
    fallback_text_fields: tuple[str, ...]
    allowed_match_fields: tuple[str, ...]
    is_entity: bool


DEFAULT_FALLBACK_TEXT_FIELDS: tuple[str, ...] = (
    "name",
    "title",
    "id",
    "code",
    "nom",
    "resume",
    "line_number",
    "item_code",
    "document_number",
)

LOCAL_ID_FIELD_HINTS: tuple[str, ...] = ("line_number", "index", "position", "item_number")
LONGER_STRING_FIELDS: tuple[str, ...] = ("name", "title", "nom")


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _preferred_property_value(
    *,
    prop_key: str,
    existing_value: Any,
    incoming_value: Any,
) -> tuple[Any, bool]:
    """Choose deterministic canonical value and report conflict presence."""
    if _is_empty(existing_value):
        return incoming_value, False
    if _is_empty(incoming_value):
        return existing_value, False

    # When both values are present, prefer richer strings.
    if isinstance(existing_value, str) and isinstance(incoming_value, str):
        existing_txt = existing_value.strip()
        incoming_txt = incoming_value.strip()
        if len(incoming_txt) > len(existing_txt):
            return incoming_value, incoming_txt != existing_txt
        return existing_value, incoming_txt != existing_txt

    # Keep deterministic stability for non-string scalar/list values.
    if existing_value != incoming_value:
        return existing_value, True
    return existing_value, False


def build_dedup_policy(catalog: DeltaNodeCatalog) -> dict[str, DedupPolicy]:
    """Build per-path dedup policy from catalog id fields."""

    policy: dict[str, DedupPolicy] = {}
    for spec in catalog.nodes:
        identity_fields = tuple(spec.id_fields or ())
        scoped_defaults = [
            f for f in DEFAULT_FALLBACK_TEXT_FIELDS if f in set(spec.property_fields or [])
        ]
        fallback_fields = tuple(dict.fromkeys([*identity_fields, *scoped_defaults]).keys())
        policy[spec.path] = DedupPolicy(
            path=spec.path,
            node_type=spec.node_type,
            identity_fields=identity_fields,
            fallback_text_fields=fallback_fields,
            allowed_match_fields=fallback_fields,
            is_entity=spec.kind == "entity",
        )
    return policy

def chunk_batches_by_token_limit(
    chunks: Sequence[str],
    token_counts: Sequence[int],
    *,
    max_batch_tokens: int,
) -> list[list[tuple[int, str, int]]]:
    """Pack sequential chunks into token-bounded batches."""

    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be > 0")

    batches: list[list[tuple[int, str, int]]] = []
    current: list[tuple[int, str, int]] = []
    current_tokens = 0

    for idx, chunk in enumerate(chunks):
        tcount = token_counts[idx] if idx < len(token_counts) else max(1, len(chunk.split()))
        if current and current_tokens + tcount > max_batch_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((idx, chunk, tcount))
        current_tokens += tcount

    if current:
        batches.append(current)

    return batches


def _normalize_primitive(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


def _normalize_list(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if isinstance(value, list):
            out.extend(_normalize_list(value))
        elif isinstance(value, dict):
            # Keep graph props flat and queryable: skip nested objects.
            continue
        else:
            out.append(_normalize_primitive(value))
    return out


def flatten_node_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Ensure node properties remain Neo4j-safe flat values."""

    flat: dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, dict):
            # Keep graph props flat and queryable: skip nested objects.
            continue
        elif isinstance(value, list):
            flat[key] = _normalize_list(value)
        else:
            flat[key] = _normalize_primitive(value)
    return flat


def node_identity_key(
    node: dict[str, Any],
    dedup_policy: dict[str, DedupPolicy] | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]] | tuple[str, str]:
    """Compute canonical key for dedup across batches."""

    path = str(node.get("path") or "")
    policy = dedup_policy.get(path) if isinstance(dedup_policy, dict) else None
    ids = node.get("ids") or {}
    parent = node.get("parent") if isinstance(node.get("parent"), dict) else None
    parent_ctx = ""
    if isinstance(parent, dict):
        parent_ids = parent.get("ids") if isinstance(parent.get("ids"), dict) else {}
        if parent_ids:
            parent_ctx = f"{parent.get('path', '')}|{tuple(sorted((str(k), str(v)) for k, v in parent_ids.items()))}"
        parent_inst = parent.get("__instance_key")
        if not parent_ctx and isinstance(parent_inst, str) and parent_inst:
            parent_ctx = f"{parent.get('path', '')}|inst:{parent_inst}"
    if isinstance(ids, dict) and ids:
        if policy is not None and policy.identity_fields:
            ordered: list[tuple[str, str]] = []
            for field_name in policy.identity_fields:
                val = ids.get(field_name)
                if val is not None:
                    ordered.append((str(field_name), str(val)))
            if ordered:
                if parent_ctx and any(field_name in LOCAL_ID_FIELD_HINTS for field_name, _ in ordered):
                    ordered.append(("__parent_ctx__", parent_ctx))
                return (path, tuple(ordered))
        norm_ids = tuple(sorted((str(k), str(v)) for k, v in ids.items()))
        return (path, norm_ids)

    props = node.get("properties") or {}
    if isinstance(props, dict):
        candidates = (
            policy.fallback_text_fields if policy is not None else DEFAULT_FALLBACK_TEXT_FIELDS
        )
        for candidate in candidates:
            val = props.get(candidate)
            if val is not None:
                fallback_key = f"{candidate}:{str(val).strip().lower()}"
                if parent_ctx:
                    fallback_key = f"{fallback_key}|{parent_ctx}"
                return (path, fallback_key)
    instance_key = node.get("__instance_key") or node.get("__delta_node_uid")
    if isinstance(instance_key, str) and instance_key:
        return (path, f"__instance__:{instance_key}")
    # Never collapse unidentified nodes into a shared key.
    return (path, f"__instance__:{id(node)}")


def _relationship_endpoint_key(
    *,
    path: str,
    ids: dict[str, Any],
    dedup_policy: dict[str, DedupPolicy] | None,
) -> tuple[str, tuple[tuple[str, str], ...]] | tuple[str, str]:
    """Build endpoint key using the same normalization as node identity."""
    return node_identity_key(
        {"path": path, "ids": ids, "properties": {}},
        dedup_policy=dedup_policy,
    )


def merge_delta_graphs(
    graph_dicts: Iterable[dict[str, Any]],
    dedup_policy: dict[str, DedupPolicy] | None = None,
) -> dict[str, Any]:
    """Merge graph batches with node and relationship deduplication."""

    node_by_key: dict[Any, dict[str, Any]] = {}
    relationships: dict[tuple[str, Any, Any, str], dict[str, Any]] = {}
    merge_stats: dict[str, int] = {
        "node_inputs": 0,
        "node_dedup_merges": 0,
        "property_updates": 0,
        "property_conflicts": 0,
        "relationship_inputs": 0,
        "relationship_dedup_replaced": 0,
    }

    for graph in graph_dicts:
        for raw_node in graph.get("nodes", []):
            if not isinstance(raw_node, dict):
                continue
            merge_stats["node_inputs"] += 1
            node = dict(raw_node)
            props = node.get("properties")
            node["properties"] = flatten_node_properties(props if isinstance(props, dict) else {})
            key = node_identity_key(node, dedup_policy=dedup_policy)
            existing = node_by_key.get(key)
            if existing is None:
                provenance = {}
                for prop_key, prop_val in (node.get("properties") or {}).items():
                    if _is_empty(prop_val):
                        continue
                    provenance[prop_key] = [node.get("provenance", node.get("__delta_node_uid", "unknown"))]
                if provenance:
                    node["__property_provenance"] = provenance
                node_by_key[key] = node
            else:
                merge_stats["node_dedup_merges"] += 1
                merged_props = dict(existing.get("properties") or {})
                for prop_key, prop_val in (node.get("properties") or {}).items():
                    previous = merged_props.get(prop_key)
                    if prop_key not in merged_props:
                        merged_props[prop_key] = prop_val
                        if not _is_empty(prop_val):
                            merge_stats["property_updates"] += 1
                        continue
                    chosen, had_conflict = _preferred_property_value(
                        prop_key=prop_key,
                        existing_value=previous,
                        incoming_value=prop_val,
                    )
                    if had_conflict:
                        merge_stats["property_conflicts"] += 1
                    if chosen != previous:
                        merge_stats["property_updates"] += 1
                    merged_props[prop_key] = chosen
                existing["properties"] = merged_props
                provenance = existing.setdefault("__property_provenance", {})
                if isinstance(provenance, dict):
                    for prop_key, prop_val in (node.get("properties") or {}).items():
                        if _is_empty(prop_val):
                            continue
                        stamp = node.get("provenance", node.get("__delta_node_uid", "unknown"))
                        raw_stamps = provenance.get(prop_key)
                        prop_stamps = (
                            raw_stamps
                            if isinstance(raw_stamps, list)
                            else ([raw_stamps] if raw_stamps else [])
                        )
                        provenance[prop_key] = prop_stamps
                        if stamp not in prop_stamps:
                            prop_stamps.append(stamp)
                if not existing.get("parent") and node.get("parent"):
                    existing["parent"] = node["parent"]

        for raw_rel in graph.get("relationships", []):
            if not isinstance(raw_rel, dict):
                continue
            merge_stats["relationship_inputs"] += 1
            rel = dict(raw_rel)
            edge_label = str(rel.get("edge_label") or "")
            _src = rel.get("source_ids")
            source_ids_raw: dict[str, Any] = _src if isinstance(_src, dict) else {}
            _tgt = rel.get("target_ids")
            target_ids_raw: dict[str, Any] = _tgt if isinstance(_tgt, dict) else {}
            source_key = _relationship_endpoint_key(
                path=str(rel.get("source_path") or ""),
                ids=source_ids_raw,
                dedup_policy=dedup_policy,
            )
            target_key = _relationship_endpoint_key(
                path=str(rel.get("target_path") or ""),
                ids=target_ids_raw,
                dedup_policy=dedup_policy,
            )
            props_key = json.dumps(flatten_node_properties(rel.get("properties") or {}), sort_keys=True)
            dedup_key = (edge_label, source_key, target_key, props_key)
            if dedup_key in relationships:
                merge_stats["relationship_dedup_replaced"] += 1
            relationships[dedup_key] = rel

    return {
        "nodes": list(node_by_key.values()),
        "relationships": list(relationships.values()),
        "__merge_stats": merge_stats,
    }


def sanitize_batch_echo_from_graph(graph: dict[str, Any]) -> None:
    """
    In-place: replace any node property or id value that is a batch-echo string
    (e.g. 'Delta extraction batch 25/49') with empty string so they are not
    projected into the template.
    """
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for attr in ("properties", "ids"):
            container = node.get(attr)
            if not isinstance(container, dict):
                continue
            for key, value in list(container.items()):
                if isinstance(value, str) and _BATCH_ECHO_PATTERN.match(value.strip()):
                    container[key] = ""


def per_path_counts(nodes: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count nodes by catalog path."""

    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        path = str(node.get("path") or "")
        counts[path] += 1
    return dict(counts)
