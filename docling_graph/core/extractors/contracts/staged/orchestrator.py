"""Catalog orchestrator: 3-pass staged extraction (catalog → ID → fill → edges).

With --extraction-contract staged and --debug, artifacts are written under
debug/: node_catalog.json, id_pass.json, id_pass_schema.json, id_pass_prompt.json,
fill_pass.json, edges_pass.json, merged_output.json, staged_trace.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel
from rich import print as rich_print

from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

from .....llm_clients.schema_utils import build_compact_semantic_guide
from . import catalog as catalog_mod
from .catalog import (
    EdgeSpec,
    NodeCatalog,
    NodeSpec,
    _is_list_under_list,
    flat_nodes_to_path_lists,
    get_id_pass_shards,
    get_id_pass_shards_v2,
    get_model_for_path,
    merge_and_dedupe_flat_nodes,
)

logger = logging.getLogger(__name__)

# Max chars to keep in prompt artifacts (avoid huge debug files)
_DEBUG_PROMPT_USER_TRUNCATE = 8000


def _build_staged_trace(
    *,
    template_name: str,
    config: CatalogOrchestratorConfig,
    catalog: NodeCatalog,
    id_elapsed: float,
    fill_elapsed: float,
    total_elapsed: float,
    per_path_counts: dict[str, int],
    id_validation_ok: bool,
    id_validation_errors: list[str],
    fill_batches: list[dict[str, Any]],
    merged_keys: list[str],
    edges_count: int,
    merge_stats: dict[str, int],
    quality_gate: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build the staged-trace dict (config, catalog summary, timings, pass details)."""
    return {
        "template": template_name,
        "config": {
            "max_nodes_per_call": config.max_nodes_per_call,
            "parallel_workers": config.parallel_workers,
            "max_pass_retries": config.max_pass_retries,
            "id_shard_size": config.id_shard_size,
            "id_identity_only": config.id_identity_only,
            "id_compact_prompt": config.id_compact_prompt,
        },
        "catalog": catalog.to_dict(),
        "timings_seconds": {
            "id_pass": round(id_elapsed, 3),
            "fill_pass": round(fill_elapsed, 3),
            "total": round(total_elapsed, 3),
        },
        "per_path_counts": per_path_counts,
        "total_instances": sum(per_path_counts.values()),
        "id_pass_validation": {
            "ok": id_validation_ok,
            "errors": id_validation_errors[:20],
        },
        "fill_batches": fill_batches,
        "merge_stats": merge_stats,
        "merged_output_keys": merged_keys,
        "edges_resolved_count": edges_count,
        "quality_gate": quality_gate or {"ok": True, "reasons": []},
        "fallback_reason": fallback_reason,
    }


def _write_staged_trace(debug_dir: str, trace: dict[str, Any]) -> None:
    """Write staged_trace.json with the given trace dict."""
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, "staged_trace.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)


@dataclass
class CatalogOrchestratorConfig:
    max_nodes_per_call: int = 5
    parallel_workers: int = 1
    max_pass_retries: int = 1
    id_shard_size: int = 0  # paths per ID-pass call; 0 = auto planning
    id_identity_only: bool = True
    id_compact_prompt: bool = True
    id_auto_shard_threshold: int = 12
    id_shard_min_size: int = 2
    quality_require_root: bool = True
    quality_min_instances: int = 1
    quality_max_parent_lookup_miss: int = 0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> CatalogOrchestratorConfig:
        if not values:
            return cls()
        return cls(
            max_nodes_per_call=int(values.get("catalog_max_nodes_per_call", 5)),
            parallel_workers=int(values.get("parallel_workers", 1)),
            max_pass_retries=int(values.get("max_pass_retries", 1)),
            id_shard_size=int(values.get("id_shard_size", 0)),
            id_identity_only=bool(values.get("id_identity_only", True)),
            id_compact_prompt=bool(values.get("id_compact_prompt", True)),
            id_auto_shard_threshold=int(values.get("id_auto_shard_threshold", 12)),
            id_shard_min_size=int(values.get("id_shard_min_size", 2)),
            quality_require_root=bool(values.get("quality_require_root", True)),
            quality_min_instances=int(values.get("quality_min_instances", 1)),
            quality_max_parent_lookup_miss=int(values.get("quality_max_parent_lookup_miss", 0)),
        )


def get_fill_batch_prompt(
    markdown_content: str,
    path: str,
    spec: NodeSpec,
    instances: list[dict[str, Any]],
    schema_json: str,
    structured_output: bool = True,
    schema_dict: dict[str, Any] | None = None,
) -> dict[str, str]:
    """instances: list of ID descriptors (path, ids, parent). Preview uses ids only."""
    n = len(instances)
    preview_items = [{**(inst.get("ids") or {})} for inst in instances[:3]]
    instances_preview = json.dumps(preview_items, indent=2, default=str)
    if n > 3:
        instances_preview += f"\n... and {n - 3} more"
    system_prompt = (
        "You are a precise extraction assistant. Fill each of the given node instances "
        f"with all schema fields from the document. Path: {path} ({spec.node_type}). "
        "Return a JSON array with one object per instance in the same order."
    )
    user_prompt = "Fill these node instances from the document.\n\n=== DOCUMENT ===\n"
    user_prompt += f"{markdown_content}\n=== END DOCUMENT ===\n\n"
    user_prompt += "=== INSTANCES ===\n" + instances_preview + "\n=== END ===\n\n"
    if structured_output:
        user_prompt += "=== SEMANTIC FIELD GUIDANCE ===\n"
        user_prompt += build_compact_semantic_guide(schema_dict or {}) + "\n=== END ===\n\n"
    else:
        user_prompt += "=== SCHEMA ===\n" + schema_json + "\n=== END ===\n\n"
    user_prompt += f"Return a JSON array of exactly {n} objects."
    return {"system": system_prompt, "user": user_prompt}


def _id_tuple(
    spec: NodeSpec, ids: dict[str, Any], instance_key: str | None = None
) -> tuple[Any, ...]:
    """Stable tuple of id values (raw) for display/ordering. Prefer _canonical_lookup_key for merge lookup."""
    if not spec.id_fields:
        return (instance_key or "",)
    return tuple(ids.get(f) for f in spec.id_fields)


def _canonical_lookup_key(
    path: str,
    spec: NodeSpec,
    ids: dict[str, Any],
    instance_key: str | None = None,
) -> tuple[Any, ...]:
    """Same key as merge_and_dedupe_flat_nodes so parent lookup works after canonical dedup."""
    if not spec.id_fields:
        return (path, (instance_key or "",))
    normalized = tuple(
        sorted((f, canonicalize_identity_for_dedup(f, ids.get(f))) for f in spec.id_fields)
    )
    return (path, normalized)


def merge_filled_into_root(
    path_filled: dict[str, list[Any]],
    path_descriptors: dict[str, list[dict[str, Any]]],
    catalog: NodeCatalog,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Build root dict by attaching filled nodes to their parent using parent path + ids.
    Lookup: (path, id_tuple) -> filled object. Root and root-children attach to root; others to parent object.
    """
    root: dict[str, Any] = {}
    merge_stats = {
        "descriptor_length_mismatch": 0,
        "non_dict_filled_objects": 0,
        "missing_parent_descriptor": 0,
        "parent_lookup_miss": 0,
        "attached_list_items": 0,
        "attached_scalar_items": 0,
    }
    spec_by_path = {spec.path: spec for spec in catalog.nodes}
    lookup: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}

    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        if len(filled_list) != len(descriptors):
            merge_stats["descriptor_length_mismatch"] += 1
            logger.warning(
                "[StagedExtraction] merge input mismatch for path '%s': %s filled vs %s descriptors",
                path,
                len(filled_list),
                len(descriptors),
            )
        for i, obj in enumerate(filled_list):
            if not isinstance(obj, dict):
                merge_stats["non_dict_filled_objects"] += 1
                continue
            desc = descriptors[i] if i < len(descriptors) else {}
            ids = desc.get("ids") or {}
            instance_key = desc.get("__instance_key") if isinstance(desc, dict) else None
            key = _canonical_lookup_key(path, spec, ids, instance_key=instance_key)
            lookup[key] = obj

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
                merge_stats["non_dict_filled_objects"] += 1
                continue
            desc = descriptors[i] if i < len(descriptors) else {}
            parent = desc.get("parent")
            if not parent or not isinstance(parent, dict):
                merge_stats["missing_parent_descriptor"] += 1
                continue
            parent_ids = parent.get("ids") or {}
            parent_instance_key = parent.get("__instance_key") if isinstance(parent, dict) else None
            parent_key = _canonical_lookup_key(
                parent_path, parent_spec, parent_ids, instance_key=parent_instance_key
            )
            parent_obj = lookup.get(parent_key)
            if parent_obj is None:
                merge_stats["parent_lookup_miss"] += 1
                continue
            if is_list:
                parent_obj.setdefault(field_name, []).append(obj)
                merge_stats["attached_list_items"] += 1
            else:
                parent_obj[field_name] = obj
                merge_stats["attached_scalar_items"] += 1
    if stats is not None:
        stats.update(merge_stats)
    return root


def _maybe_resolve_conflicts(
    merged: dict[str, Any],
    catalog: NodeCatalog,
    llm_call_fn: Callable[..., dict | list | None],
    context: str,
) -> dict[str, Any]:
    """
    If conflicts are detected in the merged root, send them with context to the LLM for resolution
    and apply the result. Otherwise return merged unchanged.
    """
    conflicts = _detect_merge_conflicts(merged)
    if not conflicts:
        return merged
    # TODO: build prompt with conflicts + context, call llm_call_fn, parse resolution, apply to merged
    return merged


def _detect_merge_conflicts(merged: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect conflicts in merged root (e.g. duplicate keys, inconsistent nesting). Returns list of conflict descriptors."""
    # Placeholder: no conflicts detected. Can be extended with schema-defined or structural rules.
    return []


def assemble_edges_from_merged(
    merged: dict[str, Any],
    catalog: NodeCatalog,
) -> list[dict[str, Any]]:
    """
    Assemble edges implied by nesting from merged root.
    Returns list of {source_id, target_id, edge_label} for debug/trace.
    Graph conversion uses the merged structure; this is for diagnostics.
    """
    resolved: list[dict[str, Any]] = []
    for edge in catalog.edges:
        resolved.append(
            {
                "edge_label": edge.edge_label,
                "source_path": edge.source_path,
                "target_path": edge.target_path,
                "implied_by_nesting": True,
            }
        )
    return resolved


def _build_projected_fill_schema(
    template: type[BaseModel], spec: NodeSpec, catalog: NodeCatalog
) -> str:
    """Return a schema for fill restricted to this path (blocks nested over-expansion)."""
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
    schema["properties"] = keep_props
    if isinstance(schema.get("required"), list):
        schema["required"] = [k for k in schema["required"] if k in keep_props]
    return json.dumps(schema, indent=2)


def _sanitize_filled_objects(
    filled: list[Any],
    allowed_keys: set[str],
    descriptors: list[dict[str, Any]],
    id_fields: list[str],
) -> list[dict[str, Any]]:
    """Drop unexpected nested keys and preserve descriptor IDs when missing."""
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(filled):
        src = obj if isinstance(obj, dict) else {}
        clean = {k: v for k, v in src.items() if k in allowed_keys}
        ids = (
            descriptors[i].get("ids")
            if i < len(descriptors) and isinstance(descriptors[i], dict)
            else {}
        )
        if isinstance(ids, dict):
            for f in id_fields:
                if f in ids and f not in clean:
                    clean[f] = ids[f]
        out.append(clean)
    return out


def _evaluate_quality_gate(
    *,
    config: CatalogOrchestratorConfig,
    per_path_counts: dict[str, int],
    merge_stats: dict[str, int],
    merged: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    total_instances = sum(per_path_counts.values())
    if config.quality_require_root and per_path_counts.get("", 0) <= 0:
        reasons.append("missing_root_instance")
    if total_instances < max(0, int(config.quality_min_instances)):
        reasons.append("insufficient_id_instances")
    if merge_stats.get("parent_lookup_miss", 0) > max(
        0, int(config.quality_max_parent_lookup_miss)
    ):
        reasons.append("excess_parent_lookup_miss")
    if not isinstance(merged, dict) or not merged:
        reasons.append("empty_merged_output")
    return (len(reasons) == 0), reasons


class CatalogOrchestrator:
    def __init__(
        self,
        llm_call_fn: Callable[..., dict | list | None],
        schema_json: str,
        template: type[BaseModel],
        config: CatalogOrchestratorConfig,
        debug_dir: str | None = None,
        structured_output: bool = True,
        on_trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm_call_fn
        self._schema_json = schema_json
        self._template = template
        self._config = config
        self._debug_dir = debug_dir or ""
        self._structured_output = structured_output
        self._on_trace = on_trace
        self._catalog = catalog_mod.build_node_catalog(template)

    def _run_id_pass(
        self, markdown: str, context: str
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, int],
        float,
        bool,
        list[str],
    ]:
        """Run identity discovery pass (sharded, with retries and optional split)."""
        id_start = time.perf_counter()
        shard_size = max(0, int(self._config.id_shard_size))
        if shard_size <= 0 and len(self._catalog.nodes) > int(self._config.id_auto_shard_threshold):
            shard_size = max(int(self._config.id_shard_min_size), 4)
        shards = get_id_pass_shards_v2(
            self._catalog,
            shard_size,
            identity_only=self._config.id_identity_only,
            root_first=True,
        )
        max_retries = max(0, int(self._config.max_pass_retries))
        all_flat: list[list[dict[str, Any]]] = []
        id_validation_ok = True
        id_validation_errors: list[str] = []
        workers = max(1, int(self._config.parallel_workers))
        shard_tasks = list(enumerate(shards))
        # When using multiple shards, default to compact prompt to reduce output size and token use
        use_compact = self._config.id_compact_prompt or (len(shards) > 1)

        def run_one_id_shard(
            item: tuple[int, tuple[list[str], list[str]]],
        ) -> tuple[int, bool, list[str], list[list[dict[str, Any]]]]:
            shard_idx, (primary_paths, allowed_paths) = item
            ok, errs, flat_lists = self._run_id_pass_shard(
                markdown,
                context,
                shard_idx,
                primary_paths,
                allowed_paths,
                max_retries,
                use_compact=use_compact,
            )
            return (shard_idx, ok, errs, flat_lists)

        if workers <= 1:
            for shard_idx, (primary_paths, allowed_paths) in shard_tasks:
                ok, errs, flat_lists = self._run_id_pass_shard(
                    markdown,
                    context,
                    shard_idx,
                    primary_paths,
                    allowed_paths,
                    max_retries,
                    use_compact=use_compact,
                )
                if not ok:
                    id_validation_ok = False
                    id_validation_errors.extend(errs[:15])
                for flat in flat_lists:
                    all_flat.append(flat)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                completed = list(ex.map(run_one_id_shard, shard_tasks))
            # Preserve shard order for merge (remap_no_id_keys uses shard order)
            completed.sort(key=lambda x: x[0])
            for _shard_idx, ok, errs, flat_lists in completed:
                if not ok:
                    id_validation_ok = False
                    id_validation_errors.extend(errs[:15])
                for flat in flat_lists:
                    all_flat.append(flat)
        flat_nodes, per_path_counts = merge_and_dedupe_flat_nodes(all_flat, self._catalog)
        id_elapsed = time.perf_counter() - id_start
        total_instances = sum(per_path_counts.values())
        rich_print(
            f"[green][IdentityDiscovery][/green] ID pass done in [cyan]{id_elapsed:.1f}s[/cyan]: "
            f"[cyan]{total_instances}[/cyan] instances across paths "
        )
        if self._debug_dir:
            catalog_mod.write_id_pass_artifact(
                {"nodes": flat_nodes}, per_path_counts, self._debug_dir
            )
        return (flat_nodes, per_path_counts, id_elapsed, id_validation_ok, id_validation_errors)

    def _run_id_pass_shard(
        self,
        markdown: str,
        context: str,
        shard_idx: int,
        primary_paths: list[str],
        allowed_paths: list[str],
        max_retries: int,
        *,
        use_compact: bool | None = None,
    ) -> tuple[bool, list[str], list[list[dict[str, Any]]]]:
        """Run one ID pass shard (with retries and optional split on failure). Returns (ok, errs, list_of_flat_lists)."""
        compact = use_compact if use_compact is not None else self._config.id_compact_prompt
        id_prompt = catalog_mod.get_discovery_prompt(
            markdown,
            self._catalog,
            primary_paths=primary_paths,
            allowed_paths=allowed_paths,
            compact=compact,
            include_schema_in_user=not compact,
            structured_output=self._structured_output,
        )
        id_schema = catalog_mod.build_discovery_schema(self._catalog, allowed_paths)
        if self._debug_dir and shard_idx == 0:
            os.makedirs(self._debug_dir, exist_ok=True)
            with open(
                os.path.join(self._debug_dir, "id_pass_schema.json"), "w", encoding="utf-8"
            ) as f:
                f.write(id_schema)
            prompt_artifact = {
                "system": id_prompt["system"],
                "user": id_prompt["user"][:_DEBUG_PROMPT_USER_TRUNCATE]
                + (
                    "\n... [truncated]"
                    if len(id_prompt["user"]) > _DEBUG_PROMPT_USER_TRUNCATE
                    else ""
                ),
            }
            with open(
                os.path.join(self._debug_dir, "id_pass_prompt.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(prompt_artifact, f, indent=2)
        shard_flat: list[dict[str, Any]] = []
        ok = False
        errs: list[str] = []
        for attempt in range(max_retries + 1):
            id_result = self._llm(
                id_prompt,
                id_schema,
                f"{context} catalog_id_pass_shard_{shard_idx}",
                response_top_level="object",
                response_schema_name="staged_id_pass",
            )
            if isinstance(id_result, list):
                id_result = {"nodes": id_result}
            if not isinstance(id_result, dict):
                logger.warning(
                    "%s ID pass shard %s attempt %s returned no dict",
                    "[IdentityDiscovery]",
                    shard_idx + 1,
                    attempt + 1,
                )
                if attempt < max_retries:
                    err_feedback = (
                        "The previous response was not valid JSON. Please return a single JSON object with "
                        '"nodes": [{"path": "...", "ids": {...}, "parent": null|{...}}].'
                    )
                    id_prompt["user"] = id_prompt["user"] + "\n\n=== FIX ===\n" + err_feedback
                continue
            ok, errs, shard_flat, _ = catalog_mod.validate_id_pass_skeleton_response(
                id_result, self._catalog, allowed_paths_override=set(allowed_paths)
            )
            if ok:
                return (True, errs, [shard_flat])
            logger.warning(
                "%s ID pass shard %s validation (attempt %s): %s",
                "[IdentityDiscovery]",
                shard_idx + 1,
                attempt + 1,
                errs[:5],
            )
            if attempt < max_retries:
                err_feedback = "Fix these validation errors:\n" + "\n".join(errs[:15])
                err_feedback += (
                    "\n\nRules reminder:\n"
                    '- Root path "" must have parent null.\n'
                    "- Every non-root node must have parent object {path, ids}.\n"
                    "- For paths with no id_fields, use ids: {} exactly.\n"
                    "- Return strict JSON only (no trailing commas, no comments)."
                )
                id_prompt = catalog_mod.get_discovery_prompt(
                    markdown,
                    self._catalog,
                    primary_paths=primary_paths,
                    allowed_paths=allowed_paths,
                    compact=compact,
                    include_schema_in_user=not compact,
                    structured_output=self._structured_output,
                )
                id_prompt["user"] = id_prompt["user"] + "\n\n=== FIX ===\n" + err_feedback
        if not ok and len(primary_paths) > max(1, int(self._config.id_shard_min_size)):
            split_size = max(1, len(primary_paths) // 2)
            sub_shards = get_id_pass_shards_v2(
                self._catalog,
                split_size,
                identity_only=False,
                root_first=False,
            )
            split_flats: list[list[dict[str, Any]]] = []
            for sub_primary, sub_allowed in sub_shards:
                if not set(sub_primary).issubset(set(primary_paths)):
                    continue
                sub_prompt = catalog_mod.get_discovery_prompt(
                    markdown,
                    self._catalog,
                    primary_paths=sub_primary,
                    allowed_paths=sub_allowed,
                    compact=compact,
                    include_schema_in_user=not compact,
                    structured_output=self._structured_output,
                )
                sub_schema = catalog_mod.build_discovery_schema(self._catalog, sub_allowed)
                sub_result = self._llm(
                    sub_prompt,
                    sub_schema,
                    f"{context} catalog_id_pass_shard_{shard_idx}_split",
                    response_top_level="object",
                    response_schema_name="staged_id_pass",
                )
                if isinstance(sub_result, list):
                    sub_result = {"nodes": sub_result}
                if isinstance(sub_result, dict):
                    sub_ok, _sub_errs, sub_flat, _ = catalog_mod.validate_id_pass_skeleton_response(
                        sub_result,
                        self._catalog,
                        allowed_paths_override=set(sub_allowed),
                    )
                    if sub_ok:
                        split_flats.append(sub_flat)
            if split_flats:
                return (False, errs, split_flats)
            return (False, errs, [shard_flat])
        if not ok:
            return (False, errs, [shard_flat])
        return (ok, errs, [shard_flat])

    def _run_fill_pass(
        self,
        markdown: str,
        context: str,
        path_to_descriptors: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], float]:
        """Run fill pass (batch jobs sequential or parallel). Returns (path_filled, fill_batches_trace, fill_elapsed)."""
        path_filled: dict[str, list[dict[str, Any]]] = {}
        fill_batches_trace: list[dict[str, Any]] = []
        fill_debug_batches: list[dict[str, Any]] = []
        workers = max(1, int(self._config.parallel_workers))
        max_nodes_per_call = self._config.max_nodes_per_call

        def run_one_batch(
            spec: NodeSpec,
            batch_descriptors: list[dict[str, Any]],
            batch_idx: int,
            sub_schema: str,
            call_index: int,
        ) -> tuple[str, int, list[dict[str, Any]], dict[str, Any]]:
            schema_dict: dict[str, Any] | None = None
            item_schema: dict[str, Any] = {}
            if self._structured_output:
                try:
                    parsed = json.loads(sub_schema)
                    if isinstance(parsed, dict):
                        schema_dict = parsed
                        item_schema = parsed
                except json.JSONDecodeError:
                    pass
            # Request object root (not array) so providers that only support object-root json_schema succeed
            wrapped_schema_dict: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": item_schema},
                },
                "required": ["items"],
            }
            fill_pass_schema_json = json.dumps(wrapped_schema_dict)
            prompt = get_fill_batch_prompt(
                markdown,
                spec.path,
                spec,
                batch_descriptors,
                sub_schema,
                structured_output=self._structured_output,
                schema_dict=schema_dict,
            )
            out = self._llm(
                prompt,
                fill_pass_schema_json,
                f"{context} fill_call_{call_index}",
                response_top_level="object",
                response_schema_name="staged_fill_pass",
            )
            if isinstance(out, list):
                result_raw = out
            elif isinstance(out, dict) and "items" in out:
                result_raw = out["items"]
            else:
                result_raw = [{}] * len(batch_descriptors)
            model = get_model_for_path(self._template, spec.path)
            allowed_keys = set(model.model_fields.keys()) if model is not None else set()
            result = _sanitize_filled_objects(
                result_raw,
                allowed_keys=allowed_keys,
                descriptors=batch_descriptors,
                id_fields=spec.id_fields,
            )
            debug_entry = {
                "call_index": call_index,
                "path": spec.path,
                "batch_idx": batch_idx,
                "instance_count": len(batch_descriptors),
                "filled": result,
            }
            return (spec.path, batch_idx, result, debug_entry)

        def _path_depth(p: str) -> int:
            return (p.count(".") + 1) if p else 0

        # For list-under-list paths, fill once per unique (path, ids) then expand to original descriptor order
        expand_after_fill: list[
            tuple[str, list[dict[str, Any]], list[dict[str, Any]], NodeSpec]
        ] = []

        specs_by_depth = sorted(
            self._catalog.nodes,
            key=lambda s: _path_depth(s.path),
            reverse=True,
        )
        fill_jobs: list[tuple[NodeSpec, list[dict[str, Any]], int, str]] = []
        for spec in specs_by_depth:
            descriptors = path_to_descriptors.get(spec.path, [])
            if not descriptors:
                continue
            if _is_list_under_list(spec.path, spec):
                seen_keys: set[tuple[Any, ...]] = set()
                unique_descriptors: list[dict[str, Any]] = []
                for d in descriptors:
                    k = _canonical_lookup_key(
                        spec.path,
                        spec,
                        d.get("ids") or {},
                        d.get("__instance_key"),
                    )
                    if k not in seen_keys:
                        seen_keys.add(k)
                        unique_descriptors.append(d)
                expand_after_fill.append((spec.path, descriptors, unique_descriptors, spec))
                descriptors = unique_descriptors
            sub_schema = _build_projected_fill_schema(self._template, spec, self._catalog)
            batches = [
                descriptors[i : i + max_nodes_per_call]
                for i in range(0, len(descriptors), max_nodes_per_call)
            ]
            for bi, batch in enumerate(batches):
                fill_jobs.append((spec, batch, bi, sub_schema))

        num_calls = len(fill_jobs)
        fill_start = time.perf_counter()
        rich_print(
            f"[blue][NodesProvisioning][/blue] Fill pass [cyan]{num_calls}[/cyan] LLM call(s) "
            f"(max [cyan]{max_nodes_per_call}[/cyan] nodes/call, [cyan]{workers}[/cyan] worker(s))..."
        )
        if workers <= 1:
            for call_index, (spec, batch, batch_idx, sub_schema) in enumerate(fill_jobs):
                _, _, result, debug_entry = run_one_batch(
                    spec, batch, batch_idx, sub_schema, call_index
                )
                path_filled.setdefault(spec.path, []).extend(result)
                fill_batches_trace.append(
                    {
                        "call_index": call_index,
                        "path": spec.path,
                        "batch_idx": batch_idx,
                        "instance_count": len(batch),
                    }
                )
                if self._debug_dir:
                    fill_debug_batches.append(debug_entry)
        else:

            def run_job(
                item: tuple[int, tuple],
            ) -> tuple[str, int, list[dict[str, Any]], dict[str, Any]]:
                call_index, (spec, batch, batch_idx, sub_schema) = item
                return run_one_batch(spec, batch, batch_idx, sub_schema, call_index)

            job_items = [(ci, job) for ci, job in enumerate(fill_jobs)]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                completed = list(ex.map(run_job, job_items))
            by_path: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
            for path, batch_idx, result, debug_entry in completed:
                by_path.setdefault(path, []).append((batch_idx, result))
                fill_batches_trace.append(
                    {
                        "call_index": debug_entry["call_index"],
                        "path": path,
                        "batch_idx": batch_idx,
                        "instance_count": debug_entry["instance_count"],
                    }
                )
                if self._debug_dir:
                    fill_debug_batches.append(debug_entry)
            for path, batch_results in by_path.items():
                batch_results.sort(key=lambda x: x[0])
                path_filled[path] = []
                for _, result in batch_results:
                    path_filled[path].extend(result)
            fill_debug_batches.sort(key=lambda e: e["call_index"])

        # Expand list-under-list paths: one filled object per original descriptor (reuse by canonical key)
        for path, original_descriptors, unique_descriptors, spec in expand_after_fill:
            filled_unique = path_filled.get(path, [])
            if len(filled_unique) != len(unique_descriptors):
                continue
            key_to_filled = {
                _canonical_lookup_key(
                    path,
                    spec,
                    u.get("ids") or {},
                    u.get("__instance_key"),
                ): filled_unique[i]
                for i, u in enumerate(unique_descriptors)
            }
            path_filled[path] = [
                key_to_filled[
                    _canonical_lookup_key(
                        path,
                        spec,
                        d.get("ids") or {},
                        d.get("__instance_key"),
                    )
                ]
                for d in original_descriptors
            ]

        fill_elapsed = time.perf_counter() - fill_start
        if self._debug_dir and fill_debug_batches:
            os.makedirs(self._debug_dir, exist_ok=True)
            with open(os.path.join(self._debug_dir, "fill_pass.json"), "w", encoding="utf-8") as f:
                json.dump({"batches": fill_debug_batches}, f, indent=2, default=str)
        rich_print(
            f"[green][NodesProvisioning][/green] Fill pass done in [cyan]{fill_elapsed:.1f}s[/cyan]."
        )
        return (path_filled, fill_batches_trace, fill_elapsed)

    def _finalize_and_return(
        self,
        path_filled: dict[str, list[dict[str, Any]]],
        path_descriptors: dict[str, list[dict[str, Any]]],
        per_path_counts: dict[str, int],
        id_elapsed: float,
        fill_elapsed: float,
        id_validation_ok: bool,
        id_validation_errors: list[str],
        fill_batches_trace: list[dict[str, Any]],
        start_total: float,
        context: str,
    ) -> dict[str, Any] | None:
        """Merge, resolve conflicts, assemble edges, run quality gate, write debug/trace, return merged or None."""
        rich_print("[blue][StagedExtraction][/blue] Merging and assembling edges...")
        merge_stats: dict[str, int] = {}
        merged = merge_filled_into_root(
            path_filled, path_descriptors, self._catalog, stats=merge_stats
        )
        merged = _maybe_resolve_conflicts(merged, self._catalog, self._llm, context)
        edges_resolved = assemble_edges_from_merged(merged, self._catalog)
        quality_ok, quality_reasons = _evaluate_quality_gate(
            config=self._config,
            per_path_counts=per_path_counts,
            merge_stats=merge_stats,
            merged=merged,
        )
        total_elapsed = time.perf_counter() - start_total
        trace = _build_staged_trace(
            template_name=getattr(self._template, "__name__", "Unknown"),
            config=self._config,
            catalog=self._catalog,
            id_elapsed=id_elapsed,
            fill_elapsed=fill_elapsed,
            total_elapsed=total_elapsed,
            per_path_counts=per_path_counts,
            id_validation_ok=id_validation_ok,
            id_validation_errors=id_validation_errors,
            fill_batches=fill_batches_trace,
            merged_keys=list(merged.keys()) if isinstance(merged, dict) else [],
            edges_count=len(edges_resolved),
            merge_stats=merge_stats,
            quality_gate={"ok": quality_ok, "reasons": quality_reasons},
            fallback_reason=("quality_gate_failed" if not quality_ok else None),
        )
        if self._debug_dir:
            os.makedirs(self._debug_dir, exist_ok=True)
            p = os.path.join(self._debug_dir, "edges_pass.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "edges": [e.to_dict() for e in self._catalog.edges],
                        "resolved": edges_resolved,
                    },
                    f,
                    indent=2,
                )
            with open(
                os.path.join(self._debug_dir, "merged_output.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(merged, f, indent=2, default=str)
            _write_staged_trace(self._debug_dir, trace)
        if self._on_trace is not None:
            self._on_trace(trace)
        logger.info("%s Staged extraction done in %.2fs", "[StagedExtraction]", total_elapsed)
        rich_print(
            f"[green][StagedExtraction][/green] Staged extraction complete in [cyan]{total_elapsed:.1f}s[/cyan] "
            f"(ID: [cyan]{id_elapsed:.1f}s[/cyan], fill: [cyan]{fill_elapsed:.1f}s[/cyan])."
        )
        if not quality_ok:
            logger.warning("[StagedExtraction] Quality gate failed: %s", ", ".join(quality_reasons))
            return None
        return merged

    def extract(self, markdown: str, context: str = "document") -> dict[str, Any] | None:
        start_total = time.perf_counter()
        n_paths = len(self._catalog.nodes)
        rich_print(
            f"[blue][StagedExtraction][/blue] Nodes Catalog: [cyan]{n_paths}[/cyan] paths identified"
        )
        if self._debug_dir:
            catalog_mod.write_catalog_artifact(self._catalog, self._debug_dir)
        rich_print("[blue][IdentityDiscovery][/blue] ID pass in progress...")
        flat_nodes, per_path_counts, id_elapsed, id_validation_ok, id_validation_errors = (
            self._run_id_pass(markdown, context)
        )
        path_to_descriptors = flat_nodes_to_path_lists(flat_nodes)
        path_descriptors = {p: list(lst) for p, lst in path_to_descriptors.items()}
        path_filled, fill_batches_trace, fill_elapsed = self._run_fill_pass(
            markdown, context, path_to_descriptors
        )
        return self._finalize_and_return(
            path_filled,
            path_descriptors,
            per_path_counts,
            id_elapsed,
            fill_elapsed,
            id_validation_ok,
            id_validation_errors,
            fill_batches_trace,
            start_total,
            context,
        )
