"""
Dense extraction orchestrator: Phase 1 (skeleton) and Phase 2 (fill).

Fully autonomous: no imports from other contracts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, cast

from pydantic import BaseModel

from docling_graph.core.utils.entity_name_normalizer import canonicalize_identity_for_dedup

from .catalog import (
    NodeCatalog,
    NodeSpec,
    bottom_up_path_order,
    build_node_catalog,
    build_projected_fill_schema,
    build_skeleton_semantic_guide,
    get_model_for_path,
    skeleton_output_schema,
)
from .models import DenseSkeletonNode
from .prompts import (
    build_skeleton_catalog_block,
    format_batch_markdown,
    get_fill_batch_prompt,
    get_skeleton_batch_prompt,
    get_skeleton_reconciliation_prompt,
    reconciliation_output_schema,
)
from .resolvers import DenseResolverConfig, resolve_skeleton_nodes

logger = logging.getLogger(__name__)

# Max times a truncated skeleton batch may be halved before keeping the partial result.
# Batches are already small (a few chunks), so a depth of 4 fully isolates pathological chunks.
_MAX_SKELETON_SPLIT_DEPTH = 4


def _skeleton_identity_key(
    node: dict[str, Any],
    spec_by_path: dict[str, NodeSpec],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    path = str(node.get("path") or "").strip()
    spec = spec_by_path.get(path)
    ids = node.get("ids") or {}
    if not isinstance(ids, dict):
        ids = {}
    if spec and spec.id_fields:
        ordered = tuple(
            (f, canonicalize_identity_for_dedup(f, ids.get(f)))
            for f in spec.id_fields
            if ids.get(f) is not None
        )
        if ordered:
            return (path, tuple(sorted(ordered, key=lambda x: x[0])))
    norm = tuple(
        sorted(
            (str(k), canonicalize_identity_for_dedup(k, v)) for k, v in ids.items() if v is not None
        )
    )
    return (path, norm if norm else (("__key", str(id(node))),))


def chunk_batches_by_token_limit(
    chunks: list[str],
    token_counts: list[int] | None,
    *,
    max_batch_tokens: int,
) -> list[list[tuple[int, str, int]]]:
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be > 0")
    if token_counts is None or len(token_counts) != len(chunks):
        token_counts = [max(1, len(c.split())) for c in chunks]
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


def _canonical_catalog_path(path: str, allowed_paths: set[str]) -> str | None:
    """Map a model-emitted path onto a catalog path, tolerating missing [] suffixes.

    Small models frequently drop the list markers (e.g. "studies.experiments"
    instead of "studies[].experiments[]"); such drift must not discard the node.
    """
    p = path.strip()
    if p in allowed_paths:
        return p
    stripped = p.replace("[]", "")
    for candidate in allowed_paths:
        if candidate.replace("[]", "") == stripped:
            return candidate
    return None


def normalize_skeleton_batch(
    nodes: list[DenseSkeletonNode],
    allowed_paths: set[str],
    *,
    source_batch_index: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve batch-local integer handles into (path, ids) parent references.

    Two passes: first canonicalize paths and index nodes by their handle ``i``;
    then resolve each node's parent handle ``p`` to the referenced node's
    (path, ids). An explicit parent object is accepted as fallback when a
    model emits one instead of a handle. Nodes with unknown paths are dropped.
    """
    prepared: list[dict[str, Any]] = []
    by_handle: dict[int, dict[str, Any]] = {}
    for node in nodes:
        path = _canonical_catalog_path(node.path or "", allowed_paths)
        if path is None:
            continue
        ids = {str(k): str(v) for k, v in (node.ids or {}).items() if v is not None}
        entry: dict[str, Any] = {"path": path, "ids": ids, "p": node.p, "parent_ref": node.parent}
        prepared.append(entry)
        if node.i is not None and node.i not in by_handle:
            by_handle[node.i] = entry

    out: list[dict[str, Any]] = []
    for entry in prepared:
        parent: dict[str, Any] | None = None
        handle = entry["p"]
        if handle is not None and handle in by_handle and by_handle[handle] is not entry:
            referenced = by_handle[handle]
            parent = {"path": referenced["path"], "ids": dict(referenced["ids"])}
        elif entry["parent_ref"] is not None:
            ref = entry["parent_ref"]
            ref_path = _canonical_catalog_path(ref.path or "", allowed_paths)
            parent = {
                "path": ref_path if ref_path is not None else (ref.path or ""),
                "ids": {str(k): str(v) for k, v in (ref.ids or {}).items() if v is not None},
            }
        result: dict[str, Any] = {"path": entry["path"], "ids": entry["ids"], "parent": parent}
        if source_batch_index is not None:
            result["_source_batch_index"] = source_batch_index
        out.append(result)
    return out


def merge_skeleton_batches(
    batch_results: list[list[dict[str, Any]]],
    catalog: NodeCatalog,
) -> list[dict[str, Any]]:
    """Dedupe skeleton nodes across batches, accumulating every source batch index.

    The union of source batches is kept so Phase 2 can scope its fill context to
    the document regions where the node was actually observed.

    The root (path "") is a singleton by definition: batches routinely emit it
    with paraphrased identifier values (e.g. title variants), so all root nodes
    are collapsed into the first one instead of trusting id-based dedup.
    """
    spec_by_path = {s.path: s for s in catalog.nodes}
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for batch in batch_results:
        for node in batch:
            key = _skeleton_identity_key(node, spec_by_path)
            source_idx = node.get("_source_batch_index")
            merged = by_key.get(key)
            if merged is None:
                merged = dict(node)
                merged.pop("_source_batch_index", None)
                merged["_source_batch_indexes"] = []
                by_key[key] = merged
            if isinstance(source_idx, int) and source_idx not in merged["_source_batch_indexes"]:
                merged["_source_batch_indexes"].append(source_idx)
    merged_nodes = list(by_key.values())

    roots = [n for n in merged_nodes if (n.get("path") or "") == ""]
    if len(roots) > 1:
        primary = roots[0]
        for extra in roots[1:]:
            for idx in extra.get("_source_batch_indexes", []):
                if idx not in primary["_source_batch_indexes"]:
                    primary["_source_batch_indexes"].append(idx)
        logger.info(
            "Dense skeleton: collapsed %s duplicate root instances into one", len(roots) - 1
        )
        merged_nodes = [n for n in merged_nodes if (n.get("path") or "") != ""]
        merged_nodes.insert(0, primary)
    return merged_nodes


def skeleton_to_descriptors(
    skeleton_nodes: list[dict[str, Any]],
    catalog: NodeCatalog,
) -> dict[str, list[dict[str, Any]]]:
    path_descriptors: dict[str, list[dict[str, Any]]] = {}
    for node in skeleton_nodes:
        path = node.get("path") or ""
        ids = node.get("ids") or {}
        parent = node.get("parent")
        desc = {"path": path, "ids": dict(ids), "parent": parent}
        source_indexes = node.get("_source_batch_indexes")
        if isinstance(source_indexes, list) and source_indexes:
            desc["_source_batch_indexes"] = list(source_indexes)
        path_descriptors.setdefault(path, []).append(desc)
    return path_descriptors


def _canonical_lookup_key(path: str, spec: NodeSpec, ids: dict[str, Any]) -> tuple[Any, ...]:
    if not spec.id_fields:
        return (path, ())
    normalized = tuple(
        sorted((f, canonicalize_identity_for_dedup(f, ids.get(f))) for f in spec.id_fields)
    )
    return (path, normalized)


def _canonical_id_text(ids: dict[str, Any]) -> str:
    """Single canonical string for an id dict, used for fuzzy parent matching."""
    parts = [canonicalize_identity_for_dedup(k, v) for k, v in sorted(ids.items()) if v is not None]
    return " ".join(p for p in parts if p)


def apply_skeleton_reconciliation(
    skeleton_nodes: list[dict[str, Any]],
    merge_groups: list[dict[str, Any]],
    spec_by_path: dict[str, NodeSpec],
) -> tuple[list[dict[str, Any]], int]:
    """Apply validated alias merge groups to the merged skeleton.

    Group indices refer to the per-path instance order of ``skeleton_nodes``.
    The kept node absorbs the merged nodes' source batches, and parent
    references to a merged node are remapped to the kept node's ids. Invalid
    entries (unknown path, out-of-range or self indices) are skipped silently —
    a bad reconciliation response must never damage the skeleton.
    """
    by_path: dict[str, list[dict[str, Any]]] = {}
    for node in skeleton_nodes:
        by_path.setdefault(node.get("path") or "", []).append(node)

    removed: set[int] = set()
    id_remap: dict[Any, dict[str, Any]] = {}
    merged_count = 0
    for group in merge_groups:
        if not isinstance(group, dict):
            continue
        path = group.get("path")
        keep_idx = group.get("keep")
        merge_idxs = group.get("merge")
        instances = by_path.get(path) if isinstance(path, str) else None
        if instances is None or not isinstance(merge_idxs, list):
            continue
        if not isinstance(keep_idx, int) or not (0 <= keep_idx < len(instances)):
            continue
        keep_node = instances[keep_idx]
        if id(keep_node) in removed:
            continue
        for merge_idx in merge_idxs:
            if not isinstance(merge_idx, int) or not (0 <= merge_idx < len(instances)):
                continue
            node = instances[merge_idx]
            if node is keep_node or id(node) in removed:
                continue
            removed.add(id(node))
            merged_count += 1
            id_remap[_skeleton_identity_key(node, spec_by_path)] = dict(keep_node.get("ids") or {})
            keep_sources = keep_node.setdefault("_source_batch_indexes", [])
            for idx in node.get("_source_batch_indexes", []) or []:
                if idx not in keep_sources:
                    keep_sources.append(idx)

    kept_nodes = [n for n in skeleton_nodes if id(n) not in removed]
    if id_remap:
        for node in kept_nodes:
            parent = node.get("parent")
            if not isinstance(parent, dict):
                continue
            parent_key = _skeleton_identity_key(
                {"path": parent.get("path"), "ids": parent.get("ids") or {}}, spec_by_path
            )
            if parent_key in id_remap:
                node["parent"] = {"path": parent.get("path"), "ids": id_remap[parent_key]}
    return kept_nodes, merged_count


def _unique_fuzzy_parent_match(
    parent_path: str,
    parent_ids: dict[str, Any],
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Return the single parent instance whose canonical id text contains (or is
    contained by) the referenced ids, or None when the match is absent/ambiguous."""
    ref_text = _canonical_id_text(parent_ids)
    if not ref_text or len(ref_text) < 3:
        return None
    parent_descs = path_descriptors.get(parent_path, [])
    matches: list[dict[str, Any]] = []
    for i, cand in enumerate(path_filled.get(parent_path, [])):
        if not isinstance(cand, dict):
            continue
        cand_ids = (parent_descs[i].get("ids") if i < len(parent_descs) else None) or {}
        cand_text = _canonical_id_text(cand_ids)
        if cand_text and (ref_text in cand_text or cand_text in ref_text):
            matches.append(cand)
    return matches[0] if len(matches) == 1 else None


def merge_filled_into_root(
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
    catalog: NodeCatalog,
    stats_out: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Attach filled instances to their parents, rescuing drifted parent references.

    LLMs (especially small ones) drift on parent identifiers, so a strict
    (path, ids) lookup silently drops entire subtrees. Resolution ladder per
    instance: exact id match -> unique parent instance -> unique fuzzy
    (canonical containment) id match -> id-only placeholder parent created up
    the chain. Only instances that survive none of these are dropped, and
    every recovery/drop is counted in stats_out.
    """
    root: dict[str, Any] = {}
    spec_by_path = {s.path: s for s in catalog.nodes}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    stats = {
        "attached_exact": 0,
        "recovered_single_parent": 0,
        "recovered_fuzzy": 0,
        "recovered_placeholder": 0,
        "dropped": 0,
    }
    for spec in catalog.nodes:
        path = spec.path
        filled_list = path_filled.get(path, [])
        descriptors = path_descriptors.get(path, [])
        for i, obj in enumerate(filled_list):
            if isinstance(obj, dict):
                desc = descriptors[i] if i < len(descriptors) else {}
                ids = desc.get("ids") or {}
                key = _canonical_lookup_key(path, spec, ids)
                lookup[key] = obj

    def _attach(parent_obj: dict[str, Any], spec: NodeSpec, obj: dict[str, Any]) -> None:
        if spec.is_list:
            parent_obj.setdefault(spec.field_name, []).append(obj)
        else:
            parent_obj[spec.field_name] = obj

    def _resolve_parent(
        parent_path: str, parent_ids: dict[str, Any], depth: int = 0
    ) -> tuple[dict[str, Any] | None, str]:
        """Return (parent_obj, how) for a child's parent reference; None if unresolvable."""
        parent_spec = spec_by_path.get(parent_path)
        if parent_spec is None:
            return None, ""
        if parent_path == "":
            return root, "exact"
        key = _canonical_lookup_key(parent_path, parent_spec, parent_ids)
        obj = lookup.get(key)
        if obj is not None:
            return obj, "exact"
        instances = [o for o in path_filled.get(parent_path, []) if isinstance(o, dict)]
        if len(instances) == 1:
            lookup[key] = instances[0]
            return instances[0], "single"
        fuzzy_match = _unique_fuzzy_parent_match(
            parent_path, parent_ids, path_filled, path_descriptors
        )
        if fuzzy_match is not None:
            lookup[key] = fuzzy_match
            return fuzzy_match, "fuzzy"
        # Last resort: materialize an id-only parent so the subtree survives.
        if depth >= 3:
            return None, ""
        placeholder = {
            f: parent_ids[f] for f in parent_spec.id_fields if parent_ids.get(f) not in (None, "")
        }
        if not placeholder:
            return None, ""
        grand_obj, _how = _resolve_parent(parent_spec.parent_path, {}, depth + 1)
        if grand_obj is None:
            return None, ""
        _attach(grand_obj, parent_spec, placeholder)
        lookup[key] = placeholder
        return placeholder, "placeholder"

    _how_to_stat = {
        "single": "recovered_single_parent",
        "fuzzy": "recovered_fuzzy",
        "placeholder": "recovered_placeholder",
    }

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
            existing = root.get(field_name)
            if is_list and isinstance(existing, list):
                # A placeholder created for a deeper orphan may already live here.
                existing.extend(o for o in filled_list if o not in existing)
            else:
                root[field_name] = (
                    filled_list if is_list else (filled_list[0] if filled_list else None)
                )
            stats["attached_exact"] += len(filled_list)
            continue
        if parent_path not in spec_by_path:
            continue
        for i, obj in enumerate(filled_list):
            if not isinstance(obj, dict):
                continue
            desc = descriptors[i] if i < len(descriptors) else {}
            parent = desc.get("parent")
            parent_ids = (parent.get("ids") or {}) if isinstance(parent, dict) else {}
            parent_obj, how = _resolve_parent(parent_path, parent_ids)
            if parent_obj is not None:
                _attach(parent_obj, spec, obj)
                stats[_how_to_stat.get(how, "attached_exact")] += 1
            else:
                stats["dropped"] += 1
                logger.warning(
                    "Dense merge: dropped %s instance ids=%s (unresolvable parent %s ids=%s)",
                    path,
                    json.dumps(desc.get("ids") or {}, ensure_ascii=False, default=str)[:120],
                    parent_path,
                    json.dumps(parent_ids, ensure_ascii=False, default=str)[:120],
                )

    recovered = (
        stats["recovered_single_parent"] + stats["recovered_fuzzy"] + stats["recovered_placeholder"]
    )
    if recovered or stats["dropped"]:
        logger.warning(
            "Dense merge: %s drifted parent link(s) recovered "
            "(%s unique-parent, %s fuzzy, %s placeholder), %s instance(s) dropped",
            recovered,
            stats["recovered_single_parent"],
            stats["recovered_fuzzy"],
            stats["recovered_placeholder"],
            stats["dropped"],
        )
    if stats_out is not None:
        stats_out.update(stats)
    return root


def _compute_branch_paths(catalog: NodeCatalog) -> set[str]:
    """Paths that have at least one other path extending them (container/branch nodes)."""
    all_paths = set(catalog.paths())
    return {p for p in all_paths if any(q != p and q.startswith(p) for q in all_paths)}


def prune_barren_branches(root: dict[str, Any], catalog: NodeCatalog) -> dict[str, Any]:
    """
    Remove branch nodes that are childless and have no non-identity scalar data.
    Domain-agnostic: uses only catalog path structure (branch vs leaf) and graph topology.
    """
    branch_paths = _compute_branch_paths(catalog)
    spec_by_path = {s.path: s for s in catalog.nodes}
    children_by_parent: dict[str, list[NodeSpec]] = {}
    for s in catalog.nodes:
        children_by_parent.setdefault(s.parent_path, []).append(s)

    def is_barren(node_obj: dict[str, Any], path: str) -> bool:
        spec = spec_by_path.get(path)
        if not spec:
            return False
        id_fields = set(spec.id_fields or [])
        for k, v in node_obj.items():
            if k in id_fields:
                continue
            if v is None:
                continue
            if v == "":
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            if isinstance(v, dict) and not v:
                continue
            return False
        return True

    def has_children(node_obj: dict[str, Any], path: str) -> bool:
        for cs in children_by_parent.get(path, []):
            val = node_obj.get(cs.field_name)
            if cs.is_list and isinstance(val, list) and len(val) > 0:
                return True
            if not cs.is_list and val is not None:
                return True
        return False

    def prune_in_place(obj: dict[str, Any], current_path: str) -> None:
        for cs in children_by_parent.get(current_path, []):
            fn = cs.field_name
            val = obj.get(fn)
            if cs.is_list and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        prune_in_place(item, cs.path)
                if cs.path in branch_paths:
                    kept = [
                        x
                        for x in val
                        if not (
                            isinstance(x, dict)
                            and not has_children(x, cs.path)
                            and is_barren(x, cs.path)
                        )
                    ]
                    obj[fn] = kept
            elif not cs.is_list and isinstance(val, dict):
                prune_in_place(val, cs.path)

    prune_in_place(root, "")
    return root


def _is_usable_id_value(value: Any) -> bool:
    """A filled identity value is usable when it is a non-empty scalar."""
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return True
    return isinstance(value, str) and bool(value.strip())


def _sanitize_filled(
    items: list[Any],
    descriptors: list[dict[str, Any]],
    spec: NodeSpec,
    model: type[BaseModel] | None,
) -> list[dict[str, Any]]:
    allowed = set(spec.id_fields)
    if model is not None:
        allowed |= set(model.model_fields.keys())
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(items):
        src = obj if isinstance(obj, dict) else {}
        clean = {k: v for k, v in src.items() if k in allowed}
        desc = descriptors[i] if i < len(descriptors) else {}
        ids = desc.get("ids") or {}
        # Identity values were already captured during the skeleton phase; when
        # the fill response omits one or returns something unusable (null,
        # empty string, a nested object), restore the known value instead of
        # letting downstream salvage synthesize an empty placeholder.
        for f in spec.id_fields:
            if f in ids and not _is_usable_id_value(clean.get(f)):
                clean[f] = ids[f]
        out.append(clean)
    return out


@dataclass
class DenseOrchestratorConfig:
    max_pass_retries: int = 1
    skeleton_batch_tokens: int = 1024
    fill_nodes_cap: int = 5
    parallel_workers: int = 1
    quality_require_root: bool = True
    quality_min_instances: int = 1
    prune_barren_branches: bool = False
    # "scoped": fill prompts only include the skeleton batches where the node was
    # observed (plus the document head); "full": always send the whole document.
    fill_context_mode: str = "scoped"
    # One id-space LLM call after skeleton merge that collapses same-entity
    # aliases discovered at different granularities across batches.
    skeleton_reconciliation: bool = True
    resolvers: DenseResolverConfig = field(default_factory=DenseResolverConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> DenseOrchestratorConfig:
        c = config or {}
        res = c.get("dense_resolvers") or {}
        resolvers = DenseResolverConfig(
            enabled=bool(res.get("enabled", False)),
            mode=str(res.get("mode", "off")).lower(),
            fuzzy_threshold=float(res.get("fuzzy_threshold", 0.8) or 0.8),
            semantic_threshold=float(res.get("semantic_threshold", 0.8) or 0.8),
            allow_merge_different_ids=bool(res.get("allow_merge_different_ids", False)),
        )
        raw_skeleton_tokens = int(c.get("dense_skeleton_batch_tokens", 1024) or 1024)
        skeleton_batch_tokens = (
            min(raw_skeleton_tokens, 4096) if raw_skeleton_tokens > 4096 else raw_skeleton_tokens
        )
        fill_context_mode = str(c.get("dense_fill_context", "scoped") or "scoped").lower()
        if fill_context_mode not in ("scoped", "full"):
            fill_context_mode = "scoped"
        skeleton_reconciliation = bool(c.get("dense_skeleton_reconciliation", True))
        return cls(
            max_pass_retries=int(c.get("max_pass_retries", 1) or 1),
            skeleton_batch_tokens=skeleton_batch_tokens,
            fill_nodes_cap=int(c.get("dense_fill_nodes_cap", 5) or 5),
            parallel_workers=max(1, int(c.get("parallel_workers", 1) or 1)),
            quality_require_root=bool(c.get("dense_quality_require_root", True)),
            quality_min_instances=max(0, int(c.get("dense_quality_min_instances", 1) or 1)),
            prune_barren_branches=bool(c.get("dense_prune_barren_branches", False)),
            fill_context_mode=fill_context_mode,
            skeleton_reconciliation=skeleton_reconciliation,
            resolvers=resolvers,
        )


class DenseOrchestrator:
    def __init__(
        self,
        *,
        llm_call_fn: Callable[..., dict | list | None],
        template: type[BaseModel],
        config: DenseOrchestratorConfig,
        debug_dir: str | None = None,
        on_trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm_call_fn
        self._template = template
        self._config = config
        self._debug_dir = debug_dir or ""
        self._on_trace = on_trace
        self._catalog = build_node_catalog(template)
        # Per-run observability (exposed after run() as last_run_stats).
        self.last_run_stats: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._counter_lock = threading.Lock()

    def _bump(self, counter: str) -> None:
        with self._counter_lock:
            self._counters[counter] = self._counters.get(counter, 0) + 1

    def _write_debug(self, name: str, data: Any) -> None:
        if not self._debug_dir:
            return
        os.makedirs(self._debug_dir, exist_ok=True)
        path = os.path.join(self._debug_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _call_skeleton_batch(
        self,
        batch_idx: int,
        batch: list[tuple[int, str, int]],
        total_batches: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        already_found_str: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run the LLM for one skeleton batch.

        Returns (normalized_nodes, truncated) where truncated is True if the model
        hit its output limit (so the caller may split the batch and retry).
        """
        batch_md = format_batch_markdown(batch)
        prompt = get_skeleton_batch_prompt(
            batch_markdown=batch_md,
            catalog_block=catalog_block,
            batch_index=batch_idx,
            total_batches=total_batches,
            allowed_paths=self._catalog.paths(),
            global_context=global_context,
            already_found=already_found_str,
            semantic_guide=semantic_guide,
        )
        truncated = False
        for attempt in range(self._config.max_pass_retries + 1):
            diag: dict[str, Any] = {}
            out = self._llm(
                prompt=prompt,
                schema_json=schema_json,
                context=f"{context}_dense_skeleton_{batch_idx}",
                response_top_level="object",
                response_schema_name="dense_skeleton",
                _diagnostics_out=diag,
            )
            if diag.get("truncated"):
                truncated = True
                self._bump("truncation_count")
            if isinstance(out, dict) and isinstance(out.get("nodes"), list):
                raw_nodes = out["nodes"]
                validated_nodes: list[DenseSkeletonNode] = []
                invalid = 0
                # Per-node validation: small models often mix malformed entries
                # (e.g. echoed schema fragments) with valid ones; keep the good
                # nodes instead of discarding the whole batch.
                for element in raw_nodes:
                    try:
                        validated_nodes.append(DenseSkeletonNode.model_validate(element))
                    except Exception:
                        invalid += 1
                if invalid:
                    logger.warning(
                        "Dense skeleton batch %s: skipped %s invalid node entr%s, kept %s",
                        batch_idx,
                        invalid,
                        "y" if invalid == 1 else "ies",
                        len(validated_nodes),
                    )
                normalized_batch = normalize_skeleton_batch(
                    validated_nodes, allowed_paths, source_batch_index=batch_idx
                )
                if normalized_batch or not raw_nodes:
                    return normalized_batch, truncated
                # Every entry was invalid: treat as a failed pass and retry.
            if attempt == self._config.max_pass_retries:
                return [], truncated
        return [], truncated

    def _run_one_skeleton_batch(
        self,
        batch_idx: int,
        batch: list[tuple[int, str, int]],
        total_batches: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        spec_by_path: dict[str, NodeSpec],
        already_found_str: str | None,
        _depth: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Run one skeleton batch; returns (batch_idx, normalized_batch_list).

        If the model truncates its output and the batch holds more than one chunk,
        the batch is split in half and each half is retried (domain-agnostic recovery
        for documents too entity-dense to fit one skeleton response in the model's
        output budget). Sub-batches keep the same batch_idx so fill-context provenance
        still points at the original batch text. Uses already_found_str=None in
        parallel mode.
        """
        nodes, truncated = self._call_skeleton_batch(
            batch_idx,
            batch,
            total_batches,
            catalog_block,
            allowed_paths,
            global_context,
            semantic_guide,
            schema_json,
            context,
            already_found_str,
        )
        if truncated and len(batch) > 1 and _depth < _MAX_SKELETON_SPLIT_DEPTH:
            self._bump("split_count")
            mid = len(batch) // 2
            logger.warning(
                "Dense skeleton batch %s truncated (%s chunks); splitting into %s + %s and retrying",
                batch_idx,
                len(batch),
                mid,
                len(batch) - mid,
            )
            merged: list[dict[str, Any]] = []
            for sub in (batch[:mid], batch[mid:]):
                _, sub_nodes = self._run_one_skeleton_batch(
                    batch_idx,
                    sub,
                    total_batches,
                    catalog_block,
                    allowed_paths,
                    global_context,
                    semantic_guide,
                    schema_json,
                    context,
                    spec_by_path,
                    already_found_str,
                    _depth + 1,
                )
                merged.extend(sub_nodes)
            return (batch_idx, merged)
        return (batch_idx, nodes)

    def _run_skeleton_reconciliation(
        self,
        merged_skeleton: list[dict[str, Any]],
        spec_by_path: dict[str, NodeSpec],
        context: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """One id-space LLM call that collapses same-entity aliases across batches.

        Parallel batches (and granularity drift within a document) produce the
        same entity at several specificity levels (e.g. "LFP slurry batch"
        alongside "LFP_20vol_5wtPVDF_4wtCB"). Pure string similarity cannot
        judge granularity, so a single cheap call over the identifier lists
        (no document content) proposes alias groups; anything invalid in the
        response is ignored, so this pass can only merge, never lose nodes.
        """
        instances_by_path: dict[str, list[dict[str, Any]]] = {}
        for node in merged_skeleton:
            path = node.get("path") or ""
            if path == "":
                continue  # the root is already collapsed to a singleton
            instances_by_path.setdefault(path, []).append(node.get("ids") or {})
        instances_by_path = {
            path: ids_list for path, ids_list in instances_by_path.items() if len(ids_list) >= 2
        }
        if not instances_by_path:
            return merged_skeleton, 0
        prompt = get_skeleton_reconciliation_prompt(instances_by_path)
        out = self._llm(
            prompt=prompt,
            schema_json=json.dumps(reconciliation_output_schema()),
            context=f"{context}_dense_reconcile",
            response_top_level="object",
            response_schema_name="dense_reconcile",
        )
        merges = out.get("merges") if isinstance(out, dict) else None
        if not isinstance(merges, list) or not merges:
            return merged_skeleton, 0
        reconciled, merged_count = apply_skeleton_reconciliation(
            merged_skeleton, merges, spec_by_path
        )
        if merged_count:
            logger.info(
                "Dense reconciliation: merged %s alias instance(s) into more specific ones",
                merged_count,
            )
        return reconciled, merged_count

    def _finalize_run_stats(
        self,
        skeleton_nodes: int,
        reconciliation_merged: int,
        merge_stats: dict[str, int],
        gate_failure: str | None = None,
    ) -> None:
        """Publish per-run observability counters as last_run_stats."""
        dropped = merge_stats.get("dropped", 0)
        stats: dict[str, Any] = {
            "skeleton_nodes": skeleton_nodes,
            "truncation_count": self._counters.get("truncation_count", 0),
            "split_count": self._counters.get("split_count", 0),
            "reconciliation_merged": reconciliation_merged,
            "merge_orphans_dropped": dropped,
            "merge_recovered": (
                merge_stats.get("recovered_single_parent", 0)
                + merge_stats.get("recovered_fuzzy", 0)
                + merge_stats.get("recovered_placeholder", 0)
            ),
            "retention_pct": (
                round(100.0 * (1 - dropped / skeleton_nodes), 1) if skeleton_nodes else 0.0
            ),
        }
        if gate_failure:
            stats["quality_gate_failure"] = gate_failure
        self.last_run_stats = stats

    def _build_fill_context(
        self,
        path: str,
        batch_descriptors: list[dict[str, Any]],
        batch_texts: list[str],
        full_markdown: str,
        global_head: str,
    ) -> str:
        """Markdown context for one fill batch.

        In "scoped" mode, only the skeleton batches where the instances were
        observed are included (plus the document head for shared context),
        which keeps Phase 2 token cost proportional to the entities being
        filled instead of resending the whole document for every batch.
        The root instance and nodes without provenance always get the full
        document, as does any scoped context that would not actually shrink it.
        """
        if self._config.fill_context_mode != "scoped" or path == "":
            return full_markdown
        indexes: set[int] = set()
        for desc in batch_descriptors:
            for idx in desc.get("_source_batch_indexes") or []:
                if isinstance(idx, int) and 0 <= idx < len(batch_texts):
                    indexes.add(idx)
        if not indexes:
            return full_markdown
        parts: list[str] = []
        if global_head and 0 not in indexes:
            parts.append(global_head)
        parts.extend(batch_texts[i] for i in sorted(indexes))
        scoped = "\n\n".join(parts)
        if len(scoped) >= len(full_markdown):
            return full_markdown
        return scoped

    def _run_one_fill_batch(
        self,
        path: str,
        spec: NodeSpec,
        batch_descriptors: list[dict[str, Any]],
        batch_index: int,
        sub_schema: str,
        fill_markdown: str,
        context: str,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Run one fill batch; returns (path, batch_index, sanitized_list)."""
        prompt = get_fill_batch_prompt(
            markdown=fill_markdown,
            path=path,
            spec=spec,
            descriptors=batch_descriptors,
            projected_schema_json=sub_schema,
        )
        wrapped_schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": json.loads(sub_schema)}},
            "required": ["items"],
        }
        fill_diag: dict[str, Any] = {}
        out = self._llm(
            prompt=prompt,
            schema_json=json.dumps(wrapped_schema),
            context=f"{context}_dense_fill_{path}",
            response_top_level="object",
            response_schema_name="dense_fill",
            _diagnostics_out=fill_diag,
        )
        if fill_diag.get("truncated"):
            self._bump("truncation_count")
        if isinstance(out, dict) and "items" in out:
            items = out["items"] if isinstance(out["items"], list) else []
        elif isinstance(out, list):
            items = out
        else:
            items = []
        # Exactly one filled object per requested instance: pad short responses
        # with empty objects (ids are restored from descriptors during sanitize)
        # so skeleton nodes are never silently dropped, and discard extras that
        # have no matching descriptor (they carry no usable parent linkage).
        if len(items) < len(batch_descriptors):
            items = [*items, *([{}] * (len(batch_descriptors) - len(items)))]
        elif len(items) > len(batch_descriptors):
            items = items[: len(batch_descriptors)]
        model = get_model_for_path(self._template, path)
        sanitized = _sanitize_filled(items, batch_descriptors, spec, model)
        return (path, batch_index, sanitized)

    def _run_skeleton_phase(
        self,
        *,
        batches: list[list[tuple[int, str, int]]],
        workers: int,
        catalog_block: str,
        allowed_paths: set[str],
        global_context: str | None,
        semantic_guide: str | None,
        schema_json: str,
        context: str,
        spec_by_path: dict[str, NodeSpec],
    ) -> list[list[dict[str, Any]]]:
        """Execute all Phase 1 skeleton batches, sequentially or in parallel."""
        total_batches = len(batches)
        skeleton_results: list[list[dict[str, Any]]]
        if workers <= 1 or total_batches <= 1:
            # Sequential: preserve already_found for cross-batch consistency
            already_found: list[str] = []
            skeleton_results = []
            for batch_idx, batch in enumerate(batches):
                already_str = "\n".join(already_found[-50:]) if already_found else None
                _, normalized_batch = self._run_one_skeleton_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    total_batches=total_batches,
                    catalog_block=catalog_block,
                    allowed_paths=allowed_paths,
                    global_context=global_context,
                    semantic_guide=semantic_guide,
                    schema_json=schema_json,
                    context=context,
                    spec_by_path=spec_by_path,
                    already_found_str=already_str,
                )
                for node in normalized_batch:
                    key = _skeleton_identity_key(node, spec_by_path)
                    already_found.append(json.dumps({"path": key[0], "ids": dict(key[1])}))
                skeleton_results.append(normalized_batch)
            return skeleton_results

        # Parallel: no already_found; merge_skeleton_batches and resolvers dedupe
        logger.info(
            "Dense Phase 1: running %s skeleton batches with %s workers",
            total_batches,
            workers,
        )
        skeleton_results = [None] * total_batches  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._run_one_skeleton_batch,
                    batch_idx=batch_idx,
                    batch=batch,
                    total_batches=total_batches,
                    catalog_block=catalog_block,
                    allowed_paths=allowed_paths,
                    global_context=global_context,
                    semantic_guide=semantic_guide,
                    schema_json=schema_json,
                    context=context,
                    spec_by_path=spec_by_path,
                    already_found_str=None,
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    idx, normalized_batch = future.result()
                    skeleton_results[idx] = normalized_batch
                except Exception as e:
                    logger.warning("Dense skeleton batch %s failed: %s", batch_idx, e)
                    skeleton_results[batch_idx] = []
        return [lst if lst is not None else [] for lst in skeleton_results]

    def run(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        full_markdown: str,
        context: str = "document",
    ) -> dict[str, Any] | None:
        self._counters = {}
        self.last_run_stats = {}
        token_counts: list[int] | None = None
        if chunk_metadata and len(chunk_metadata) == len(chunks):
            raw_counts = [m.get("token_count") for m in chunk_metadata]
            if not any(t is None for t in raw_counts):
                token_counts = [int(cast(int, t)) for t in raw_counts]
        allowed_paths = set(self._catalog.paths())
        schema_json = json.dumps(skeleton_output_schema(self._catalog.paths()))
        catalog_block = build_skeleton_catalog_block(self._catalog)
        semantic_guide = build_skeleton_semantic_guide(self._catalog)
        global_context = chunks[0][:2000] if chunks and isinstance(chunks[0], str) else None

        batches = chunk_batches_by_token_limit(
            chunks, token_counts, max_batch_tokens=self._config.skeleton_batch_tokens
        )
        logger.info(
            "Dense Phase 1: %s batches (skeleton_batch_tokens=%s)",
            len(batches),
            self._config.skeleton_batch_tokens,
        )
        workers = max(1, self._config.parallel_workers)
        if workers > 1 and len(batches) == 1:
            logger.warning(
                "Dense Phase 1: only one batch; parallel workers will not be used. "
                "Ensure chunk_max_tokens (per chunk) and dense_skeleton_batch_tokens are set "
                "so long documents split into multiple chunks and batches."
            )
        phase1_start = time.perf_counter()
        spec_by_path = {s.path: s for s in self._catalog.nodes}
        skeleton_results = self._run_skeleton_phase(
            batches=batches,
            workers=workers,
            catalog_block=catalog_block,
            allowed_paths=allowed_paths,
            global_context=global_context,
            semantic_guide=semantic_guide,
            schema_json=schema_json,
            context=context,
            spec_by_path=spec_by_path,
        )
        phase1_elapsed = time.perf_counter() - phase1_start
        merged_skeleton = merge_skeleton_batches(skeleton_results, self._catalog)
        if self._config.resolvers.enabled:

            def key_fn(n: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
                return _skeleton_identity_key(n, spec_by_path)

            merged_skeleton, resolver_stats = resolve_skeleton_nodes(
                merged_skeleton, key_fn, self._config.resolvers
            )
            if self._on_trace and resolver_stats.get("merged_count", 0) > 0:
                self._on_trace({"contract": "dense", "phase1_resolvers": resolver_stats})
        reconciliation_merged = 0
        if self._config.skeleton_reconciliation and len(merged_skeleton) > 1:
            merged_skeleton, reconciliation_merged = self._run_skeleton_reconciliation(
                merged_skeleton, spec_by_path, context
            )
        if self._debug_dir:
            self._write_debug("dense_skeleton_graph.json", {"nodes": merged_skeleton})

        path_counts: dict[str, int] = {}
        for n in merged_skeleton:
            p = n.get("path") or ""
            path_counts[p] = path_counts.get(p, 0) + 1
        total = len(merged_skeleton)
        if self._config.quality_require_root and path_counts.get("", 0) <= 0:
            logger.warning("Dense Phase 1: no root instance")
            self._finalize_run_stats(total, reconciliation_merged, {}, gate_failure="missing_root")
            if self._on_trace:
                self._on_trace(
                    {"contract": "dense", "phase1_quality": False, "reason": "missing_root"}
                )
            return None
        if total < self._config.quality_min_instances:
            logger.warning("Dense Phase 1: too few instances (%s)", total)
            self._finalize_run_stats(
                total, reconciliation_merged, {}, gate_failure="insufficient_instances"
            )
            if self._on_trace:
                self._on_trace(
                    {
                        "contract": "dense",
                        "phase1_quality": False,
                        "reason": "insufficient_instances",
                    }
                )
            return None

        path_descriptors = skeleton_to_descriptors(merged_skeleton, self._catalog)
        path_filled: dict[str, list[dict[str, Any]]] = {}
        fill_paths = [p for p in bottom_up_path_order(self._catalog) if path_descriptors.get(p)]
        phase2_start = time.perf_counter()

        batch_texts = [format_batch_markdown(batch) for batch in batches]
        global_head = chunks[0][:2000] if chunks and isinstance(chunks[0], str) else ""

        # Build flat list of fill jobs:
        # (path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown)
        fill_jobs: list[tuple[str, NodeSpec, list[dict[str, Any]], int, str, str]] = []
        for path in fill_paths:
            descriptors = path_descriptors[path]
            spec = spec_by_path.get(path)
            if not spec:
                continue
            sub_schema = build_projected_fill_schema(self._template, spec, self._catalog)
            fill_batches = [
                descriptors[i : i + self._config.fill_nodes_cap]
                for i in range(0, len(descriptors), self._config.fill_nodes_cap)
            ]
            for batch_index, batch_descriptors in enumerate(fill_batches):
                fill_markdown = self._build_fill_context(
                    path, batch_descriptors, batch_texts, full_markdown, global_head
                )
                fill_jobs.append(
                    (path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown)
                )

        if workers <= 1 or len(fill_jobs) <= 1:
            for path, spec, batch_descriptors, _bi, sub_schema, fill_markdown in fill_jobs:
                _p, _bi, sanitized = self._run_one_fill_batch(
                    path=path,
                    spec=spec,
                    batch_descriptors=batch_descriptors,
                    batch_index=_bi,
                    sub_schema=sub_schema,
                    fill_markdown=fill_markdown,
                    context=context,
                )
                path_filled.setdefault(path, []).extend(sanitized)
        else:
            logger.info(
                "Dense Phase 2: running %s fill jobs with %s workers",
                len(fill_jobs),
                workers,
            )
            results_by_path: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fill_futures: dict[
                    Future[tuple[str, int, list[dict[str, Any]]]], tuple[str, int]
                ] = {
                    ex.submit(
                        self._run_one_fill_batch,
                        path=path,
                        spec=spec,
                        batch_descriptors=batch_descriptors,
                        batch_index=batch_index,
                        sub_schema=sub_schema,
                        fill_markdown=fill_markdown,
                        context=context,
                    ): (path, batch_index)
                    for path, spec, batch_descriptors, batch_index, sub_schema, fill_markdown in fill_jobs
                }
                for future in as_completed(fill_futures):  # type: ignore[assignment]
                    path, batch_index = fill_futures[future]  # type: ignore[index]
                    try:
                        p, bi, sanitized = future.result()  # type: ignore[misc]
                        results_by_path.setdefault(p, []).append((bi, sanitized))
                    except Exception as e:
                        logger.warning(
                            "Dense fill job %s batch %s failed: %s", path, batch_index, e
                        )
            for path, pairs in results_by_path.items():
                pairs.sort(key=lambda x: x[0])
                for _bi, sanitized in pairs:
                    path_filled.setdefault(path, []).extend(sanitized)

        phase2_elapsed = time.perf_counter() - phase2_start
        merge_stats: dict[str, int] = {}
        root = merge_filled_into_root(
            path_filled, path_descriptors, self._catalog, stats_out=merge_stats
        )
        if self._config.prune_barren_branches:
            root = prune_barren_branches(root, self._catalog)
        self._finalize_run_stats(len(merged_skeleton), reconciliation_merged, merge_stats)
        if self._debug_dir:
            self._write_debug("dense_merge_stats.json", merge_stats)
            self._write_debug("dense_run_stats.json", self.last_run_stats)
        if self._on_trace:
            self._on_trace(
                {
                    "contract": "dense",
                    "phase1_elapsed": round(phase1_elapsed, 3),
                    "phase2_elapsed": round(phase2_elapsed, 3),
                    "skeleton_nodes": len(merged_skeleton),
                    "path_counts": path_counts,
                    "merge_stats": merge_stats,
                    "run_stats": self.last_run_stats,
                }
            )
        return root
