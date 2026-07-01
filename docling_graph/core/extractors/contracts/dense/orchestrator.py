"""
Dense extraction orchestrator: Phase 1 (skeleton) and Phase 2 (fill).

Fully autonomous: no imports from other contracts.
"""

from __future__ import annotations

import json
import logging
import os
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
from .models import DenseSkeletonGraph
from .prompts import (
    build_skeleton_catalog_block,
    format_batch_markdown,
    get_fill_batch_prompt,
    get_skeleton_batch_prompt,
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


def normalize_skeleton_node(
    raw: dict[str, Any],
    allowed_paths: set[str],
    *,
    source_batch_index: int | None = None,
) -> dict[str, Any] | None:
    path = str(raw.get("path") or "").strip()
    if path not in allowed_paths:
        return None
    ids = raw.get("ids")
    if not isinstance(ids, dict):
        ids = {}
    ids = {str(k): str(v) for k, v in ids.items() if v is not None}
    parent = raw.get("parent")
    ancestry = raw.get("ancestry")
    if isinstance(ancestry, list) and len(ancestry) > 0:
        last = ancestry[-1]
        if isinstance(last, dict):
            parent = {"path": str(last.get("path") or "").strip(), "ids": last.get("ids") or {}}
        else:
            parent = last
    if parent is not None and isinstance(parent, dict):
        parent = {"path": str(parent.get("path") or "").strip(), "ids": parent.get("ids") or {}}
        if isinstance(parent["ids"], dict):
            parent["ids"] = {str(k): str(v) for k, v in parent["ids"].items() if v is not None}
    result: dict[str, Any] = {"path": path, "ids": ids, "parent": parent}
    if source_batch_index is not None:
        result["_source_batch_index"] = source_batch_index
    return result


def merge_skeleton_batches(
    batch_results: list[list[dict[str, Any]]],
    catalog: NodeCatalog,
) -> list[dict[str, Any]]:
    """Dedupe skeleton nodes across batches, accumulating every source batch index.

    The union of source batches is kept so Phase 2 can scope its fill context to
    the document regions where the node was actually observed.
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
    return list(by_key.values())


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


def merge_filled_into_root(
    path_filled: dict[str, list[dict[str, Any]]],
    path_descriptors: dict[str, list[dict[str, Any]]],
    catalog: NodeCatalog,
) -> dict[str, Any]:
    root: dict[str, Any] = {}
    spec_by_path = {s.path: s for s in catalog.nodes}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
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
            root[field_name] = filled_list if is_list else (filled_list[0] if filled_list else None)
            continue
        parent_spec = spec_by_path.get(parent_path)
        if not parent_spec:
            continue
        for i, obj in enumerate(filled_list):
            if isinstance(obj, dict):
                desc = descriptors[i] if i < len(descriptors) else {}
                parent = desc.get("parent")
                if isinstance(parent, dict):
                    parent_ids = parent.get("ids") or {}
                    parent_key = _canonical_lookup_key(parent_path, parent_spec, parent_ids)
                    parent_obj = lookup.get(parent_key)
                    if parent_obj is not None:
                        if is_list:
                            parent_obj.setdefault(field_name, []).append(obj)
                        else:
                            parent_obj[field_name] = obj
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
        for f in spec.id_fields:
            if f in ids and f not in clean:
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
        return cls(
            max_pass_retries=int(c.get("max_pass_retries", 1) or 1),
            skeleton_batch_tokens=skeleton_batch_tokens,
            fill_nodes_cap=int(c.get("dense_fill_nodes_cap", 5) or 5),
            parallel_workers=max(1, int(c.get("parallel_workers", 1) or 1)),
            quality_require_root=bool(c.get("dense_quality_require_root", True)),
            quality_min_instances=max(0, int(c.get("dense_quality_min_instances", 1) or 1)),
            prune_barren_branches=bool(c.get("dense_prune_barren_branches", False)),
            fill_context_mode=fill_context_mode,
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
            if isinstance(out, dict) and "nodes" in out:
                try:
                    validated = DenseSkeletonGraph.model_validate(out)
                    normalized_batch = []
                    for n in validated.nodes:
                        raw = n.model_dump()
                        norm = normalize_skeleton_node(
                            raw, allowed_paths, source_batch_index=batch_idx
                        )
                        if norm is not None:
                            normalized_batch.append(norm)
                    return normalized_batch, truncated
                except Exception as e:
                    logger.warning("Dense skeleton batch %s validation: %s", batch_idx, e)
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
        out = self._llm(
            prompt=prompt,
            schema_json=json.dumps(wrapped_schema),
            context=f"{context}_dense_fill_{path}",
            response_top_level="object",
            response_schema_name="dense_fill",
        )
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

    def run(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        full_markdown: str,
        context: str = "document",
    ) -> dict[str, Any] | None:
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
        skeleton_results: list[list[dict[str, Any]]]
        phase1_start = time.perf_counter()
        spec_by_path = {s.path: s for s in self._catalog.nodes}
        total_batches = len(batches)

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
        else:
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
            skeleton_results = [lst if lst is not None else [] for lst in skeleton_results]
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
        if self._debug_dir:
            self._write_debug("dense_skeleton_graph.json", {"nodes": merged_skeleton})

        path_counts: dict[str, int] = {}
        for n in merged_skeleton:
            p = n.get("path") or ""
            path_counts[p] = path_counts.get(p, 0) + 1
        total = len(merged_skeleton)
        if self._config.quality_require_root and path_counts.get("", 0) <= 0:
            logger.warning("Dense Phase 1: no root instance")
            if self._on_trace:
                self._on_trace(
                    {"contract": "dense", "phase1_quality": False, "reason": "missing_root"}
                )
            return None
        if total < self._config.quality_min_instances:
            logger.warning("Dense Phase 1: too few instances (%s)", total)
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
        root = merge_filled_into_root(path_filled, path_descriptors, self._catalog)
        if self._config.prune_barren_branches:
            root = prune_barren_branches(root, self._catalog)
        if self._on_trace:
            self._on_trace(
                {
                    "contract": "dense",
                    "phase1_elapsed": round(phase1_elapsed, 3),
                    "phase2_elapsed": round(phase2_elapsed, 3),
                    "skeleton_nodes": len(merged_skeleton),
                    "path_counts": path_counts,
                }
            )
        return root
