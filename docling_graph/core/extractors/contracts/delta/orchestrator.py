"""Delta extraction orchestrator (chunk -> token batch -> graph merge -> template projection)."""

from __future__ import annotations

import json
import logging
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from pydantic import BaseModel, ValidationError

from .catalog import build_delta_node_catalog
from .helpers import (
    DedupPolicy,
    build_dedup_policy,
    chunk_batches_by_token_limit,
    filter_entity_nodes_by_identity,
    merge_delta_graphs,
    per_path_counts,
    sanitize_batch_echo_from_graph,
)
from .ir_normalizer import DeltaIrNormalizerConfig, normalize_delta_ir_batch_results
from .models import DeltaGraph
from .prompts import format_batch_markdown, get_delta_batch_prompt
from .resolvers import DeltaResolverConfig, resolve_post_merge_graph
from .schema_mapper import (
    build_catalog_prompt_block,
    build_delta_semantic_guide,
    project_graph_to_template_root,
)

logger = logging.getLogger(__name__)


@dataclass
class DeltaOrchestratorConfig:
    """Config for delta extraction orchestration."""

    max_pass_retries: int = 1
    llm_batch_token_size: int = 1024
    parallel_workers: int = 1
    quality_require_root: bool = True
    quality_min_instances: int = 1
    quality_max_parent_lookup_miss: int = 4
    quality_adaptive_parent_lookup: bool = True
    # -1 disables the corresponding gate (recommended default for noisy LLM outputs).
    quality_max_unknown_path_drops: int = -1
    quality_max_id_mismatch: int = -1
    quality_max_nested_property_drops: int = -1
    quality_require_relationships: bool = False
    quality_require_structural_attachments: bool = False
    # Optional semantic completeness gates (-1 disables).
    quality_min_non_empty_properties: int = -1
    quality_min_root_non_empty_fields: int = -1
    quality_min_non_empty_by_path: dict[str, int] = field(default_factory=dict)
    quality_max_orphan_ratio: float = -1.0
    quality_max_canonical_duplicates: int = -1
    batch_split_max_retries: int = 1
    identity_filter_enabled: bool = True
    identity_filter_strict: bool = False
    ir_normalizer: DeltaIrNormalizerConfig = field(default_factory=DeltaIrNormalizerConfig)
    resolvers: DeltaResolverConfig = field(default_factory=DeltaResolverConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> DeltaOrchestratorConfig:
        conf = config or {}
        resolver_mode = str(conf.get("delta_resolvers_mode", "semantic")).lower()
        if resolver_mode not in {"off", "fuzzy", "semantic", "chain"}:
            resolver_mode = "off"
        return cls(
            max_pass_retries=int(conf.get("max_pass_retries", 1) or 1),
            llm_batch_token_size=int(conf.get("llm_batch_token_size", 1024) or 1024),
            parallel_workers=max(1, int(conf.get("parallel_workers", 1) or 1)),
            quality_require_root=bool(conf.get("delta_quality_require_root", True)),
            quality_min_instances=max(0, int(conf.get("delta_quality_min_instances", 20) or 20)),
            quality_max_parent_lookup_miss=max(
                0, int(conf.get("delta_quality_max_parent_lookup_miss", 4) or 0)
            ),
            quality_adaptive_parent_lookup=bool(
                conf.get("delta_quality_adaptive_parent_lookup", True)
            ),
            quality_max_unknown_path_drops=int(conf.get("quality_max_unknown_path_drops", -1)),
            quality_max_id_mismatch=int(conf.get("quality_max_id_mismatch", -1)),
            quality_max_nested_property_drops=int(
                conf.get("quality_max_nested_property_drops", -1)
            ),
            quality_require_relationships=bool(
                conf.get("delta_quality_require_relationships", False)
            ),
            quality_require_structural_attachments=bool(
                conf.get("delta_quality_require_structural_attachments", False)
            ),
            quality_min_non_empty_properties=int(
                conf.get("delta_quality_min_non_empty_properties", -1)
            ),
            quality_min_root_non_empty_fields=int(
                conf.get("delta_quality_min_root_non_empty_fields", -1)
            ),
            quality_min_non_empty_by_path={
                str(k): int(v)
                for k, v in dict(conf.get("delta_quality_min_non_empty_by_path", {}) or {}).items()
            },
            quality_max_orphan_ratio=float(conf.get("delta_quality_max_orphan_ratio", -1.0)),
            quality_max_canonical_duplicates=int(
                conf.get("delta_quality_max_canonical_duplicates", -1)
            ),
            batch_split_max_retries=max(0, int(conf.get("delta_batch_split_max_retries", 1) or 0)),
            identity_filter_enabled=bool(conf.get("delta_identity_filter_enabled", True)),
            identity_filter_strict=bool(conf.get("delta_identity_filter_strict", False)),
            ir_normalizer=DeltaIrNormalizerConfig(
                validate_paths=bool(conf.get("delta_normalizer_validate_paths", True)),
                canonicalize_ids=bool(conf.get("delta_normalizer_canonicalize_ids", True)),
                strip_nested_properties=bool(
                    conf.get("delta_normalizer_strip_nested_properties", True)
                ),
                attach_provenance=bool(conf.get("delta_normalizer_attach_provenance", True)),
            ),
            resolvers=DeltaResolverConfig(
                enabled=bool(conf.get("delta_resolvers_enabled", True)),
                mode=resolver_mode,
                fuzzy_threshold=float(conf.get("delta_resolver_fuzzy_threshold", 0.8) or 0.8),
                semantic_threshold=float(conf.get("delta_resolver_semantic_threshold", 0.8) or 0.8),
                properties=list(conf.get("delta_resolver_properties", []) or []),
                paths=list(conf.get("delta_resolver_paths", []) or []),
            ),
        )


class DeltaOrchestrator:
    """Run delta extraction over token-batched chunks with parallel LLM calls."""

    _DEBUG_PROMPT_USER_TRUNCATE = 8000
    _TRACE_TOP_N = 10

    def __init__(
        self,
        *,
        llm_call_fn: Callable[..., dict | list | None],
        template: type[BaseModel],
        config: DeltaOrchestratorConfig,
        structured_output: bool = True,
        debug_dir: str | None = None,
        on_trace: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._llm = llm_call_fn
        self._template = template
        self._config = config
        self._structured_output = structured_output
        self._debug_dir = debug_dir or ""
        self._on_trace = on_trace
        self._catalog = build_delta_node_catalog(template)

    def _write_debug_json(self, file_name: str, payload: Any) -> None:
        if not self._debug_dir:
            return
        os.makedirs(self._debug_dir, exist_ok=True)
        path = os.path.join(self._debug_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True, default=str)

    def _run_one_batch(
        self,
        *,
        batch_index: int,
        total_batches: int,
        batch: list[tuple[int, str, int]],
        semantic_guide: str,
        catalog_block: str,
    ) -> tuple[int, dict[str, Any] | None, list[str], float]:
        batch_markdown = format_batch_markdown([chunk for _, chunk, _ in batch])
        delta_schema_json = json.dumps(DeltaGraph.model_json_schema(), indent=2)

        errors: list[str] = []
        start = time.time()
        prompt = get_delta_batch_prompt(
            batch_markdown=batch_markdown,
            schema_semantic_guide=semantic_guide,
            path_catalog_block=catalog_block,
            batch_index=batch_index,
            total_batches=total_batches,
        )

        last_parsed: dict | list | None = None
        for attempt in range(self._config.max_pass_retries + 1):
            call_prompt = prompt
            if attempt > 0 and errors:
                feedback = "\n".join(f"- {err}" for err in errors[:15])
                call_prompt = {
                    "system": prompt["system"],
                    "user": (
                        prompt["user"]
                        + "\n\n=== FIX ===\n"
                        + "Previous output had validation issues. Correct them and return full JSON.\n"
                        + feedback
                    ),
                }
            parsed = self._llm(
                prompt=call_prompt,
                schema_json=delta_schema_json,
                context=f"delta_batch_{batch_index}",
                response_top_level="object",
                response_schema_name="delta_extraction",
            )
            last_parsed = parsed
            if not isinstance(parsed, dict):
                errors = ["Output must be a JSON object with {nodes, relationships}."]
                continue
            try:
                validated = DeltaGraph.model_validate(parsed)
                if self._debug_dir:
                    self._write_debug_json(
                        f"delta_batch_{batch_index}_success.json",
                        {
                            "prompt": {
                                "system": prompt["system"],
                                "user": prompt["user"][: self._DEBUG_PROMPT_USER_TRUNCATE],
                            },
                            "output": validated.model_dump(),
                        },
                    )
                elapsed = time.time() - start
                return batch_index, validated.model_dump(), [], elapsed
            except ValidationError as exc:
                errors = []
                for err in exc.errors():
                    loc = " -> ".join(str(v) for v in err.get("loc", ()))
                    errors.append(f"{loc}: {err.get('msg', 'invalid value')}")

        elapsed = time.time() - start
        if self._debug_dir:
            self._write_debug_json(
                f"delta_batch_{batch_index}_failed.json",
                {
                    "prompt": {
                        "system": prompt["system"],
                        "user": prompt["user"][: self._DEBUG_PROMPT_USER_TRUNCATE],
                    },
                    "last_output": last_parsed,
                    "errors": errors,
                },
            )
        return batch_index, None, errors, elapsed

    def _quality_gate(
        self,
        merged_root: dict[str, Any],
        path_counts: dict[str, int],
        merge_stats: dict[str, int],
        normalizer_stats: dict[str, int],
        property_sparsity: dict[str, Any],
        relationship_count: int = 0,
        orphan_ratio: float = 0.0,
        canonical_duplicates_total: int = 0,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        total_instances = sum(path_counts.values())
        attached_node_count = int(merge_stats.get("attached_node_count", 0) or 0)
        allowed_parent_lookup_miss = max(0, self._config.quality_max_parent_lookup_miss)
        if self._config.quality_adaptive_parent_lookup and path_counts.get("", 0) > 0:
            adaptive_cap = min(150, max(8, total_instances // 4))
            allowed_parent_lookup_miss = max(allowed_parent_lookup_miss, adaptive_cap)
        if self._config.quality_require_root and path_counts.get("", 0) <= 0:
            reasons.append("missing_root_instance")
        if attached_node_count < max(0, self._config.quality_min_instances):
            reasons.append("insufficient_instances")
        if merge_stats.get("parent_lookup_miss", 0) > allowed_parent_lookup_miss:
            reasons.append("parent_lookup_miss")
        if (
            self._config.quality_max_unknown_path_drops >= 0
            and normalizer_stats.get("unknown_path_dropped", 0)
            > self._config.quality_max_unknown_path_drops
        ):
            reasons.append("unknown_path_dropped")
        if (
            self._config.quality_max_id_mismatch >= 0
            and normalizer_stats.get("id_key_mismatch", 0) > self._config.quality_max_id_mismatch
        ):
            reasons.append("id_key_mismatch")
        if (
            self._config.quality_max_nested_property_drops >= 0
            and normalizer_stats.get("nested_property_dropped", 0)
            > self._config.quality_max_nested_property_drops
        ):
            reasons.append("nested_property_dropped")
        if (
            self._config.quality_require_relationships
            and merge_stats.get("attached_list_items", 0) <= 0
        ):
            reasons.append("missing_relationship_attachments")
        if self._config.quality_require_structural_attachments:
            attached_total = merge_stats.get("attached_list_items", 0) + merge_stats.get(
                "attached_scalar_items", 0
            )
            expected_attachments = (
                attached_total
                + merge_stats.get("parent_lookup_miss", 0)
                + merge_stats.get("missing_parent_descriptor", 0)
            )
            if expected_attachments > 0 and attached_total <= 0:
                reasons.append("missing_structural_attachments")
        if self._config.quality_require_relationships and relationship_count <= 0:
            reasons.append("missing_relationships")
        if (
            self._config.quality_min_non_empty_properties >= 0
            and int(property_sparsity.get("total_non_empty_properties", 0))
            < self._config.quality_min_non_empty_properties
        ):
            reasons.append("insufficient_non_empty_properties")
        if (
            self._config.quality_min_root_non_empty_fields >= 0
            and int(property_sparsity.get("root_non_empty_fields", 0))
            < self._config.quality_min_root_non_empty_fields
        ):
            reasons.append("insufficient_root_fields")
        per_path_non_empty = property_sparsity.get("non_empty_properties_by_path", {})
        if isinstance(per_path_non_empty, dict):
            for path, minimum in self._config.quality_min_non_empty_by_path.items():
                if int(minimum) < 0:
                    continue
                if int(per_path_non_empty.get(path, 0)) < int(minimum):
                    reasons.append(f"insufficient_path_fields:{path}")
        if (
            self._config.quality_max_orphan_ratio >= 0
            and orphan_ratio > self._config.quality_max_orphan_ratio
        ):
            reasons.append("orphan_ratio_exceeded")
        if (
            self._config.quality_max_canonical_duplicates >= 0
            and canonical_duplicates_total > self._config.quality_max_canonical_duplicates
        ):
            reasons.append("canonical_identity_duplicates")
        if not merged_root:
            reasons.append("empty_output")
        return (len(reasons) == 0), reasons

    @staticmethod
    def _is_non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list | dict | tuple | set):
            return len(value) > 0
        return True

    def _compute_property_sparsity(
        self, *, merged_graph: dict[str, Any], merged_root: dict[str, Any]
    ) -> dict[str, Any]:
        by_path: dict[str, int] = {}
        total_non_empty = 0
        for node in merged_graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            path = str(node.get("path") or "")
            props = node.get("properties")
            if not isinstance(props, dict):
                continue
            non_empty_count = sum(1 for value in props.values() if self._is_non_empty(value))
            if non_empty_count <= 0:
                continue
            by_path[path] = int(by_path.get(path, 0)) + non_empty_count
            total_non_empty += non_empty_count

        root_non_empty_fields = 0
        if isinstance(merged_root, dict):
            for value in merged_root.values():
                if isinstance(value, dict | list):
                    continue
                if self._is_non_empty(value):
                    root_non_empty_fields += 1

        return {
            "total_non_empty_properties": total_non_empty,
            "root_non_empty_fields": root_non_empty_fields,
            "non_empty_properties_by_path": by_path,
        }

    @staticmethod
    def _canonicalize_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = " ".join(value.strip().split()).casefold()
            normalized = unicodedata.normalize("NFKD", text)
            return "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return str(value)

    def _compute_orphan_diagnostics(
        self, merged_root: dict[str, Any], path_counts: dict[str, int]
    ) -> dict[str, Any]:
        orphan_items = merged_root.get("__orphans__", [])
        if not isinstance(orphan_items, list):
            return {"orphan_count": 0, "orphan_ratio": 0.0, "orphans_by_parent_path": {}}
        total_instances = max(1, int(sum(path_counts.values())))
        per_parent: dict[str, int] = {}
        for item in orphan_items:
            if not isinstance(item, dict):
                continue
            parent_path = str(item.get("parent_path") or "")
            per_parent[parent_path] = int(per_parent.get(parent_path, 0)) + 1
        return {
            "orphan_count": len(orphan_items),
            "orphan_ratio": float(len(orphan_items) / total_instances),
            "orphans_by_parent_path": per_parent,
        }

    def _compute_duplicate_canonical_identities(
        self,
        *,
        merged_graph: dict[str, Any],
        dedup_policy: dict[str, DedupPolicy],
    ) -> dict[str, int]:
        per_path: dict[str, dict[tuple[str, ...], int]] = {}
        for node in merged_graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            path = str(node.get("path") or "")
            policy = dedup_policy.get(path)
            if policy is None or not policy.identity_fields:
                continue
            raw_ids = node.get("ids")
            ids: dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
            key = tuple(
                self._canonicalize_value(ids.get(field)) for field in policy.identity_fields
            )
            if not key or all(part == "" for part in key):
                continue
            per_path.setdefault(path, {})
            per_path[path][key] = int(per_path[path].get(key, 0)) + 1

        duplicates_by_path: dict[str, int] = {}
        for path, key_counts in per_path.items():
            duplicates = sum(max(0, count - 1) for count in key_counts.values())
            if duplicates > 0:
                duplicates_by_path[path] = duplicates
        return duplicates_by_path

    def extract(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        context: str,
    ) -> dict[str, Any] | None:
        """Run complete delta extraction and return template-shaped root dict."""

        if not chunks:
            return None

        token_counts: list[int] = []
        if chunk_metadata:
            for idx, chunk in enumerate(chunks):
                if idx < len(chunk_metadata) and isinstance(chunk_metadata[idx], dict):
                    token_counts.append(
                        int(chunk_metadata[idx].get("token_count", max(1, len(chunk.split()))))
                    )
                else:
                    token_counts.append(max(1, len(chunk.split())))
        else:
            token_counts = [max(1, len(chunk.split())) for chunk in chunks]

        batch_plan = chunk_batches_by_token_limit(
            chunks,
            token_counts,
            max_batch_tokens=self._config.llm_batch_token_size,
        )

        schema_dict = self._template.model_json_schema()
        semantic_guide = build_delta_semantic_guide(self._template, schema_dict)
        catalog_block = build_catalog_prompt_block(self._catalog)

        successful_results: list[dict[str, Any]] = []
        effective_batch_plan: list[list[tuple[int, str, int]]] = []
        failed_batches: list[tuple[int, list[tuple[int, str, int]], list[str]]] = []
        batch_errors: dict[int, list[str]] = {}
        batch_timings: list[dict[str, Any]] = []
        batch_split_retries = 0
        split_batch_sizes: list[list[int]] = []
        split_failures = 0

        if self._config.parallel_workers > 1 and len(batch_plan) > 1:
            with ThreadPoolExecutor(max_workers=self._config.parallel_workers) as pool:
                futures = {
                    pool.submit(
                        self._run_one_batch,
                        batch_index=i,
                        total_batches=len(batch_plan),
                        batch=batch,
                        semantic_guide=semantic_guide,
                        catalog_block=catalog_block,
                    ): i
                    for i, batch in enumerate(batch_plan)
                }
                for future in as_completed(futures):
                    original_batch_idx = futures[future]
                    batch_idx, graph_dict, errors, elapsed = future.result()
                    batch_timings.append({"batch_index": batch_idx, "elapsed_seconds": elapsed})
                    if graph_dict is not None:
                        successful_results.append(graph_dict)
                        effective_batch_plan.append(batch_plan[original_batch_idx])
                    elif errors:
                        failed_batches.append((batch_idx, batch_plan[original_batch_idx], errors))
        else:
            for i, batch in enumerate(batch_plan):
                batch_idx, graph_dict, errors, elapsed = self._run_one_batch(
                    batch_index=i,
                    total_batches=len(batch_plan),
                    batch=batch,
                    semantic_guide=semantic_guide,
                    catalog_block=catalog_block,
                )
                batch_timings.append({"batch_index": batch_idx, "elapsed_seconds": elapsed})
                if graph_dict is not None:
                    successful_results.append(graph_dict)
                    effective_batch_plan.append(batch)
                elif errors:
                    failed_batches.append((batch_idx, batch, errors))

        split_round = 0
        while failed_batches and split_round < self._config.batch_split_max_retries:
            next_failed_batches: list[tuple[int, list[tuple[int, str, int]], list[str]]] = []
            for parent_idx, failed_batch, failed_errors in failed_batches:
                if len(failed_batch) <= 1:
                    split_failures += 1
                    batch_errors[parent_idx] = failed_errors
                    continue
                midpoint = len(failed_batch) // 2
                sub_batches = [failed_batch[:midpoint], failed_batch[midpoint:]]
                split_batch_sizes.append([len(sub_batches[0]), len(sub_batches[1])])
                batch_split_retries += 1
                for sub_idx, sub_batch in enumerate(sub_batches):
                    synthetic_idx = (parent_idx * 100) + ((split_round + 1) * 10) + sub_idx
                    batch_idx, graph_dict, errors, elapsed = self._run_one_batch(
                        batch_index=synthetic_idx,
                        total_batches=max(2, len(sub_batches)),
                        batch=sub_batch,
                        semantic_guide=semantic_guide,
                        catalog_block=catalog_block,
                    )
                    batch_timings.append(
                        {
                            "batch_index": batch_idx,
                            "elapsed_seconds": elapsed,
                            "split_from": parent_idx,
                            "split_round": split_round + 1,
                        }
                    )
                    if graph_dict is not None:
                        successful_results.append(graph_dict)
                        effective_batch_plan.append(sub_batch)
                    elif errors:
                        next_failed_batches.append((batch_idx, sub_batch, errors))
            failed_batches = next_failed_batches
            split_round += 1

        if failed_batches:
            split_failures += len(failed_batches)
            for failed_idx, _failed_batch, failed_errors in failed_batches:
                batch_errors[failed_idx] = failed_errors

        if not successful_results:
            return None

        dedup_policy = build_dedup_policy(self._catalog)
        normalized_batch_results, normalizer_stats = normalize_delta_ir_batch_results(
            batch_results=successful_results,
            batch_plan=effective_batch_plan,
            chunk_metadata=chunk_metadata,
            catalog=self._catalog,
            dedup_policy=dedup_policy,
            config=self._config.ir_normalizer,
        )
        merged_graph = merge_delta_graphs(normalized_batch_results, dedup_policy=dedup_policy)
        sanitize_batch_echo_from_graph(merged_graph)
        merge_core_stats = (
            merged_graph.get("__merge_stats", {})
            if isinstance(merged_graph.get("__merge_stats"), dict)
            else {}
        )
        merged_graph, resolver_stats = resolve_post_merge_graph(
            merged_graph,
            dedup_policy=dedup_policy,
            config=self._config.resolvers,
        )
        merged_graph, identity_filter_stats = filter_entity_nodes_by_identity(
            merged_graph,
            self._catalog,
            dedup_policy,
            enabled=self._config.identity_filter_enabled,
            strict=self._config.identity_filter_strict,
        )
        merged_root, merge_stats = project_graph_to_template_root(merged_graph, self._template)
        path_counts = per_path_counts(merged_graph.get("nodes", []))
        property_sparsity = self._compute_property_sparsity(
            merged_graph=merged_graph, merged_root=merged_root
        )
        orphan_diagnostics = self._compute_orphan_diagnostics(merged_root, path_counts)
        duplicate_identities_by_path = self._compute_duplicate_canonical_identities(
            merged_graph=merged_graph,
            dedup_policy=dedup_policy,
        )
        merge_counters = {k: v for k, v in merge_stats.items() if isinstance(v, int)}
        quality_ok, quality_reasons = self._quality_gate(
            merged_root,
            path_counts,
            merge_counters,
            normalizer_stats,
            property_sparsity,
            relationship_count=len(merged_graph.get("relationships", [])),
            orphan_ratio=float(orphan_diagnostics.get("orphan_ratio", 0.0)),
            canonical_duplicates_total=sum(duplicate_identities_by_path.values()),
        )
        missing_by_path = normalizer_stats.get("id_missing_required_by_path", {})
        if isinstance(missing_by_path, dict):
            top_missing_id_paths = sorted(
                (
                    {"path": str(path), "missing_required_ids": int(count)}
                    for path, count in missing_by_path.items()
                ),
                key=lambda x: cast(int, x.get("missing_required_ids", 0)),
                reverse=True,
            )[: self._TRACE_TOP_N]
        else:
            top_missing_id_paths = []

        unknown_path_examples = normalizer_stats.get("unknown_path_examples", [])
        if not isinstance(unknown_path_examples, list):
            unknown_path_examples = []
        raw_parent_lookup = merge_stats.get("parent_lookup_miss_examples", [])
        parent_lookup_examples = raw_parent_lookup if isinstance(raw_parent_lookup, list) else []

        trace = {
            "contract": "delta",
            "context": context,
            "chunk_count": len(chunks),
            "batch_count": len(batch_plan),
            "parallel_workers": self._config.parallel_workers,
            "llm_batch_token_size": self._config.llm_batch_token_size,
            "batch_timings": sorted(batch_timings, key=lambda x: x["batch_index"]),
            "batch_errors": batch_errors,
            "path_counts": path_counts,
            "normalizer_stats": normalizer_stats,
            "dedup_policy": {
                path: {
                    "node_type": p.node_type,
                    "identity_fields": list(p.identity_fields),
                    "fallback_text_fields": list(p.fallback_text_fields),
                    "allowed_match_fields": list(p.allowed_match_fields),
                    "is_entity": p.is_entity,
                }
                for path, p in dedup_policy.items()
            },
            "merge_stats": merge_stats,
            "merge_core_stats": merge_core_stats,
            "identity_filter": identity_filter_stats,
            "resolver": resolver_stats,
            "quality_gate": {"ok": quality_ok, "reasons": quality_reasons},
            "diagnostics": {
                "top_missing_id_paths": top_missing_id_paths,
                "unknown_path_examples": unknown_path_examples[: self._TRACE_TOP_N],
                "parent_lookup_miss_examples": parent_lookup_examples[: self._TRACE_TOP_N],
                "property_sparsity": property_sparsity,
                "orphan_count": orphan_diagnostics.get("orphan_count", 0),
                "orphan_ratio": orphan_diagnostics.get("orphan_ratio", 0.0),
                "orphans_by_parent_path": orphan_diagnostics.get("orphans_by_parent_path", {}),
                "duplicate_canonical_identity_by_path": duplicate_identities_by_path,
                "batch_split_retries": batch_split_retries,
                "split_batch_sizes": split_batch_sizes,
                "split_failures": split_failures,
            },
            "quality_thresholds": {
                "delta_quality_require_root": self._config.quality_require_root,
                "delta_quality_min_instances": self._config.quality_min_instances,
                "delta_quality_max_parent_lookup_miss": self._config.quality_max_parent_lookup_miss,
                "delta_quality_adaptive_parent_lookup": self._config.quality_adaptive_parent_lookup,
                "delta_quality_require_relationships": self._config.quality_require_relationships,
                "delta_quality_require_structural_attachments": self._config.quality_require_structural_attachments,
                "delta_quality_min_non_empty_properties": self._config.quality_min_non_empty_properties,
                "delta_quality_min_root_non_empty_fields": self._config.quality_min_root_non_empty_fields,
                "delta_quality_min_non_empty_by_path": self._config.quality_min_non_empty_by_path,
                "delta_quality_max_orphan_ratio": self._config.quality_max_orphan_ratio,
                "delta_quality_max_canonical_duplicates": self._config.quality_max_canonical_duplicates,
            },
        }

        if self._debug_dir:
            self._write_debug_json("delta_trace.json", trace)
            self._write_debug_json("delta_merged_graph.json", merged_graph)
            self._write_debug_json("delta_merged_output.json", merged_root)

        if self._on_trace is not None:
            self._on_trace(trace)

        if not quality_ok:
            logger.warning(
                "[DeltaExtraction] Quality gate failed: %s | path_counts=%s | normalizer_stats=%s",
                ", ".join(quality_reasons),
                path_counts,
                normalizer_stats,
            )
            return None

        return merged_root
