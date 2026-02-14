"""Delta runtime orchestrator (contract-local implementation)."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .catalog import build_delta_node_catalog
from .helpers import (
    build_dedup_policy,
    chunk_batches_by_token_limit,
    merge_delta_graphs,
    per_path_counts,
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
    max_pass_retries: int = 1
    llm_batch_token_size: int = 2048
    parallel_workers: int = 1
    quality_require_root: bool = True
    quality_min_instances: int = 1
    quality_max_parent_lookup_miss: int = 4
    quality_adaptive_parent_lookup: bool = True
    quality_max_unknown_path_drops: int = -1
    quality_max_id_mismatch: int = -1
    quality_max_nested_property_drops: int = -1
    quality_require_relationships: bool = False
    quality_min_non_empty_properties: int = -1
    quality_min_root_non_empty_fields: int = -1
    quality_min_non_empty_by_path: dict[str, int] = field(default_factory=dict)
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
            llm_batch_token_size=int(conf.get("llm_batch_token_size", 2048) or 2048),
            parallel_workers=max(1, int(conf.get("parallel_workers", 1) or 1)),
            quality_require_root=bool(conf.get("delta_quality_require_root", conf.get("quality_require_root", True))),
            quality_min_instances=int(conf.get("delta_quality_min_instances", conf.get("quality_min_instances", 1)) or 1),
            quality_max_parent_lookup_miss=int(conf.get("delta_quality_max_parent_lookup_miss", conf.get("quality_max_parent_lookup_miss", 4)) or 0),
            quality_adaptive_parent_lookup=bool(conf.get("delta_quality_adaptive_parent_lookup", True)),
            quality_max_unknown_path_drops=int(conf.get("quality_max_unknown_path_drops", -1)),
            quality_max_id_mismatch=int(conf.get("quality_max_id_mismatch", -1)),
            quality_max_nested_property_drops=int(conf.get("quality_max_nested_property_drops", -1)),
            quality_require_relationships=bool(conf.get("delta_quality_require_relationships", False)),
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
            ir_normalizer=DeltaIrNormalizerConfig(
                validate_paths=bool(conf.get("delta_normalizer_validate_paths", True)),
                canonicalize_ids=bool(conf.get("delta_normalizer_canonicalize_ids", True)),
                strip_nested_properties=bool(conf.get("delta_normalizer_strip_nested_properties", True)),
                attach_provenance=bool(conf.get("delta_normalizer_attach_provenance", True)),
            ),
            resolvers=DeltaResolverConfig(
                enabled=bool(conf.get("delta_resolvers_enabled", True)),
                mode=resolver_mode,
                fuzzy_threshold=float(conf.get("delta_resolver_fuzzy_threshold", 0.9) or 0.9),
                semantic_threshold=float(conf.get("delta_resolver_semantic_threshold", 0.9) or 0.9),
                properties=list(conf.get("delta_resolver_properties", []) or []),
                paths=list(conf.get("delta_resolver_paths", []) or []),
            ),
        )


class DeltaOrchestrator:
    _DEBUG_PROMPT_USER_TRUNCATE = 8000

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
        with open(os.path.join(self._debug_dir, file_name), "w", encoding="utf-8") as f:
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
                    "user": prompt["user"] + "\n\n=== FIX ===\nPrevious output had validation issues.\n" + feedback,
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
                elapsed = time.time() - start
                return batch_index, validated.model_dump(), [], elapsed
            except ValidationError as exc:
                errors = []
                for err in exc.errors():
                    loc = " -> ".join(str(v) for v in err.get("loc", ()))
                    errors.append(f"{loc}: {err.get('msg', 'invalid value')}")
        elapsed = time.time() - start
        if self._debug_dir:
            self._write_debug_json(f"delta_batch_{batch_index}_failed.json", {"prompt": prompt, "last_output": last_parsed, "errors": errors})
        return batch_index, None, errors, elapsed

    def _quality_gate(
        self,
        merged_root: dict[str, Any],
        path_counts: dict[str, int],
        merge_stats: dict[str, int],
        normalizer_stats: dict[str, int],
        property_sparsity: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        total_instances = sum(path_counts.values())
        allowed_parent_lookup_miss = max(0, self._config.quality_max_parent_lookup_miss)
        if self._config.quality_adaptive_parent_lookup and path_counts.get("", 0) > 0:
            # Scale tolerance with document size: allow up to ~25% of instances as parent misses.
            adaptive_cap = min(150, max(8, total_instances // 4))
            allowed_parent_lookup_miss = max(allowed_parent_lookup_miss, adaptive_cap)
        if self._config.quality_require_root and path_counts.get("", 0) <= 0:
            reasons.append("missing_root_instance")
        if total_instances < max(0, self._config.quality_min_instances):
            reasons.append("insufficient_instances")
        if merge_stats.get("parent_lookup_miss", 0) > allowed_parent_lookup_miss:
            reasons.append("parent_lookup_miss")
        if self._config.quality_max_unknown_path_drops >= 0 and normalizer_stats.get("unknown_path_dropped", 0) > self._config.quality_max_unknown_path_drops:
            reasons.append("unknown_path_dropped")
        if self._config.quality_max_id_mismatch >= 0 and normalizer_stats.get("id_key_mismatch", 0) > self._config.quality_max_id_mismatch:
            reasons.append("id_key_mismatch")
        if self._config.quality_max_nested_property_drops >= 0 and normalizer_stats.get("nested_property_dropped", 0) > self._config.quality_max_nested_property_drops:
            reasons.append("nested_property_dropped")
        if self._config.quality_require_relationships and merge_stats.get("attached_list_items", 0) <= 0:
            reasons.append("missing_relationship_attachments")
        if self._config.quality_min_non_empty_properties >= 0 and int(
            property_sparsity.get("total_non_empty_properties", 0)
        ) < self._config.quality_min_non_empty_properties:
            reasons.append("insufficient_non_empty_properties")
        if self._config.quality_min_root_non_empty_fields >= 0 and int(
            property_sparsity.get("root_non_empty_fields", 0)
        ) < self._config.quality_min_root_non_empty_fields:
            reasons.append("insufficient_root_fields")
        per_path_non_empty = property_sparsity.get("non_empty_properties_by_path", {})
        if isinstance(per_path_non_empty, dict):
            for path, minimum in self._config.quality_min_non_empty_by_path.items():
                if int(minimum) < 0:
                    continue
                if int(per_path_non_empty.get(path, 0)) < int(minimum):
                    reasons.append(f"insufficient_path_fields:{path}")
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

    def extract(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        context: str,
    ) -> dict[str, Any] | None:
        if not chunks:
            return None
        token_counts = []
        if chunk_metadata:
            for idx, chunk in enumerate(chunks):
                if idx < len(chunk_metadata) and isinstance(chunk_metadata[idx], dict):
                    token_counts.append(int(chunk_metadata[idx].get("token_count", max(1, len(chunk.split())))))
                else:
                    token_counts.append(max(1, len(chunk.split())))
        else:
            token_counts = [max(1, len(chunk.split())) for chunk in chunks]

        batch_plan = chunk_batches_by_token_limit(chunks, token_counts, max_batch_tokens=self._config.llm_batch_token_size)
        schema_dict = self._template.model_json_schema()
        semantic_guide = build_delta_semantic_guide(self._template, schema_dict)
        catalog_block = build_catalog_prompt_block(self._catalog)

        batch_results: list[tuple[int, dict[str, Any]]] = []
        batch_errors: dict[int, list[str]] = {}
        batch_timings: list[dict[str, Any]] = []

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
                    batch_idx, graph_dict, errors, elapsed = future.result()
                    batch_timings.append({"batch_index": batch_idx, "elapsed_seconds": elapsed})
                    if graph_dict is not None:
                        batch_results.append((batch_idx, graph_dict))
                    elif errors:
                        batch_errors[batch_idx] = errors
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
                    batch_results.append((batch_idx, graph_dict))
                elif errors:
                    batch_errors[batch_idx] = errors
        if not batch_results:
            return None

        batch_results.sort(key=lambda item: item[0])
        ordered_batch_results = [result for _, result in batch_results]
        dedup_policy = build_dedup_policy(self._catalog)
        normalized_batch_results, normalizer_stats = normalize_delta_ir_batch_results(
            batch_results=ordered_batch_results,
            batch_plan=batch_plan,
            chunk_metadata=chunk_metadata,
            catalog=self._catalog,
            dedup_policy=dedup_policy,
            config=self._config.ir_normalizer,
        )
        merged_graph = merge_delta_graphs(normalized_batch_results, dedup_policy=dedup_policy)
        merged_graph, resolver_stats = resolve_post_merge_graph(merged_graph, dedup_policy=dedup_policy, config=self._config.resolvers)
        merged_root, merge_stats = project_graph_to_template_root(merged_graph, self._template)
        path_counts = per_path_counts(merged_graph.get("nodes", []))
        property_sparsity = self._compute_property_sparsity(
            merged_graph=merged_graph, merged_root=merged_root
        )
        merge_counters = {k: v for k, v in merge_stats.items() if isinstance(v, int)}
        quality_ok, quality_reasons = self._quality_gate(
            merged_root, path_counts, merge_counters, normalizer_stats, property_sparsity
        )

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
            "merge_stats": merge_stats,
            "resolver": resolver_stats,
            "quality_gate": {"ok": quality_ok, "reasons": quality_reasons},
            "diagnostics": {"property_sparsity": property_sparsity},
        }
        if self._debug_dir:
            self._write_debug_json("delta_trace.json", trace)
        if self._on_trace is not None:
            self._on_trace(trace)
        if not quality_ok:
            logger.warning("[DeltaExtraction] Quality gate failed: %s", ", ".join(quality_reasons))
            return None
        return merged_root
