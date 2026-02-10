"""
Orchestrator for staged extraction (skeleton -> groups -> repair).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from . import prompts
from .materialize_nested_edges import materialize_nested_edges
from .reconciliation import ReconciliationPolicy, merge_pass_output

logger = logging.getLogger(__name__)


@dataclass
class StagedPassConfig:
    max_fields_per_group: int = 6
    max_skeleton_fields: int = 10
    max_repair_rounds: int = 2
    max_pass_retries: int = 1
    quality_depth: int = 3
    include_prior_context: bool = True
    merge_similarity_fallback: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> StagedPassConfig:
        if not values:
            return cls()
        return cls(
            max_fields_per_group=int(values.get("max_fields_per_group", 6)),
            max_skeleton_fields=int(values.get("max_skeleton_fields", 10)),
            max_repair_rounds=int(values.get("max_repair_rounds", 2)),
            max_pass_retries=int(values.get("max_pass_retries", 1)),
            quality_depth=int(values.get("quality_depth", 3)),
            include_prior_context=bool(values.get("include_prior_context", True)),
            merge_similarity_fallback=bool(values.get("merge_similarity_fallback", True)),
        )


@dataclass
class ExtractionPassResult:
    stage: str
    success: bool
    data: dict[str, Any]
    attempts: int
    errors: list[str]
    duration_seconds: float = 0.0


class StagedOrchestrator:
    """
    Coordinates staged extraction passes and deterministic reconciliation.
    """

    def __init__(
        self,
        llm_call_fn: Callable[[dict[str, str], str, str], dict | list | None],
        schema_json: str,
        template: type[BaseModel] | None,
        config: StagedPassConfig,
        on_pass_complete: Callable[[Any], None] | None = None,
        run_llm_consolidation: Callable[
            [dict[str, Any], str, list[str], str], dict[str, Any] | None
        ] | None = None,
    ) -> None:
        self._llm_call_fn = llm_call_fn
        self._schema_json = schema_json
        self._template = template
        self._config = config
        self._on_pass_complete = on_pass_complete
        self._run_llm_consolidation = run_llm_consolidation
        self._metadata = prompts.get_template_graph_metadata(template, schema_json)
        self._plan = prompts.plan_extraction_passes(
            schema_json=schema_json,
            template_metadata=self._metadata,
            max_fields_per_group=config.max_fields_per_group,
            max_skeleton_fields=config.max_skeleton_fields,
        )
        self._pass_id_counter = 0

    def _emit_pass_telemetry(
        self,
        result: ExtractionPassResult,
        stage_type: str,
        fields_requested: list[str],
        fields_returned: list[str],
        duration_seconds: float,
    ) -> None:
        if self._on_pass_complete is None:
            return
        self._pass_id_counter += 1
        from .....pipeline.trace import StagedPassData

        data = StagedPassData(
            pass_id=self._pass_id_counter,
            stage_name=result.stage,
            stage_type=stage_type,
            success=result.success,
            attempts=result.attempts,
            errors=result.errors,
            duration_seconds=duration_seconds,
            fields_requested=fields_requested,
            fields_returned=fields_returned,
            metadata={},
        )
        self._on_pass_complete(data)

    def _run_pass(
        self,
        *,
        stage_name: str,
        stage_type: str,
        prompt: dict[str, str],
        schema_json: str,
        fields_requested: list[str],
    ) -> ExtractionPassResult:
        attempts = self._config.max_pass_retries + 1
        errors: list[str] = []
        start = time.perf_counter()
        for attempt_idx in range(attempts):
            logger.info(
                "[staged] pass=%s stage_type=%s attempt=%s/%s fields=%s",
                stage_name,
                stage_type,
                attempt_idx + 1,
                attempts,
                fields_requested[:10] if len(fields_requested) > 10 else fields_requested,
            )
            result = self._llm_call_fn(prompt, schema_json, f"{stage_name} attempt {attempt_idx + 1}")
            if isinstance(result, dict):
                duration = time.perf_counter() - start
                fields_returned = list(result.keys())
                res = ExtractionPassResult(
                    stage_name, True, result, attempt_idx + 1, errors, duration_seconds=duration
                )
                if self._on_pass_complete is not None:
                    self._emit_pass_telemetry(
                        res, stage_type, fields_requested, fields_returned, duration
                    )
                logger.info(
                    "[staged] pass=%s stage_type=%s success=true attempts=%s duration_seconds=%.2f fields_returned=%s",
                    stage_name,
                    stage_type,
                    attempt_idx + 1,
                    duration,
                    len(fields_returned),
                )
                return res
            errors.append(f"empty_or_invalid_response_attempt_{attempt_idx + 1}")
            logger.warning(
                "[staged] pass=%s stage_type=%s attempt=%s failed (empty or invalid JSON)",
                stage_name,
                stage_type,
                attempt_idx + 1,
            )
        duration = time.perf_counter() - start
        res = ExtractionPassResult(
            stage_name, False, {}, attempts, errors, duration_seconds=duration
        )
        if self._on_pass_complete is not None:
            self._emit_pass_telemetry(res, stage_type, fields_requested, [], duration)
        logger.warning(
            "[staged] pass=%s stage_type=%s success=false attempts=%s errors=%s",
            stage_name,
            stage_type,
            attempts,
            errors,
        )
        return res

    def extract(self, markdown: str, context: str) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        identity_fields_map = self._metadata.nested_entity_identity_fields

        # Pass 1: Skeleton
        if self._plan.skeleton_fields:
            skeleton_schema = prompts.build_root_subschema(self._schema_json, self._plan.skeleton_fields)
            skeleton_hints = {
                k: v
                for k, v in self._metadata.nested_edge_targets.items()
                if k in self._plan.skeleton_fields
            } or None
            skeleton_prompt = prompts.get_skeleton_prompt(
                markdown_content=markdown,
                schema_json=skeleton_schema,
                anchor_fields=self._plan.skeleton_fields,
                prior_extraction=merged if self._config.include_prior_context else None,
                nested_edge_hints=skeleton_hints,
            )
            result = self._run_pass(
                stage_name=f"{context} skeleton",
                stage_type="skeleton",
                prompt=skeleton_prompt,
                schema_json=skeleton_schema,
                fields_requested=self._plan.skeleton_fields,
            )
            if result.success:
                merge_pass_output(
                    merged,
                    result.data,
                    context_tag="staged:skeleton",
                    identity_fields_map=identity_fields_map,
                    policy=ReconciliationPolicy(),
                    merge_similarity_fallback=self._config.merge_similarity_fallback,
                )

        # Pass 2: Disjoint groups
        for idx, fields in enumerate(self._plan.groups):
            group_schema = prompts.build_root_subschema(self._schema_json, fields)
            group_hints = {
                k: v for k, v in self._metadata.nested_edge_targets.items() if k in fields
            } or None
            group_prompt = prompts.get_group_prompt(
                markdown_content=markdown,
                schema_json=group_schema,
                group_name=f"group_{idx}",
                focus_fields=fields,
                prior_extraction=merged if self._config.include_prior_context else None,
                critical_fields=self._plan.critical_fields,
                nested_edge_hints=group_hints,
            )
            result = self._run_pass(
                stage_name=f"{context} group_{idx}",
                stage_type="group",
                prompt=group_prompt,
                schema_json=group_schema,
                fields_requested=fields,
            )
            if result.success:
                merge_pass_output(
                    merged,
                    result.data,
                    context_tag=f"staged:group:{idx}",
                    identity_fields_map=identity_fields_map,
                    policy=ReconciliationPolicy(),
                    merge_similarity_fallback=self._config.merge_similarity_fallback,
                )

        # Pass 3: Quality-based targeted repair
        previous_report: prompts.QualityReport | None = None
        for repair_round in range(self._config.max_repair_rounds):
            report = prompts.assess_quality(
                candidate_data=merged,
                schema_json=self._schema_json,
                critical_fields=self._plan.critical_fields,
                max_depth=self._config.quality_depth,
            )
            fields_to_repair = report.root_fields()
            logger.info(
                "[staged] repair_round=%s quality_issues=%s fields_to_repair=%s",
                repair_round + 1,
                len(report.issues),
                fields_to_repair[:12] if len(fields_to_repair) > 12 else fields_to_repair,
            )
            if not fields_to_repair:
                break
            if previous_report is not None and not report.improved_over(previous_report):
                logger.info("[staged] repair quality stalled; stopping repair loop")
                break

            repair_schema = prompts.build_root_subschema(self._schema_json, fields_to_repair)
            nested_obligations = report.nested_path_obligations(self._metadata)
            repair_prompt = prompts.get_repair_prompt(
                markdown_content=markdown,
                schema_json=repair_schema,
                failed_fields=fields_to_repair,
                prior_extraction=merged if self._config.include_prior_context else None,
                issue_summary=", ".join(issue.reason for issue in report.issues[:8]),
                nested_obligations=nested_obligations if nested_obligations else None,
            )
            result = self._run_pass(
                stage_name=f"{context} repair",
                stage_type="repair",
                prompt=repair_prompt,
                schema_json=repair_schema,
                fields_requested=fields_to_repair,
            )
            if result.success:
                merge_pass_output(
                    merged,
                    result.data,
                    context_tag="staged:repair",
                    identity_fields_map=identity_fields_map,
                    policy=ReconciliationPolicy(repair_override_roots=set(fields_to_repair)),
                    merge_similarity_fallback=self._config.merge_similarity_fallback,
                )
            previous_report = report

        # Optional LLM consolidation when heuristic repair left issues
        if merged and self._run_llm_consolidation is not None:
            final_report = prompts.assess_quality(
                candidate_data=merged,
                schema_json=self._schema_json,
                critical_fields=self._plan.critical_fields,
                max_depth=self._config.quality_depth,
            )
            fields_with_issues = final_report.root_fields()
            if fields_with_issues:
                logger.info(
                    "[staged] triggering LLM consolidation for fields=%s",
                    fields_with_issues[:15] if len(fields_with_issues) > 15 else fields_with_issues,
                )
                consolidation_result = self._run_llm_consolidation(
                    merged, markdown, fields_with_issues, self._schema_json
                )
                if isinstance(consolidation_result, dict) and consolidation_result:
                    merge_pass_output(
                        merged,
                        consolidation_result,
                        context_tag="staged:llm_consolidation",
                        identity_fields_map=identity_fields_map,
                        policy=ReconciliationPolicy(repair_override_roots=set(fields_with_issues)),
                        merge_similarity_fallback=self._config.merge_similarity_fallback,
                    )
                    logger.info("[staged] LLM consolidation merged successfully")

        if merged and self._template is not None:
            materialize_nested_edges(merged, self._metadata)

        return merged or None
