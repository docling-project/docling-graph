"""
Staged extraction contract.

Provides prompt and schema utilities for multi-pass extraction to improve
small-model quality while keeping deterministic reconciliation.
"""

from .prompts import (
    ExtractionFieldPlan,
    QualityIssue,
    QualityReport,
    TemplateGraphMetadata,
    assess_quality,
    build_root_subschema,
    detect_quality_issues,
    get_template_graph_metadata,
    get_group_prompt,
    plan_extraction_passes,
    get_root_field_groups,
    get_skeleton_fields,
    get_skeleton_prompt,
    get_repair_prompt,
    get_consolidation_prompt,
)

from .benchmark import (
    QualityDelta,
    RunMetrics,
    compare_metrics,
    load_run_metrics,
    summarize_comparison,
)
from .orchestrator import (
    ExtractionPassResult,
    StagedOrchestrator,
    StagedPassConfig,
)
from .reconciliation import ReconciliationPolicy, merge_pass_output

__all__ = [
    "ExtractionPassResult",
    "QualityDelta",
    "ReconciliationPolicy",
    "RunMetrics",
    "StagedOrchestrator",
    "StagedPassConfig",
    "ExtractionFieldPlan",
    "QualityIssue",
    "QualityReport",
    "TemplateGraphMetadata",
    "assess_quality",
    "build_root_subschema",
    "detect_quality_issues",
    "get_template_graph_metadata",
    "get_group_prompt",
    "plan_extraction_passes",
    "get_root_field_groups",
    "get_skeleton_fields",
    "get_skeleton_prompt",
    "get_repair_prompt",
    "get_consolidation_prompt",
    "compare_metrics",
    "load_run_metrics",
    "merge_pass_output",
    "summarize_comparison",
]

