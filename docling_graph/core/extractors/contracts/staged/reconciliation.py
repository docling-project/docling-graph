"""
Deterministic reconciliation policies for staged extraction passes.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ....utils.dict_merger import deep_merge_dicts

_PLACEHOLDER_VALUE_PATTERNS = (
    r"\.\.\.$",
    r"^sample\b",
    r"\bexample\b",
    r"^value$",
    r"^unknown$",
    r"^n/?a$",
    r"^tbd$",
    r"^placeholder$",
    r"^null$",
    r"^none$",
    r"^-$",
)


@dataclass
class ReconciliationPolicy:
    """
    Configurable merge precedence for staged pass outputs.
    """

    prefer_non_placeholder: bool = True
    repair_override_roots: set[str] = field(default_factory=set)


def _is_placeholder_string(value: str) -> bool:
    compact = value.strip().lower()
    return any(re.search(pattern, compact) for pattern in _PLACEHOLDER_VALUE_PATTERNS)


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _is_better_scalar(source_value: Any, target_value: Any, prefer_non_placeholder: bool) -> bool:
    if _is_empty(source_value):
        return False
    if _is_empty(target_value):
        return True
    if isinstance(source_value, str) and isinstance(target_value, str) and prefer_non_placeholder:
        if _is_placeholder_string(target_value) and not _is_placeholder_string(source_value):
            return True
        if len(source_value.strip()) > len(target_value.strip()) and _is_placeholder_string(target_value):
            return True
    return False


def merge_pass_output(
    merged: dict[str, Any],
    pass_output: dict[str, Any],
    *,
    context_tag: str,
    identity_fields_map: dict[str, list[str]] | None = None,
    policy: ReconciliationPolicy | None = None,
    merge_similarity_fallback: bool = False,
) -> dict[str, Any]:
    """
    Merge pass output into cumulative model using deterministic policy.
    """
    policy = policy or ReconciliationPolicy()
    if not pass_output:
        return merged

    # Start with a deep merge for nested/list behavior and identity-aware list dedupe.
    deep_merge_dicts(
        merged,
        copy.deepcopy(pass_output),
        context_tag=context_tag,
        identity_fields_map=identity_fields_map,
        override_roots=policy.repair_override_roots or None,
        merge_similarity_fallback=merge_similarity_fallback,
    )

    # Improve scalar conflicts when both values exist.
    for key, source_value in pass_output.items():
        if key not in merged:
            continue
        target_value = merged[key]
        if isinstance(source_value, dict) or isinstance(source_value, list):
            continue
        if _is_better_scalar(source_value, target_value, policy.prefer_non_placeholder):
            merged[key] = copy.deepcopy(source_value)

    return merged

