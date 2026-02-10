"""
Deterministic materialization of nested edge targets from sibling scalars.
"""

from __future__ import annotations

from typing import Any

from .prompts import TemplateGraphMetadata


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _scalar_for_identity(
    element: dict[str, Any],
    identity_field: str,
) -> Any:
    """Get value for identity field: same-name scalar first, then optional fallback (e.g. description -> name)."""
    if identity_field in element and element[identity_field] is not None:
        return element[identity_field]
    if identity_field == "name" and "description" in element and element["description"] is not None:
        return element["description"]
    return None


def materialize_nested_edges(
    data: dict[str, Any],
    metadata: TemplateGraphMetadata,
) -> None:
    """
    When a merged extraction has a null nested edge target but sibling scalars match
    the target's identity fields, synthesize a minimal nested object so the graph
    gets the node and edge. Mutates `data` in place. Domain-agnostic: driven only
    by metadata (nested_edge_targets).
    """
    if not metadata.nested_edge_targets:
        return
    for root_key, targets in metadata.nested_edge_targets.items():
        raw = data.get(root_key)
        if not isinstance(raw, list):
            continue
        for element in raw:
            if not isinstance(element, dict):
                continue
            for child_field, target_identity_fields in targets:
                if not _is_empty(element.get(child_field)):
                    continue
                minimal: dict[str, Any] = {}
                for id_field in target_identity_fields:
                    val = _scalar_for_identity(element, id_field)
                    if val is not None:
                        minimal[id_field] = val
                if minimal:
                    element[child_field] = minimal
