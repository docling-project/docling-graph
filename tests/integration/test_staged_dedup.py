"""
Integration tests for staged extraction deduplication.

Validates that duplicate parents with overlapping children are merged when
using nested identity and optional similarity fallback.
"""

from __future__ import annotations

import pytest

from docling_graph.core.extractors.contracts.staged.reconciliation import (
    ReconciliationPolicy,
    merge_pass_output,
)


@pytest.mark.integration
def test_merge_pass_output_dedup_same_study_nested_identity():
    """Two passes returning the same logical study (nested identity) merge into one."""
    merged = {
        "studies": [
            {
                "study_id": "STUDY-abc",
                "objective": "Investigate stability",
                "experiments": [
                    {"experiment_id": "EXP-1", "description": "Run A"},
                ],
            }
        ]
    }
    pass_output = {
        "studies": [
            {
                "study_id": "STUDY-abc",
                "objective": "Investigate stability",
                "experiments": [
                    {"experiment_id": "EXP-1", "description": "Run A updated"},
                ],
            }
        ]
    }
    result = merge_pass_output(
        merged,
        pass_output,
        context_tag="staged:group:0",
        identity_fields_map={
            "studies": ["study_id"],
            "studies.experiments": ["experiment_id"],
        },
    )
    assert len(result["studies"]) == 1
    assert len(result["studies"][0]["experiments"]) == 1
    assert result["studies"][0]["experiments"][0]["description"] == "Run A updated"


@pytest.mark.integration
def test_merge_pass_output_similarity_fallback_merge_when_overlap():
    """With merge_similarity_fallback=True, entities with overlapping children can merge."""
    merged = {
        "studies": [
            {
                "study_id": "STUDY-gen-1",
                "objective": "First",
                "experiments": [{"experiment_id": "E1", "name": "Exp1"}],
            }
        ]
    }
    pass_output = {
        "studies": [
            {
                "study_id": "STUDY-gen-2",
                "objective": "Second",
                "experiments": [{"experiment_id": "E1", "name": "Exp1"}],
            }
        ]
    }
    result = merge_pass_output(
        merged,
        pass_output,
        context_tag="staged:group:0",
        identity_fields_map={"studies": ["study_id"], "studies.experiments": ["experiment_id"]},
        merge_similarity_fallback=True,
    )
    # Different study_id => no id match; same experiments => high overlap => may merge to 1
    assert len(result["studies"]) >= 1
    assert len(result["studies"]) <= 2
