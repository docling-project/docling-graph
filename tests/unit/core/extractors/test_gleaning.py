"""Unit tests for gleaning (second-pass extraction)."""

from __future__ import annotations

import pytest

from docling_graph.core.extractors.gleaning import (
    build_already_found_summary_delta,
    get_gleaning_prompt_direct,
    merge_gleaned_direct,
    run_gleaning_pass_direct,
)


def test_get_gleaning_prompt_direct():
    prompt = get_gleaning_prompt_direct(
        markdown="Doc text",
        existing_result={"name": "Acme", "value": 1},
        schema_json='{"type": "object"}',
    )
    assert "system" in prompt and "user" in prompt
    assert "ALREADY EXTRACTED" in prompt["user"]
    assert "Acme" in prompt["user"]
    assert "Doc text" in prompt["user"]


def test_merge_gleaned_direct():
    existing = {"a": 1, "description": "First."}
    extra = {"b": 2, "description": "Second."}
    merged = merge_gleaned_direct(existing, extra)
    assert merged["a"] == 1
    assert merged["b"] == 2
    assert "First." in merged["description"] and "Second." in merged["description"]


def test_run_gleaning_pass_direct_returns_none_on_failure():
    def fail(_):
        raise ValueError("mock fail")
    out = run_gleaning_pass_direct("doc", {"x": 1}, "{}", fail)
    assert out is None


def test_run_gleaning_pass_direct_returns_dict_when_llm_returns_dict():
    def ok(_):
        return {"extra": "value"}
    out = run_gleaning_pass_direct("doc", {"x": 1}, "{}", ok)
    assert out == {"extra": "value"}


def test_build_already_found_summary_delta():
    graph = {
        "nodes": [
            {"path": "p", "ids": {"name": "X"}, "properties": {"description": "D1"}},
        ],
        "relationships": [
            {"source_key": "a", "target_key": "b", "label": "L"},
        ],
    }
    summary = build_already_found_summary_delta(graph, max_nodes=10, max_rels=10)
    assert "path=p" in summary
    assert "X" in summary
    assert "a" in summary and "b" in summary
