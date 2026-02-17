"""Unit tests for gleaning (second-pass extraction)."""

from __future__ import annotations

from typing import NoReturn

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
    def fail(_) -> NoReturn:
        raise ValueError("mock fail")

    out = run_gleaning_pass_direct("doc", {"x": 1}, "{}", fail)
    assert out is None


def test_run_gleaning_pass_direct_logs_warning_on_exception(caplog):
    """Exception in llm_call_fn triggers except block and logger.warning (lines 86-88)."""

    def fail(_) -> NoReturn:
        raise RuntimeError("gleaning error")

    with caplog.at_level("WARNING"):
        out = run_gleaning_pass_direct("doc", {}, "{}", fail)
    assert out is None
    assert "Gleaning pass failed" in caplog.text
    assert "gleaning error" in caplog.text


def test_run_gleaning_pass_direct_returns_dict_when_llm_returns_dict():
    def ok(_: object) -> dict:
        return {"extra": "value"}

    out = run_gleaning_pass_direct("doc", {"x": 1}, "{}", ok)
    assert out == {"extra": "value"}


def test_get_gleaning_prompt_direct_truncates_large_existing():
    """When existing_result serializes to > 8000 chars, prompt truncates with hint."""
    large = {"key": "x" * 9000}
    prompt = get_gleaning_prompt_direct(
        markdown="Doc",
        existing_result=large,
        schema_json="{}",
    )
    assert "... (truncated)" in prompt["user"]


def test_merge_gleaned_direct_custom_merge_options():
    """merge_gleaned_direct accepts custom description_merge_fields and description_merge_max_length."""
    existing = {"title": "A", "summary": "First."}
    extra = {"summary": "Second."}
    merged = merge_gleaned_direct(
        existing,
        extra,
        description_merge_fields=frozenset({"summary"}),
        description_merge_max_length=100,
    )
    assert merged["title"] == "A"
    assert "First." in merged["summary"] and "Second." in merged["summary"]


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


def test_build_already_found_summary_delta_uses_source_id_target_id_and_edge_label():
    """Summary includes rels that use source_id/target_id and edge_label keys."""
    graph = {
        "nodes": [{"path": "items[]", "ids": {"id": "1"}, "properties": {}}],
        "relationships": [
            {
                "source_id": "src-1",
                "target_id": "tgt-2",
                "edge_label": "LINKS",
            },
        ],
    }
    summary = build_already_found_summary_delta(graph, max_nodes=5, max_rels=5)
    assert "src-1" in summary and "tgt-2" in summary
    assert "LINKS" in summary


def test_build_already_found_summary_delta_skips_non_dict_node():
    """Non-dict entries in nodes are skipped."""
    graph = {
        "nodes": ["not a dict", {"path": "p", "ids": {}, "properties": {}}],
        "relationships": [],
    }
    summary = build_already_found_summary_delta(graph)
    assert "path=p" in summary
    assert "not a dict" not in summary


def test_build_already_found_summary_delta_skips_non_dict_relationship():
    """Non-dict entries in relationships are skipped (if not isinstance(r, dict): continue)."""
    graph = {
        "nodes": [{"path": "p", "ids": {}, "properties": {}}],
        "relationships": [
            {"source_key": "a", "target_key": "b", "label": "L"},
            "not a dict",
            None,
        ],
    }
    summary = build_already_found_summary_delta(graph, max_rels=10)
    assert "a" in summary and "b" in summary and "L" in summary
    assert "not a dict" not in summary
