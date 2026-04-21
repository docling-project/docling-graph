"""Unit tests for gleaning (second-pass extraction)."""

from __future__ import annotations

from typing import NoReturn

import pytest

from docling_graph.core.extractors.gleaning import (
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
