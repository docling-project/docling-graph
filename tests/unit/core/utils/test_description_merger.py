"""Unit tests for description merger."""

from typing import NoReturn

import pytest

from docling_graph.core.utils.description_merger import (
    merge_descriptions,
    truncate_at_sentence_boundary,
)


def test_truncate_short():
    assert truncate_at_sentence_boundary("Short text.", 100) == "Short text."


def test_truncate_at_sentence():
    long_text = "First sentence. Second sentence. Third sentence."
    out = truncate_at_sentence_boundary(long_text, 30)
    assert out.endswith(".")
    assert len(out) <= 30
    assert "First sentence." in out


def test_truncate_empty():
    assert truncate_at_sentence_boundary("", 100) == ""
    assert truncate_at_sentence_boundary("x", 0) == ""


def test_merge_empty_existing():
    assert merge_descriptions("", "New text.", 1000) == "New text."


def test_merge_empty_new():
    assert merge_descriptions("Existing.", "", 1000) == "Existing."


def test_merge_duplicate_sentence_not_added():
    result = merge_descriptions("First sentence.", "First sentence.", 1000)
    assert result == "First sentence."


def test_merge_adds_new_sentence():
    result = merge_descriptions("First sentence.", "Second sentence.", 1000)
    assert "First sentence" in result
    assert "Second sentence" in result


def test_merge_truncates():
    a = "A. " * 200
    b = "B. " * 200
    result = merge_descriptions(a, b, 50)
    assert len(result) <= 50
    assert result.endswith(".") or result == ""


def test_merge_new_contained_in_existing():
    result = merge_descriptions("Long existing with bit.", "with bit.", 1000)
    assert result == "Long existing with bit."


def test_merge_with_summarizer_when_above_threshold():
    def summarizer(existing: str, new_list: list) -> str:
        return "Summarized: " + existing[:10] + " + " + str(len(new_list)) + " new."

    result = merge_descriptions(
        "First part. " * 200,
        "Second part. " * 200,
        max_length=5000,
        summarizer=summarizer,
        summarizer_min_total_length=100,
    )
    assert "Summarized:" in result


def test_merge_summarizer_below_threshold_uses_sentence_dedup():
    calls = []

    def summarizer(existing: str, new_list: list) -> str:
        calls.append(1)
        return "Only if used."

    result = merge_descriptions(
        "Short.",
        "Also short.",
        max_length=5000,
        summarizer=summarizer,
        summarizer_min_total_length=10_000,
    )
    assert len(calls) == 0
    assert "Short" in result and "Also short" in result


def test_merge_summarizer_failure_fallback():
    def summarizer(_e, _n) -> NoReturn:
        raise ValueError("mock failure")

    result = merge_descriptions(
        "A. " * 300,
        "B. " * 300,
        max_length=5000,
        summarizer=summarizer,
        summarizer_min_total_length=100,
    )
    assert "A." in result
