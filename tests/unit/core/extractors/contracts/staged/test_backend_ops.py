"""Unit tests for staged contract backend_ops (run_staged_orchestrator)."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from docling_graph.core.extractors.contracts.staged.backend_ops import (
    _make_wrapped_llm,
    run_staged_orchestrator,
)
from tests.fixtures.sample_templates.test_template import SampleCompany, SampleInvoice


def test_make_wrapped_llm_else_branch_context_without_id_or_fill():
    """When context has neither 'catalog_id_pass' nor 'fill_call_', call_max_tokens is None."""
    seen_kwargs: list[dict] = []

    def record_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        **kwargs: Any,
    ) -> dict:
        seen_kwargs.append({"context": context_arg, **kwargs})
        return {"nodes": []}

    _fill_structured_failed = [False]
    _diagnostics: dict = {}
    wrapped = _make_wrapped_llm(record_llm, {}, _fill_structured_failed, _diagnostics)
    wrapped(None, "{}", "other_context")
    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["max_tokens"] is None
    assert seen_kwargs[0]["context"] == "other_context"


def test_run_staged_orchestrator_returns_none_when_template_is_none(caplog):
    """When template is None, return None and log warning."""
    result = run_staged_orchestrator(
        llm_call_fn=MagicMock(),
        staged_config_raw={},
        markdown="x",
        schema_json="{}",
        context="doc",
        template=None,
        trace_data=None,
        structured_output=False,
    )
    assert result is None
    assert "template" in caplog.text.lower() or "skipping" in caplog.text.lower()


def test_run_staged_orchestrator_runs_orchestrator_and_returns_merged():
    """With template and mock LLM returning valid ID + fill, returns merged root dict."""
    call_log: list[tuple[str, dict]] = []

    def mock_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        *,
        response_top_level: str = "object",
        response_schema_name: str = "staged_extraction",
        **kwargs: Any,
    ) -> dict | list | None:
        call_log.append((context_arg, kwargs))
        if "catalog_id_pass" in context_arg:
            return {
                "nodes": [
                    {"path": "", "ids": {"invoice_number": "INV-1"}, "parent": None},
                ],
            }
        if "fill_call_" in context_arg:
            return [
                {
                    "invoice_number": "INV-1",
                    "date": "2024-01-01",
                    "total_amount": 10.0,
                    "vendor_name": "V",
                    "items": [],
                }
            ]
        return None

    result = run_staged_orchestrator(
        llm_call_fn=mock_llm,
        staged_config_raw={"debug_dir": ""},
        markdown="Invoice INV-1...",
        schema_json='{"type":"object"}',
        context="test",
        template=SampleInvoice,
        trace_data=None,
        structured_output=False,
    )
    assert result is not None
    assert isinstance(result, dict)
    assert result.get("invoice_number") == "INV-1"
    assert len(call_log) >= 2
    assert any("catalog_id_pass" in c[0] for c in call_log)
    assert any("fill_call_" in c[0] for c in call_log)


def test_run_staged_orchestrator_passes_id_max_tokens_and_fill_max_tokens():
    """Wrapper passes id_max_tokens / fill_max_tokens from config into LLM calls."""
    call_log: list[dict] = []

    def mock_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        *,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict | list | None:
        call_log.append({"context": context_arg, "max_tokens": max_tokens})
        if "catalog_id_pass" in context_arg:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "I1"}, "parent": None}]}
        if "fill_call_" in context_arg:
            return [
                {
                    "invoice_number": "I1",
                    "date": "",
                    "total_amount": 0,
                    "vendor_name": "",
                    "items": [],
                }
            ]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        result = run_staged_orchestrator(
            llm_call_fn=mock_llm,
            staged_config_raw={
                "debug_dir": tmp,
                "id_max_tokens": 4096,
                "fill_max_tokens": 8192,
            },
            markdown="x",
            schema_json='{"type":"object"}',
            context="test",
            template=SampleInvoice,
            trace_data=None,
            structured_output=False,
        )
    assert result is not None
    id_calls = [c for c in call_log if "catalog_id_pass" in c["context"]]
    fill_calls = [c for c in call_log if "fill_call_" in c["context"]]
    assert len(id_calls) >= 1
    assert len(fill_calls) >= 1
    assert id_calls[0]["max_tokens"] == 4096
    assert fill_calls[0]["max_tokens"] == 8192


def test_run_staged_orchestrator_trace_emit_called_when_trace_data_provided():
    """When trace_data is provided, on_trace is set and trace emit is called."""
    trace_emit = MagicMock()
    trace_data = MagicMock()
    trace_data.emit = trace_emit

    def mock_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        **kwargs: Any,
    ) -> dict | list | None:
        if "catalog_id_pass" in context_arg:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "T1"}, "parent": None}]}
        if "fill_call_" in context_arg:
            return [
                {
                    "invoice_number": "T1",
                    "date": "",
                    "total_amount": 0,
                    "vendor_name": "",
                    "items": [],
                }
            ]
        return None

    result = run_staged_orchestrator(
        llm_call_fn=mock_llm,
        staged_config_raw={},
        markdown="x",
        schema_json='{"type":"object"}',
        context="test",
        template=SampleInvoice,
        trace_data=trace_data,
        structured_output=False,
    )
    assert result is not None
    trace_emit.assert_called_once()
    (event, _source, payload) = trace_emit.call_args[0]
    assert event == "staged_trace_emitted"
    assert "timings_seconds" in payload or "per_path_counts" in payload


def test_run_staged_orchestrator_structured_fallback_after_diagnostics():
    """When _diagnostics_out['fallback_used'] is set on first fill call, next fill gets structured_output_override=False."""
    fill_calls: list[dict] = []

    def mock_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        *,
        structured_output_override: bool | None = None,
        _diagnostics_out: dict | None = None,
        **kwargs: Any,
    ) -> dict | list | None:
        if "catalog_id_pass" in context_arg:
            return {
                "nodes": [
                    {"path": "", "ids": {"company_name": "Acme"}, "parent": None},
                    {
                        "path": "employees[]",
                        "ids": {"email": "e@a.com"},
                        "parent": {"path": "", "ids": {"company_name": "Acme"}},
                    },
                ],
            }
        if "fill_call_" in context_arg:
            fill_calls.append({"structured_output_override": structured_output_override})
            if _diagnostics_out is not None and len(fill_calls) == 1:
                _diagnostics_out["fallback_used"] = True
            if len(fill_calls) == 1:
                return [{"email": "e@a.com", "first_name": "F", "last_name": "L", "age": None}]
            return [{"company_name": "Acme", "industry": "", "founded_year": 0, "employees": []}]
        return None

    result = run_staged_orchestrator(
        llm_call_fn=mock_llm,
        staged_config_raw={},
        markdown="x",
        schema_json='{"type":"object"}',
        context="test",
        template=SampleCompany,
        trace_data=None,
        structured_output=False,
    )
    assert result is not None
    assert len(fill_calls) >= 2
    assert fill_calls[0]["structured_output_override"] is None
    assert fill_calls[1]["structured_output_override"] is False


def test_run_staged_orchestrator_debug_dir_used():
    """debug_dir from config is passed to orchestrator and artifacts are written."""

    def mock_llm(
        prompt: Any,
        schema_json_arg: Any,
        context_arg: Any,
        **kwargs: Any,
    ) -> dict | list | None:
        if "catalog_id_pass" in context_arg:
            return {"nodes": [{"path": "", "ids": {"invoice_number": "D1"}, "parent": None}]}
        if "fill_call_" in context_arg:
            return [
                {
                    "invoice_number": "D1",
                    "date": "",
                    "total_amount": 0,
                    "vendor_name": "",
                    "items": [],
                }
            ]
        return None

    with tempfile.TemporaryDirectory() as tmp:
        result = run_staged_orchestrator(
            llm_call_fn=mock_llm,
            staged_config_raw={"debug_dir": tmp},
            markdown="x",
            schema_json='{"type":"object"}',
            context="test",
            template=SampleInvoice,
            trace_data=None,
            structured_output=False,
        )
        assert result is not None
        assert os.path.isdir(tmp)
        files = os.listdir(tmp)
        assert "staged_trace.json" in files or "merged_output.json" in files
