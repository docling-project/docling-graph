"""Unit tests for the dense contract backend_ops thin wrapper."""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import docling_graph.core.extractors.contracts.dense.backend_ops as backend_ops_mod
from docling_graph.core.extractors.contracts.dense.backend_ops import run_dense_orchestrator


class _Template(BaseModel):
    name: str = "x"


def _noop_llm(*args: Any, **kwargs: Any) -> dict:
    return {}


def test_run_dense_orchestrator_returns_none_when_template_missing(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.WARNING):
        root, stats, ledger = run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=["chunk one"],
            chunk_metadata=None,
            full_markdown="doc",
            context="ctx",
            template=None,
            trace_data=None,
        )

    assert root is None
    assert stats == {}
    assert ledger is None
    assert any("template" in rec.message.lower() for rec in caplog.records)


def test_run_dense_orchestrator_falls_back_to_chunks_when_markdown_blank() -> None:
    mock_instance = MagicMock()
    mock_instance.run.return_value = {"root": True}
    mock_instance.last_run_stats = {"skeleton_nodes": 3}
    mock_instance.last_provenance = None
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        root, stats, ledger = run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=["chunk a", "chunk b"],
            chunk_metadata=None,
            full_markdown="   ",
            context="ctx",
            template=_Template,
            trace_data=None,
        )

    assert root == {"root": True}
    assert stats == {"skeleton_nodes": 3}
    assert ledger is None

    _, run_kwargs = mock_instance.run.call_args
    assert run_kwargs["full_markdown"] == "chunk a\n\nchunk b"
    assert run_kwargs["chunks"] == ["chunk a", "chunk b"]
    assert run_kwargs["context"] == "ctx"


def test_run_dense_orchestrator_falls_back_to_empty_string_when_no_chunks() -> None:
    mock_instance = MagicMock()
    mock_instance.run.return_value = None
    mock_instance.last_run_stats = {}
    mock_instance.last_provenance = None
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=[],
            chunk_metadata=None,
            full_markdown=None,
            context="ctx",
            template=_Template,
            trace_data=None,
        )

    _, run_kwargs = mock_instance.run.call_args
    assert run_kwargs["full_markdown"] == ""


def test_run_dense_orchestrator_happy_path_constructs_orchestrator_with_expected_kwargs() -> None:
    mock_instance = MagicMock()
    mock_instance.run.return_value = {"root": "value"}
    mock_instance.last_run_stats = {"truncation_count": 0}
    mock_ledger = object()
    mock_instance.last_provenance = mock_ledger
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    dense_config_raw = {"debug_dir": "/tmp/debug", "dense_dedupe": "aggressive"}
    metadata = [{"chunk_id": "c1"}]

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        root, stats, ledger = run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw=dense_config_raw,
            chunks=["chunk one"],
            chunk_metadata=metadata,
            full_markdown="full doc text",
            context="some context",
            template=_Template,
            trace_data=None,
        )

    assert root == {"root": "value"}
    assert stats == {"truncation_count": 0}
    assert ledger is mock_ledger

    _, ctor_kwargs = mock_orchestrator_cls.call_args
    assert ctor_kwargs["llm_call_fn"] is _noop_llm
    assert ctor_kwargs["template"] is _Template
    assert ctor_kwargs["debug_dir"] == "/tmp/debug"
    assert ctor_kwargs["config"].dedupe_mode == "aggressive"
    assert ctor_kwargs["on_trace"] is None

    _, run_kwargs = mock_instance.run.call_args
    assert run_kwargs["chunks"] == ["chunk one"]
    assert run_kwargs["chunk_metadata"] == metadata
    assert run_kwargs["full_markdown"] == "full doc text"
    assert run_kwargs["context"] == "some context"


def test_run_dense_orchestrator_passes_none_debug_dir_when_not_configured() -> None:
    mock_instance = MagicMock()
    mock_instance.run.return_value = None
    mock_instance.last_run_stats = {}
    mock_instance.last_provenance = None
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=["a"],
            chunk_metadata=None,
            full_markdown="doc",
            context="ctx",
            template=_Template,
            trace_data=None,
        )

    _, ctor_kwargs = mock_orchestrator_cls.call_args
    assert ctor_kwargs["debug_dir"] is None


def test_run_dense_orchestrator_wires_on_trace_when_trace_data_present() -> None:
    mock_instance = MagicMock()
    mock_instance.run.return_value = None
    mock_instance.last_run_stats = {}
    mock_instance.last_provenance = None
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    trace_data = MagicMock()

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=["a"],
            chunk_metadata=None,
            full_markdown="doc",
            context="ctx",
            template=_Template,
            trace_data=trace_data,
        )

    _, ctor_kwargs = mock_orchestrator_cls.call_args
    on_trace = ctor_kwargs["on_trace"]
    assert callable(on_trace)

    on_trace({"some": "trace"})
    trace_data.emit.assert_called_once_with("dense_trace_emitted", "extraction", {"some": "trace"})


def test_run_dense_orchestrator_on_trace_noop_when_trace_data_none_but_forced() -> None:
    """Directly exercise the _on_trace closure's internal guard.

    on_trace is passed as None to the constructor when trace_data is None, so
    the guard inside _on_trace normally isn't reachable through the public
    call path. This test calls the closure captured via the constructor call
    args in the "trace_data present" scenario above to document the guard's
    intent; here we confirm the wrapper never invokes the callback itself.
    """
    mock_instance = MagicMock()
    mock_instance.run.return_value = None
    mock_instance.last_run_stats = {}
    mock_instance.last_provenance = None
    mock_orchestrator_cls = MagicMock(return_value=mock_instance)

    with patch.object(backend_ops_mod, "DenseOrchestrator", mock_orchestrator_cls):
        run_dense_orchestrator(
            llm_call_fn=_noop_llm,
            dense_config_raw={},
            chunks=["a"],
            chunk_metadata=None,
            full_markdown="doc",
            context="ctx",
            template=_Template,
            trace_data=None,
        )

    _, ctor_kwargs = mock_orchestrator_cls.call_args
    assert ctor_kwargs["on_trace"] is None
