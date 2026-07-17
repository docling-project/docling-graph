"""PipelineConfig.gc_collect gates the per-run gc.collect() in _cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec, patch

from docling_graph.config import PipelineConfig
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.orchestrator import PipelineOrchestrator


def _cleanup_with(gc_collect: bool) -> bool:
    """Run _cleanup with a bare context; return whether gc.collect was called."""
    orch = PipelineOrchestrator(PipelineConfig(gc_collect=gc_collect), mode="api")
    ctx = PipelineContext(config=orch.config)
    ctx.extractor = None  # nothing to clean up beyond gc
    with patch("docling_graph.pipeline.orchestrator.gc.collect") as gc_collect_mock:
        orch._cleanup(ctx)
        return gc_collect_mock.called


def test_gc_collect_true_calls_gc():
    assert _cleanup_with(gc_collect=True) is True


def test_gc_collect_false_skips_gc():
    assert _cleanup_with(gc_collect=False) is False


def test_gc_collect_defaults_true():
    assert PipelineConfig().gc_collect is True


def _extractor_cleanup_collect_args(gc_collect: bool) -> tuple[bool, bool]:
    """Run _cleanup with a mock extractor; return the collect= each cleanup got."""
    orch = PipelineOrchestrator(PipelineConfig(gc_collect=gc_collect), mode="api")
    ctx = PipelineContext(config=orch.config)

    def _cleanup_sig(collect: bool = True) -> None: ...

    # create_autospec preserves the signature, so the orchestrator's
    # inspect-based forwarding sees the collect parameter.
    backend = MagicMock()
    backend.cleanup = create_autospec(_cleanup_sig)
    doc_processor = MagicMock()
    doc_processor.cleanup = create_autospec(_cleanup_sig)
    ctx.extractor = MagicMock()
    ctx.extractor.backend = backend
    ctx.extractor.doc_processor = doc_processor

    with patch("docling_graph.pipeline.orchestrator.gc.collect"):
        orch._cleanup(ctx)
    return (
        backend.cleanup.call_args.kwargs.get("collect"),
        doc_processor.cleanup.call_args.kwargs.get("collect"),
    )


def test_gc_collect_flag_reaches_backend_and_doc_processor_cleanups():
    """gc_collect=False must gate the backends' internal gc.collect() calls too,
    or a long-lived service still pays the stop-the-world pause every run."""
    assert _extractor_cleanup_collect_args(gc_collect=False) == (False, False)
    assert _extractor_cleanup_collect_args(gc_collect=True) == (True, True)
