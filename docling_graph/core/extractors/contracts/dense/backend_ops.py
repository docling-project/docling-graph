"""Dense contract backend operations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .orchestrator import DenseOrchestrator, DenseOrchestratorConfig

logger = logging.getLogger(__name__)


def run_dense_orchestrator(
    *,
    llm_call_fn: Any,
    staged_config_raw: dict[str, Any],
    chunks: list[str],
    chunk_metadata: list[dict[str, Any]] | None,
    full_markdown: str | None,
    context: str,
    template: type[BaseModel] | None,
    trace_data: Any,
) -> dict | list | None:
    """Run Dense orchestrator (Phase 1 skeleton + Phase 2 fill)."""
    if template is None:
        logger.warning("Dense extraction requires a template; skipping.")
        return None
    if full_markdown is None or not full_markdown.strip():
        full_markdown = "\n\n".join(chunks) if chunks else ""
    debug_dir = staged_config_raw.get("debug_dir") or ""
    config = DenseOrchestratorConfig.from_dict(staged_config_raw)

    def _on_trace(trace_dict: dict) -> None:
        if trace_data is not None:
            trace_data.emit("dense_trace_emitted", "extraction", trace_dict)

    orchestrator = DenseOrchestrator(
        llm_call_fn=llm_call_fn,
        template=template,
        config=config,
        debug_dir=debug_dir or None,
        on_trace=_on_trace if trace_data is not None else None,
    )
    logger.info("[DenseExtraction] Starting dense extraction (skeleton + fill)")
    return orchestrator.run(
        chunks=chunks,
        chunk_metadata=chunk_metadata,
        full_markdown=full_markdown,
        context=context,
    )
