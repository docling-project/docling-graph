"""Delta contract backend operations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .orchestrator import DeltaOrchestrator, DeltaOrchestratorConfig

logger = logging.getLogger(__name__)


def run_delta_orchestrator(
    *,
    llm_call_fn: Any,
    staged_config_raw: dict[str, Any],
    chunks: list[str],
    chunk_metadata: list[dict[str, Any]] | None,
    context: str,
    template: type[BaseModel] | None,
    trace_data: Any,
    structured_output: bool,
) -> dict | list | None:
    """Run Delta orchestrator from contract-local module."""
    if template is None:
        logger.warning("Delta extraction requires a template; skipping.")
        return None
    debug_dir = staged_config_raw.get("debug_dir") or ""
    delta_config = DeltaOrchestratorConfig.from_dict(staged_config_raw)

    def _on_trace(trace_dict: dict) -> None:
        if trace_data is not None:
            trace_data.emit("delta_trace_emitted", "extraction", trace_dict)

    orchestrator = DeltaOrchestrator(
        llm_call_fn=llm_call_fn,
        template=template,
        config=delta_config,
        debug_dir=debug_dir or None,
        structured_output=structured_output,
        on_trace=_on_trace if trace_data is not None else None,
    )
    logger.info("[DeltaExtraction] Starting delta extraction (chunk batch -> merge -> project)")
    return orchestrator.extract(chunks=chunks, chunk_metadata=chunk_metadata, context=context)
