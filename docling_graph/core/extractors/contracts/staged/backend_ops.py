"""Staged contract backend operations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .orchestrator import CatalogOrchestrator, CatalogOrchestratorConfig

logger = logging.getLogger(__name__)


def run_staged_orchestrator(
    *,
    llm_call_fn: Any,
    staged_config_raw: dict[str, Any],
    markdown: str,
    schema_json: str,
    context: str,
    template: type[BaseModel] | None,
    trace_data: Any,
    structured_output: bool,
) -> dict | list | None:
    """Run staged orchestrator from contract-local module."""
    if template is None:
        logger.warning("Staged extraction requires a template; skipping.")
        return None
    debug_dir = staged_config_raw.get("debug_dir") or ""
    catalog_config = CatalogOrchestratorConfig.from_dict(staged_config_raw)

    def _on_trace(trace_dict: dict) -> None:
        if trace_data is not None:
            trace_data.emit("staged_trace_emitted", "extraction", trace_dict)

    orchestrator = CatalogOrchestrator(
        llm_call_fn=llm_call_fn,
        schema_json=schema_json,
        template=template,
        config=catalog_config,
        debug_dir=debug_dir or None,
        structured_output=structured_output,
        on_trace=_on_trace if trace_data is not None else None,
    )
    logger.info("[StagedExtraction] Starting catalog extraction (ID + fill + edges)")
    return orchestrator.extract(markdown=markdown, context=context)
