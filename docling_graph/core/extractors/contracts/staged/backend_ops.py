"""Staged contract backend operations."""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel

from .orchestrator import CatalogOrchestrator, CatalogOrchestratorConfig

logger = logging.getLogger(__name__)


def _make_wrapped_llm(
    llm_call_fn: Any,
    staged_config_raw: dict[str, Any],
    _fill_structured_failed: list[bool],
    _diagnostics: dict[str, Any],
) -> Callable[..., Any]:
    """Build the wrapped LLM callable that injects max_tokens and diagnostics. Used by run_staged_orchestrator and tests."""

    def _wrapped_llm(
        prompt: Any,
        schema_json_arg: str,
        context_arg: str,
        *,
        response_top_level: str = "object",
        response_schema_name: str = "staged_extraction",
        **kwargs: Any,
    ) -> Any:
        if "catalog_id_pass" in context_arg:
            max_tok = staged_config_raw.get("id_max_tokens")
            call_max_tokens = int(max_tok) if max_tok is not None else None
        elif "fill_call_" in context_arg:
            max_tok = staged_config_raw.get("fill_max_tokens")
            call_max_tokens = int(max_tok) if max_tok is not None else None
        else:
            call_max_tokens = None
        structured_override: bool | None = None
        if "fill_call_" in context_arg and _fill_structured_failed[0]:
            structured_override = False
        result = llm_call_fn(
            prompt,
            schema_json_arg,
            context_arg,
            response_top_level=response_top_level,
            response_schema_name=response_schema_name,
            max_tokens=call_max_tokens,
            structured_output_override=structured_override,
            _diagnostics_out=_diagnostics,
        )
        if "fill_call_" in context_arg and _diagnostics.get("fallback_used"):
            _fill_structured_failed[0] = True
        return result

    return _wrapped_llm


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

    _fill_structured_failed: list[bool] = [False]
    _diagnostics: dict[str, Any] = {}
    _wrapped_llm = _make_wrapped_llm(
        llm_call_fn, staged_config_raw, _fill_structured_failed, _diagnostics
    )

    def _on_trace(trace_dict: dict) -> None:
        if trace_data is not None:
            trace_data.emit("staged_trace_emitted", "extraction", trace_dict)

    orchestrator = CatalogOrchestrator(
        llm_call_fn=_wrapped_llm,
        schema_json=schema_json,
        template=template,
        config=catalog_config,
        debug_dir=debug_dir or None,
        structured_output=structured_output,
        on_trace=_on_trace if trace_data is not None else None,
    )
    logger.info("[StagedExtraction] Starting catalog extraction (ID + fill + edges)")
    return orchestrator.extract(markdown=markdown, context=context)
