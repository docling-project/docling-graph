"""Public constructor for the template-induction ``LlmCallFn``.

``induce_spec_from_documents`` / ``generate_template(kind="docs")`` take an
injected ``llm_call_fn`` (the ``templategen.induce.documents`` contract). Wiring
one to a real LiteLLM client is non-trivial — it owns thread-local clients
(induction fans calls out across ``templategen.workers``), a one-shot
truncation-escalation retry with a hard 2x ceiling, and provider-safe schema-name
sanitization. This module exposes that wiring as a public, CLI-free API so
downstream callers (services, notebooks) do not reimplement it or reach into the
CLI internals.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any

from ..logging_utils import get_component_logger

if TYPE_CHECKING:
    from ..llm_clients.config import LlmRuntimeOverrides
    from .induce.documents import LlmCallFn

logger = get_component_logger("TemplateInduction", __name__)

__all__ = ["build_llm_call_fn"]


def build_llm_call_fn(
    provider: str,
    model: str,
    *,
    llm_overrides: LlmRuntimeOverrides | dict[str, Any] | None = None,
    structured_output: bool = True,
) -> LlmCallFn:
    """Bind a thread-safe ``llm_call_fn`` to a LiteLLM client.

    The returned callable follows the ``templategen.induce.documents`` contract
    (``llm_call_fn(*, prompt, schema_json, context) -> Any``) and owns truncation
    handling: one retry with escalated ``max_tokens`` (doubled, capped at half the
    model's context window). The retry fires whether the truncated response was
    salvaged into JSON or failed to parse outright — the unrecoverable case
    surfaces as a :class:`~docling_graph.exceptions.ClientError` and needs the
    escalation the most. The retry's result is returned (or its error raised)
    either way.

    ``structured_output=False`` (the one-shot induction strategy) requests plain
    ``json_object`` decoding instead of a schema grammar — small models that
    degenerate under guided decoding answer normally, and the response handler's
    fence-stripping/repair absorbs the slack.

    Thread-safe by construction: induction runs calls concurrently
    (``templategen.workers``), and one shared client would race on
    ``last_call_diagnostics`` and the escalation's ``max_tokens`` swap — so every
    thread gets its own client instance (the one built eagerly here doubles as a
    fail-fast config/dependency check and the building thread's client).

    Args:
        provider: Provider id (``mistral``, ``openai``, ``ollama``, ``vllm``, ...).
        model: Model name/path for that provider.
        llm_overrides: Optional runtime overrides (``LlmRuntimeOverrides`` or the
            equivalent dict) forwarded to ``resolve_effective_model_config``.
        structured_output: ``True`` (default, three-pass strategy) uses a schema
            grammar; ``False`` (one-shot strategy) requests ``json_object``.

    Returns:
        The bound ``llm_call_fn``.

    Raises:
        ConfigurationError: The ``litellm`` package is not installed (from eager
            client construction). Unknown providers do NOT raise — they fall back
            to generic defaults with a warning — and credentials are not checked
            here either: a bad key only surfaces as a ``ClientError`` from the
            returned callable's first real call.
    """
    import threading

    from ..exceptions import ClientError
    from ..llm_clients import get_client
    from ..llm_clients.config import resolve_effective_model_config

    # resolve_effective_model_config accepts LlmRuntimeOverrides | dict | None.
    effective = resolve_effective_model_config(provider, model, overrides=llm_overrides)
    client_factory = get_client(provider)

    thread_clients = threading.local()
    # Every client gets its own DEEP COPY of the config: the escalation retry
    # mutates generation.max_tokens, and a config shared across thread clients
    # lets concurrent escalations compound (4092 -> 8184 -> 16368 -> 65472...)
    # and race on the restore — the budget must never ratchet across calls.
    thread_clients.client = client_factory(copy.deepcopy(effective))  # fail fast on config/deps
    # Escalation baseline is fixed at build time: a retry may double the
    # CONFIGURED budget once, never the (possibly already escalated) live one.
    baseline_max_tokens = int(
        getattr(getattr(effective, "generation", None), "max_tokens", 0)
        or getattr(effective, "max_output_tokens", 0)
        or 0
    )

    def llm_call_fn(*, prompt: dict[str, str], schema_json: str, context: str) -> Any:
        client = getattr(thread_clients, "client", None)
        if client is None:
            client = thread_clients.client = client_factory(copy.deepcopy(effective))
        # OpenAI-family providers require response_format schema names to match
        # ^[a-zA-Z0-9_-]+$ (<=64 chars); the context tag carries ':' and the
        # source filename's '.', so sanitize before it reaches the provider.
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", context)[:40]

        def call() -> Any:
            return client.get_json_response(
                prompt,
                schema_json,
                structured_output=structured_output,
                response_top_level="object",
                response_schema_name=schema_name,
            )

        def warn_still_truncated(max_tokens: int) -> None:
            logger.warning(
                "Induction call '%s' is still truncated at max_tokens=%d — this pass "
                "should need far less output, so the model is likely looping "
                "(degenerate repetition); keeping the salvaged partial result. A "
                "stronger model is the reliable fix; llm_overrides.max_output_tokens "
                "raises the budget only if the output is genuinely large",
                context,
                max_tokens,
                extra={"component": "TemplateInduction"},
            )

        out: Any = None
        error: ClientError | None = None
        try:
            out = call()
        except ClientError as e:
            # Truncated output that could not be repaired into JSON raises,
            # but the client diagnostics still flag the truncation.
            if not client.last_call_diagnostics.get("truncated"):
                raise
            error = e
        if not client.last_call_diagnostics.get("truncated"):
            return out
        generation = getattr(client, "_generation", None)
        current = int(getattr(client, "max_tokens", 0) or 0)
        context_limit = int(getattr(client, "context_limit", 0) or 0)
        # Hard ceiling: one doubling of the configured budget, ever. A model
        # in a repetition loop truncates at ANY budget; escalating past 2x
        # only converts junk into minutes of generation time (and timeouts).
        ceiling = 2 * (baseline_max_tokens or current)
        if context_limit > 0:
            ceiling = min(ceiling, context_limit // 2)
        escalated = min(current * 2, ceiling)
        if generation is None or current <= 0 or escalated <= current:
            if error is not None:
                warn_still_truncated(current)
                raise error
            return out
        logger.warning(
            "Induction call '%s' was truncated; retrying once with max_tokens=%d",
            context,
            escalated,
            extra={"component": "TemplateInduction"},
        )
        original = getattr(generation, "max_tokens", None)
        try:
            generation.max_tokens = escalated
            out = call()
        except ClientError:
            if client.last_call_diagnostics.get("truncated"):
                warn_still_truncated(escalated)
            raise
        finally:
            generation.max_tokens = original
        if client.last_call_diagnostics.get("truncated"):
            warn_still_truncated(escalated)
        return out

    return llm_call_fn
