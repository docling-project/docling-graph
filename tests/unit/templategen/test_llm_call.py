"""Unit tests for the public ``build_llm_call_fn`` templategen API.

Exercises the ``LlmCallFn`` contract (keyword-only ``prompt``/``schema_json``/
``context`` -> Any), provider-safe schema-name sanitization, the eager
fail-fast client construction, and the one-shot truncation-escalation retry —
all against a fake client so no network or real provider is required.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest


class _FakeGeneration:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens


class _FakeClient:
    """Minimal stand-in for a LiteLLMClient satisfying the builder's duck-typing."""

    def __init__(self, effective: Any, *, truncate_first_call: bool = False) -> None:
        self.effective = effective
        self._generation = _FakeGeneration(100)
        self.max_tokens = 100
        self.context_limit = 4096
        self.model = "fake-model"
        self.last_call_diagnostics: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self._truncate_first_call = truncate_first_call

    def get_json_response(
        self,
        prompt: dict | str,
        schema_json: str,
        structured_output: bool = True,
        response_top_level: str = "object",
        response_schema_name: str = "extraction_result",
    ) -> dict:
        self.calls.append(
            {
                "prompt": prompt,
                "schema_json": schema_json,
                "structured_output": structured_output,
                "response_top_level": response_top_level,
                "response_schema_name": response_schema_name,
                "max_tokens": self._generation.max_tokens,
            }
        )
        if self._truncate_first_call and len(self.calls) == 1:
            self.last_call_diagnostics = {"truncated": True}
            return {"partial": True}
        self.last_call_diagnostics = {"truncated": False}
        return {"ok": True, "call": len(self.calls)}


class _FakeEffective:
    def __init__(self) -> None:
        self.generation = _FakeGeneration(200)
        self.max_output_tokens = 200


@pytest.fixture
def patched_builder(monkeypatch):
    """Patch the builder's provider resolution to use fake clients.

    Returns a list that collects every constructed fake client so tests can
    assert on eager construction and per-call behavior.
    """
    import docling_graph.llm_clients as llm_clients
    import docling_graph.llm_clients.config as llm_config

    created: list[_FakeClient] = []
    state = {"truncate_first_call": False}

    def fake_get_client(provider: str) -> Any:
        def factory(effective: Any) -> _FakeClient:
            client = _FakeClient(effective, truncate_first_call=state["truncate_first_call"])
            created.append(client)
            return client

        return factory

    monkeypatch.setattr(llm_clients, "get_client", fake_get_client)
    monkeypatch.setattr(
        llm_config,
        "resolve_effective_model_config",
        lambda provider, model, overrides=None: _FakeEffective(),
    )
    return created, state


def test_returns_callable_satisfying_the_contract(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    created, _ = patched_builder
    fn = build_llm_call_fn("mistral", "mistral-small-latest")

    assert callable(fn)
    # Eager fail-fast: one client built at construction time, before any call.
    assert len(created) == 1
    assert created[0].calls == []

    out = fn(prompt={"system": "s", "user": "u"}, schema_json="{}", context="Root")
    assert out == {"ok": True, "call": 1}
    # Same-thread call reuses the eagerly built client (no new client).
    assert len(created) == 1
    assert len(created[0].calls) == 1


def test_call_is_keyword_only(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    fn = build_llm_call_fn("mistral", "mistral-small-latest")
    with pytest.raises(TypeError):
        fn({"user": "u"}, "{}", "Root")  # type: ignore[call-arg]


def test_schema_name_is_sanitized_for_providers(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    created, _ = patched_builder
    fn = build_llm_call_fn("openai", "gpt-4o-mini")
    # ':' (context tag) and '.' (filename) are illegal in response_format names.
    fn(prompt={"user": "u"}, schema_json="{}", context="Invoice:line_item.total")

    schema_name = created[0].calls[0]["response_schema_name"]
    assert schema_name == "Invoice_line_item_total"
    assert len(schema_name) <= 40


def test_structured_output_flag_is_forwarded(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    created, _ = patched_builder
    fn = build_llm_call_fn("mistral", "m", structured_output=False)
    fn(prompt={"user": "u"}, schema_json="{}", context="Root")
    assert created[0].calls[0]["structured_output"] is False


def test_truncation_triggers_one_escalated_retry(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    created, state = patched_builder
    state["truncate_first_call"] = True
    fn = build_llm_call_fn("mistral", "mistral-small-latest")

    out = fn(prompt={"user": "u"}, schema_json="{}", context="Root")

    client = created[0]
    # Exactly two calls: the truncated one and one escalated retry.
    assert len(client.calls) == 2
    # First call at the configured budget (100); retry doubled to 200
    # (min(current*2=200, ceiling=min(2*baseline=400, context//2=2048))).
    assert client.calls[0]["max_tokens"] == 100
    assert client.calls[1]["max_tokens"] == 200
    # The retry's result is returned.
    assert out == {"ok": True, "call": 2}
    # The live generation budget is restored after the retry.
    assert client._generation.max_tokens == 100


def test_each_thread_gets_its_own_client(patched_builder):
    from docling_graph.templategen import build_llm_call_fn

    created, _ = patched_builder
    fn = build_llm_call_fn("mistral", "mistral-small-latest")
    assert len(created) == 1  # eager client on the building thread

    def worker() -> None:
        fn(prompt={"user": "u"}, schema_json="{}", context="Root")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # The worker thread lazily constructed its own client (no shared state).
    assert len(created) == 2
