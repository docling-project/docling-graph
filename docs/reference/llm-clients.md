# LLM Clients API

!!! note
    LiteLLM is the default client path for LLM calls. You can also supply a custom client (see below).

## Overview

**Module:** `docling_graph.llm_clients`

All LLM calls go through `LiteLLMClient.get_json_response()` when using the default provider/model path. That method implements `LLMClientProtocol` directly. This preserves the extraction/consolidation pipeline while standardizing provider differences through LiteLLM.

---

## LiteLLMClient (Default)

`LiteLLMClient` wraps `litellm.completion()` and uses OpenAI-style parameters with
`drop_params=True` to avoid provider-specific branching.

### Example

```python
from docling_graph.llm_clients import get_client
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config(
    "mistral",
    "mistral-large-latest",
    overrides={"generation": {"max_tokens": 4096}},
)
client_class = get_client("mistral")
client = client_class(model_config=effective)

result = client.get_json_response(
    prompt={"system": "Extract data", "user": "Alice is a manager"},
    schema_json="{}",
)
```

### JSON Mode

JSON/Structured Outputs are requested by default via `response_format`, with
`ResponseHandler` providing a fallback if the model output is not strictly JSON.

---

## Custom LLM Clients

You can supply your own LLM client so the pipeline uses your inference URL, auth, or protocol while docling-graph still builds prompts, schemas, and runs the normal extraction flow.

**Contract:** Your client must implement `LLMClientProtocol`: a single method

- `get_json_response(self, prompt: str | Mapping[str, str], schema_json: str) -> dict | list`

Pass it to the pipeline via `PipelineConfig(llm_client=your_client)` (or `run_pipeline({"llm_client": your_client, ...})`). When `llm_client` is set, the pipeline uses it and does not initialize a provider/model client from config.

### Example: custom LiteLLM-backed client (custom URL)

Use this pattern to point at an OpenAI-compatible or custom endpoint (e.g. on-prem WatsonX, vLLM, proxy) while reusing docling-graph’s `ResponseHandler` for consistent JSON parsing:

```python
from typing import Any, Dict, List, Mapping

import litellm

from docling_graph.exceptions import ClientError
from docling_graph.llm_clients.response_handler import ResponseHandler
from docling_graph.protocols import LLMClientProtocol


class LiteLLMEndpointClient:
    """Custom client that calls a single endpoint via LiteLLM with your URL/auth."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: int = 120,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_json_response(
        self, prompt: str | Mapping[str, str], schema_json: str
    ) -> Dict[str, Any] | List[Any]:
        messages = self._messages(prompt)
        request = {
            "model": self.model,
            "messages": messages,
            "api_base": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_s,
            "response_format": {"type": "json_object"},
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.headers:
            request["headers"] = dict(self.headers)

        try:
            response = litellm.completion(**request)
        except Exception as e:
            raise ClientError(
                f"LiteLLM call failed: {type(e).__name__}",
                details={"model": self.model, "api_base": self.base_url, "error": str(e)},
                cause=e,
            ) from e

        choices = response.get("choices", [])
        if not choices:
            raise ClientError("No choices in response", details={"model": self.model})
        content = choices[0].get("message", {}).get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        truncated = finish_reason == "length"

        return ResponseHandler.parse_json_response(
            content,
            client_name=self.__class__.__name__,
            aggressive_clean=False,
            truncated=truncated,
            max_tokens=self.max_tokens,
        )

    def _messages(self, prompt: str | Mapping[str, str]) -> list[dict[str, str]]:
        if isinstance(prompt, Mapping):
            out = []
            if prompt.get("system"):
                out.append({"role": "system", "content": prompt["system"]})
            if prompt.get("user"):
                out.append({"role": "user", "content": prompt["user"]})
            return out if out else [{"role": "user", "content": ""}]
        return [{"role": "user", "content": prompt}]
```

Usage with the pipeline:

```python
from docling_graph import PipelineConfig, run_pipeline

client = LiteLLMEndpointClient(
    model="openai/your-model-name",  # or e.g. hosted_vllm/...
    base_url="https://your-inference.example.com/v1",
    api_key="optional-key",
)

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    llm_client=client,
)
run_pipeline(config)
# or: config.run()
```

For OpenAI-compatible endpoints, use a model string like `openai/<model-name>` so LiteLLM routes correctly. For other protocols, implement your own HTTP call inside `get_json_response()` and return a `dict` or `list`; you can still use `ResponseHandler.parse_json_response(raw_text, ...)` for the response body.

---

## See Also

- **[API Keys Setup](../fundamentals/installation/api-keys.md)** - Configure API keys including WatsonX
- **[Model Configuration](../fundamentals/pipeline-configuration/model-configuration.md)** - Model setup
- **[Remote Inference](../fundamentals/pipeline-configuration/backend-selection.md)** - Backend selection
