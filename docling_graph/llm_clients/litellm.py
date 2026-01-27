"""
LiteLLM-backed client implementation.

This client standardizes chat/completion calls through LiteLLM's OpenAI-style
API surface while implementing the LLMClientProtocol directly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Mapping

from ..exceptions import ClientError, ConfigurationError
from .config import EffectiveModelConfig
from .response_handler import ResponseHandler

logger = logging.getLogger(__name__)

_litellm_import_error: ImportError | None

try:
    import litellm
except ImportError as e:  # pragma: no cover - handled by configuration checks
    litellm = None  # type: ignore[assignment]
    _litellm_import_error = e
else:
    _litellm_import_error = None


class LiteLLMClient:
    """LiteLLM client implementation using OpenAI-style calls."""

    def __init__(self, model_config: EffectiveModelConfig, **kwargs: Any) -> None:
        self._config = model_config
        self.model = model_config.model_id
        self.model_id = model_config.model_id
        self._context_limit = model_config.context_limit
        self._max_output_tokens = model_config.max_output_tokens
        self._generation = model_config.generation
        self._reliability = model_config.reliability
        self._connection = model_config.connection

        self._setup_client(**kwargs)
        logger.info("%s initialized for model: %s", self.__class__.__name__, self.model)

    def _setup_client(self, **_kwargs: Any) -> None:
        if litellm is None:
            raise ConfigurationError(
                "LiteLLM client requires 'litellm' package",
                details={"install_command": "pip install litellm"},
                cause=_litellm_import_error,
            )

    def get_json_response(
        self, prompt: str | Mapping[str, str], schema_json: str
    ) -> Dict[str, Any] | list[Any]:
        messages = self._prepare_messages(prompt)
        raw_response, metadata = self._call_api(messages, schema_json=schema_json)
        truncated = self._check_truncation(metadata)
        return ResponseHandler.parse_json_response(
            raw_response,
            self.__class__.__name__,
            aggressive_clean=self._needs_aggressive_cleaning(),
            truncated=truncated,
            max_tokens=self.max_tokens,
        )

    def _prepare_messages(self, prompt: str | Mapping[str, str]) -> list[Dict[str, str]]:
        if isinstance(prompt, Mapping):
            prompt_mapping = dict(prompt)
            messages: list[Dict[str, str]] = []
            system_content = prompt_mapping.get("system")
            if system_content is not None:
                messages.append({"role": "system", "content": system_content})
            user_content = prompt_mapping.get("user")
            if user_content is not None:
                messages.append({"role": "user", "content": user_content})
            return messages
        return [{"role": "user", "content": prompt}]

    def _needs_aggressive_cleaning(self) -> bool:
        return self._config.provider_id == "watsonx"

    def _check_truncation(self, metadata: Dict[str, Any]) -> bool:
        finish_reason = metadata.get("finish_reason")
        if finish_reason:
            return bool(finish_reason == "length")
        return False

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        try:
            request = self._build_request(messages)
            response = litellm.completion(**request)

            choices = response.get("choices", [])
            if not choices:
                raise ClientError("LiteLLM returned no choices", details={"model": self.model})

            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                raise ClientError("LiteLLM returned empty content", details={"model": self.model})

            metadata = {
                "finish_reason": choices[0].get("finish_reason"),
                "model": response.get("model", self.model),
                "usage": response.get("usage"),
            }
            return str(content), metadata
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"LiteLLM API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
                cause=e,
            ) from e

    def _build_request(self, messages: list[Dict[str, str]]) -> dict[str, Any]:
        gen = self.generation
        max_tokens = gen.max_tokens or self._max_output_tokens
        model_name = self.model_config.litellm_model

        request: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": gen.temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "drop_params": True,
        }
        if self.model_config.provider_id != "vllm":
            request["response_format"] = {"type": "json_object"}

        if gen.top_p is not None:
            request["top_p"] = gen.top_p
        if gen.top_k is not None:
            request["top_k"] = gen.top_k
        if gen.frequency_penalty is not None:
            request["frequency_penalty"] = gen.frequency_penalty
        if gen.presence_penalty is not None:
            request["presence_penalty"] = gen.presence_penalty
        if gen.seed is not None:
            request["seed"] = gen.seed
        if gen.stop is not None:
            request["stop"] = gen.stop

        connection = self.connection
        api_key = connection.api_key.get_secret_value() if connection.api_key else None
        if api_key:
            request["api_key"] = api_key
        if connection.base_url:
            request["api_base"] = connection.base_url
        if connection.organization:
            request["organization"] = connection.organization
        if connection.headers:
            request["headers"] = dict(connection.headers)

        supported_fn = getattr(litellm, "get_supported_openai_params", None)
        if callable(supported_fn):
            try:
                supported = supported_fn(model=model_name)
                if supported:
                    required = {
                        "model",
                        "messages",
                        "api_base",
                        "api_key",
                        "headers",
                        "organization",
                        "timeout",
                        "drop_params",
                        "response_format",
                    }
                    filtered = {key: value for key, value in request.items() if key in required}
                    filtered.update(
                        {key: value for key, value in request.items() if key in supported}
                    )
                    request = filtered
            except Exception:
                logger.debug("LiteLLM supported params lookup failed for %s", model_name)

        return request

    @property
    def provider(self) -> str:
        return self._config.provider_id

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def max_tokens(self) -> int:
        return self._generation.max_tokens or self._max_output_tokens

    @property
    def timeout(self) -> int:
        return self._reliability.timeout_s

    @property
    def generation(self) -> Any:
        return self._generation

    @property
    def reliability(self) -> Any:
        return self._reliability

    @property
    def connection(self) -> Any:
        return self._connection

    @property
    def model_config(self) -> EffectiveModelConfig:
        return self._config
