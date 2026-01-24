"""
Ollama (local LLM) client implementation.
Based on https://ollama.com/blog/structured-outputs
"""

import logging
from typing import Any, Dict

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

logger = logging.getLogger(__name__)

_ollama: Any | None = None
try:
    import ollama as ollama_module

    _ollama = ollama_module
except ImportError:
    logger.warning(
        "ollama package not found. Please run `pip install ollama` to use Ollama client."
    )
    _ollama = None

ollama: Any = _ollama


class OllamaClient(BaseLlmClient):
    """Ollama (local LLM) implementation using template method pattern."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "ollama"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize Ollama-specific client."""
        if ollama is None:
            raise ConfigurationError(
                "ollama package not installed",
                details={"package": "ollama", "install": "pip install ollama"},
            )

        try:
            logger.info(f"Checking Ollama connection and model '{self.model}'...")
            ollama.show(self.model)
            logger.info(f"Ollama client initialized with model: {self.model}")
        except Exception as e:
            raise ConfigurationError(
                f"Ollama connection failed: {e}",
                details={
                    "model": self.model,
                    "error": str(e),
                    "instructions": [
                        "1. Ensure Ollama is running: ollama serve",
                        f"2. Model is available: ollama pull {self.model}",
                    ],
                },
            ) from e

    def _call_api(
        self, messages: list[Dict[str, str]], **params: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Call Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Tuple of (raw_response, metadata) - Ollama doesn't provide finish_reason

        Raises:
            ClientError: If API call fails
        """
        try:
            # Get max_tokens from instance (Ollama uses num_predict)
            max_tokens = getattr(self, "_max_tokens", 8192)

            response = ollama.chat(
                model=self.model,
                messages=messages,
                format="json",
                options={
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                },
            )

            raw_json = response["message"]["content"]

            if not raw_json:
                raise ClientError("Ollama returned empty content", details={"model": self.model})

            # Ollama doesn't provide finish_reason
            metadata = {
                "model": self.model,
            }

            return str(raw_json), metadata

        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"Ollama API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
            ) from e
