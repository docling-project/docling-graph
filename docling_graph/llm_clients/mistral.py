"""
Mistral API client implementation.
Based on https://docs.mistral.ai/api/endpoint/chat
"""

import logging
from typing import Any, Dict, cast

from dotenv import load_dotenv

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

load_dotenv()

logger = logging.getLogger(__name__)

_Mistral: Any | None = None
try:
    from mistralai import Mistral as Mistral_module

    _Mistral = Mistral_module
except ImportError:
    logger.warning(
        "mistralai package not found. Please run `pip install mistralai` to use Mistral client."
    )
    _Mistral = None

Mistral: Any = _Mistral


class MistralClient(BaseLlmClient):
    """Mistral API implementation using template method pattern."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "mistral"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize Mistral-specific client."""
        if Mistral is None:
            raise ConfigurationError(
                "mistralai package not installed",
                details={"package": "mistralai", "install": "pip install mistralai"},
            )

        self.api_key = self._get_required_env("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)

        logger.info(f"Mistral client initialized for model: {self.model}")

    def _call_api(self, messages: list[Dict[str, str]], **params: Any) -> str:
        """
        Call Mistral API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Raw response string from Mistral

        Raises:
            ClientError: If API call fails
        """
        try:
            # Get max_tokens and timeout from instance
            max_tokens = getattr(self, "_max_tokens", 8192)
            timeout_seconds = getattr(self, "_timeout", 300)

            response = self.client.chat.complete(
                model=self.model,
                messages=cast(Any, messages),
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
            )

            response_content = response.choices[0].message.content

            if not response_content:
                raise ClientError("Mistral returned empty content", details={"model": self.model})

            if isinstance(response_content, str):
                return response_content
            else:
                parts: list[str] = []
                for chunk in response_content:
                    text = getattr(chunk, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
                return "".join(parts)

        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"Mistral API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
            ) from e
