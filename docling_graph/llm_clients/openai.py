"""
OpenAI API client implementation (refactored).

Reduced from 124 lines to ~70 lines by using the new base class
and ResponseHandler.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict

from dotenv import load_dotenv
from rich import print as rich_print

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_OpenAI: Any | None = None

try:
    from openai import OpenAI as OpenAI_module

    _OpenAI = OpenAI_module
except ImportError:
    logger.warning("openai package not found. Install with: pip install 'docling-graph[openai]'")
    _OpenAI = None

OpenAI: Any = _OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class OpenAIClient(BaseLlmClient):
    """OpenAI API client - refactored version."""

    def _provider_id(self) -> str:
        """Return provider ID for configuration."""
        return "openai"

    def _setup_client(self, **kwargs: Any) -> None:
        """
        Initialize OpenAI client.

        Raises:
            ConfigurationError: If API key is missing or package not installed
        """
        if _OpenAI is None:
            raise ConfigurationError(
                "OpenAI client requires 'openai' package",
                details={
                    "install_command": "pip install 'docling-graph[openai]'",
                    "alternative": "pip install openai",
                },
            )

        # Load API key
        self.api_key = self._get_required_env("OPENAI_API_KEY")

        # Initialize client
        self.client = OpenAI(api_key=self.api_key)

        rich_print(f"[blue][OpenAI][/blue] Initialized for model: [cyan]{self.model}[/cyan]")

    def _call_api(self, messages: list[Dict[str, str]], **params: Any) -> str:
        """
        Call OpenAI API.

        Args:
            messages: List of message dicts
            **params: Additional parameters

        Returns:
            Raw response string

        Raises:
            ClientError: If API call fails
        """
        # Convert to OpenAI message format
        if TYPE_CHECKING:
            api_messages: list[ChatCompletionMessageParam] = messages  # type: ignore
        else:
            api_messages = messages

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = response.choices[0].message.content

            if not content:
                raise ClientError("OpenAI returned empty content", details={"model": self.model})

            return str(content)

        except Exception as e:
            raise ClientError(
                f"OpenAI API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
                cause=e,
            ) from e
