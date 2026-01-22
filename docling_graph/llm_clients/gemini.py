"""
Google Gemini API client implementation.
Based on https://ai.google.dev/gemini-api/docs/structured-output
"""

import logging
from typing import Any, Dict

from dotenv import load_dotenv

from ..exceptions import ClientError, ConfigurationError
from .base import BaseLlmClient

load_dotenv()

logger = logging.getLogger(__name__)

_genai: Any | None = None
_genai_types: Any | None = None
try:
    import google.genai as genai_module
    from google.genai import types as types_module

    _genai = genai_module
    _genai_types = types_module
except ImportError:
    logger.warning(
        "google-genai package not found. "
        "Please run `pip install google-genai` to use Gemini client."
    )
    _genai = None
    _genai_types = None

genai: Any = _genai
types: Any = _genai_types


class GeminiClient(BaseLlmClient):
    """Google Gemini API implementation using template method pattern."""

    def _provider_id(self) -> str:
        """Return provider ID for config lookup."""
        return "google"

    def _setup_client(self, **kwargs: Any) -> None:
        """Initialize Gemini-specific client."""
        if genai is None or types is None:
            raise ConfigurationError(
                "google-genai package not installed",
                details={"package": "google-genai", "install": "pip install google-genai"},
            )

        self.api_key = self._get_required_env("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Gemini client initialized for model: {self.model}")

    def _call_api(self, messages: list[Dict[str, str]], **params: Any) -> str:
        """
        Call Gemini API.

        Gemini doesn't use message arrays - it combines system and user into single content.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **params: Additional parameters (schema_json, etc.)

        Returns:
            Raw response string from Gemini

        Raises:
            ClientError: If API call fails
        """
        try:
            contents_parts = []
            for msg in messages:
                content = msg.get("content", "")
                if content:
                    contents_parts.append(content)

            contents = "\n\n".join(contents_parts)

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )

            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            response_text = response.text

            if not response_text:
                raise ClientError("Gemini returned empty response", details={"model": self.model})

            return str(response_text)

        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(
                f"Gemini API call failed: {type(e).__name__}",
                details={"model": self.model, "error": str(e)},
            ) from e


# Made with Bob
