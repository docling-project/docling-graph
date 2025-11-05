"""
Google Gemini API client implementation.
Based on https://ai.google.dev/gemini-api/docs/structured-output
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit

# Load environment variables
load_dotenv()

# Requires `pip install google-generativeai`
# Make the lazy import optional to satisfy type checkers when assigning None
_genai: Any | None = None
_genai_types: Any | None = None
try:
    import google.genai as genai_module
    from google.genai import types as types_module

    _genai = genai_module
    _genai_types = types_module
except ImportError:
    rich_print(
        "[red]Error:[/red] `google-genai` package not found. "
        "Please run `pip install google-genai` to use Gemini client."
    )
    _genai = None
    _genai_types = None

# Expose as Any to allow None fallback without mypy issues
genai: Any = _genai
types: Any = _genai_types


class GeminiClient(BaseLlmClient):
    """Google Gemini API implementation with proper JSON response format."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[GeminiClient] [red]Error:[/red] GEMINI_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Use centralized config registry
        self._context_limit = get_context_limit("google", model)

        rich_print(f"[GeminiClient] Initialized for [blue]{self.model}[/blue]")

    def get_json_response(self, prompt: str | dict[str, str], schema_json: str) -> Dict[str, Any]:
        """
        Execute Gemini generate_content with JSON response mode.

        Official docs: https://ai.google.dev/gemini-api/docs/structured-output

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (for reference).

        Returns:
            Parsed JSON response from Gemini.
        """
        # Handle both legacy string prompts and new dict prompts
        if isinstance(prompt, dict):
            # Combine system and user into single content
            contents = f"{prompt['system']}\n\n{prompt['user']}"
        else:
            contents = prompt

        try:
            # Configure JSON response mode (official method)
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for consistent extraction
            )

            # Generate content
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            # Get response text
            response_text = response.text

            # Parse JSON
            try:
                parsed: Any = json.loads(response_text)
                # Normalize to a dict for return type consistency
                if isinstance(parsed, dict):
                    result: Dict[str, Any] = parsed
                elif isinstance(parsed, list):
                    result = {"result": parsed}
                else:
                    result = {"value": parsed}

                # Validate it's not empty
                if not result or (isinstance(result, dict) and not any(result.values())):
                    rich_print("[yellow]Warning:[/yellow] Gemini returned empty or all-null JSON")

                return result

            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse Gemini response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {response_text}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] Gemini API call failed: {type(e).__name__}: {e}")
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
