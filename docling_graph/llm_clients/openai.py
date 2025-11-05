"""
OpenAI API client implementation.
Based on https://platform.openai.com/docs/guides/structured-outputs
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, cast

from dotenv import load_dotenv
from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit

# Load environment variables
load_dotenv()

# Requires `pip install openai`
_OpenAI: Any | None = None
try:
    from openai import OpenAI as OpenAI_module

    _OpenAI = OpenAI_module
except ImportError:
    rich_print(
        "[red]Error:[/red] `openai` package not found. "
        "Please run `pip install openai` to use OpenAI client."
    )
    _OpenAI = None

OpenAI: Any = _OpenAI


class OpenAIClient(BaseLlmClient):
    """OpenAI API implementation with proper message structure."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[OpenAIClient] [red]Error:[/red] OPENAI_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Use centralized config registry
        self._context_limit = get_context_limit("openai", model)

        rich_print(f"[OpenAIClient] Initialized for [blue]{self.model}[/blue]")

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        """
        Execute OpenAI chat completion with JSON mode.
        Official docs: https://platform.openai.com/docs/guides/structured-outputs

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (for reference, not directly used by OpenAI).

        Returns:
            Parsed JSON response from OpenAI.
        """
        if TYPE_CHECKING:
            from openai.types.chat import ChatCompletionMessageParam

            messages: list[ChatCompletionMessageParam]

        if isinstance(prompt, dict):
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format.",
                },
                {"role": "user", "content": prompt},
            ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = response.choices[0].message.content or ""

            try:
                parsed_json = json.loads(content)

                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    rich_print("[yellow]Warning:[/yellow] OpenAI returned empty or all-null JSON")

                if isinstance(parsed_json, dict):
                    return cast(Dict[str, Any], parsed_json)
                else:
                    rich_print(
                        "[yellow]Warning:[/yellow] Expected a JSON object; got non-dict. "
                        "Returning empty dict."
                    )
                    return {}

            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse OpenAI response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {content}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] OpenAI API call failed: {type(e).__name__}: {e}")
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
