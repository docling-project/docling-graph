"""
Mistral API client implementation.
Based on https://docs.mistral.ai/api/endpoint/chat
"""

import json
import os
from typing import Any, Dict, cast

from dotenv import load_dotenv
from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit, get_model_config

# Load environment variables
load_dotenv()

# Requires `pip install mistralai`
_Mistral: Any | None = None
try:
    from mistralai import Mistral as Mistral_module

    _Mistral = Mistral_module
except ImportError:
    rich_print(
        "[red]Error:[/red] `mistralai` package not found. "
        "Please run `pip install mistralai` to use Mistral client."
    )
    _Mistral = None

Mistral: Any = _Mistral


class MistralClient(BaseLlmClient):
    """Mistral API implementation matching official example."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[MistralClient] [red]Error:[/red] MISTRAL_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)

        # Use centralized config registry
        self._context_limit = get_context_limit("mistral", model)

        rich_print(f"[MistralClient] Initialized for [blue]{self.model}[/blue]")

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        """
        Execute Mistral chat.complete with proper message structure.
        Official example: https://docs.mistral.ai/api/endpoint/chat

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (for reference).

        Returns:
            Parsed JSON response from Mistral.
        """
        messages: list[dict[str, str]] = []

        if isinstance(prompt, dict):
            system_content = prompt.get("system", "")
            user_content = prompt.get("user", "")

            if not system_content or not user_content:
                rich_print("[yellow]Warning:[/yellow] Empty system or user prompt")
                rich_print(f" System: {bool(system_content)}")
                rich_print(f" User: {bool(user_content)}")

            if system_content:
                messages.append({"role": "system", "content": system_content})

            messages.append(
                {"role": "user", "content": user_content or "Please provide a JSON response."}
            )
        else:
            if not prompt:
                rich_print("[yellow]Warning:[/yellow] Empty prompt string")
                prompt = "Please provide a JSON response."

            messages.append({"role": "user", "content": prompt})

        try:
            res = self.client.chat.complete(
                model=self.model,
                messages=cast(Any, messages),
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            response_content = res.choices[0].message.content

            if not response_content:
                rich_print("[red]Error:[/red] Mistral returned empty content")
                return {}

            try:
                if isinstance(response_content, str):
                    raw = response_content
                else:
                    parts: list[str] = []
                    for chunk in response_content:
                        text = getattr(chunk, "text", None)
                        if isinstance(text, str):
                            parts.append(text)
                    raw = "".join(parts)

                parsed = json.loads(raw)

                if not parsed or (isinstance(parsed, dict) and not any(parsed.values())):
                    rich_print("[yellow]Warning:[/yellow] Mistral returned empty or all-null JSON")

                if isinstance(parsed, dict):
                    return cast(Dict[str, Any], parsed)
                else:
                    rich_print(
                        "[yellow]Warning:[/yellow] Expected a JSON object; got non-dict. "
                        "Returning empty dict."
                    )
                    return {}

            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse Mistral response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {response_content}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] Mistral API call failed: {type(e).__name__}: {e}")
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
