"""
Ollama (local LLM) client implementation.
Based on https://ollama.com/blog/structured-outputs
"""

import json
from typing import Any, Dict, cast

from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit

# Requires `pip install ollama`
# Make the lazy import optional to satisfy type checkers when assigning None
_ollama: Any | None = None
try:
    import ollama as ollama_module  # Import to a temporary name

    _ollama = ollama_module  # Assign to the existing variable
except ImportError:
    rich_print(
        "[red]Error:[/red] `ollama` package not found. "
        "Please run `pip install ollama` to use Ollama client."
    )
    _ollama = None

# Expose as Any to allow None fallback without mypy issues
ollama: Any = _ollama


class OllamaClient(BaseLlmClient):
    """Ollama (local LLM) implementation with proper message structure."""

    def __init__(self, model: str) -> None:
        if ollama is None:
            raise ImportError(
                "Ollama client could not be imported. Please install it with: pip install ollama"
            )

        self.model = model

        # Use centralized config registry (ollama provider)
        self._context_limit = get_context_limit("ollama", model)

        try:
            rich_print(f"[OllamaClient] Checking Ollama connection and model '{self.model}'...")
            ollama.show(self.model)
            rich_print(f"[OllamaClient] Initialized with Ollama model: [blue]{self.model}[/blue]")
        except Exception as e:
            rich_print(f"[red]Ollama connection error:[/red] {e}")
            rich_print("Please ensure:")
            rich_print("  1. Ollama is running: ollama serve")
            rich_print(f"  2. Model is available: ollama pull {self.model}")
            raise RuntimeError(str(e)) from e

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        """
        Execute Ollama chat with JSON format.

        Official docs: https://ollama.com/blog/structured-outputs

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (can be used as format parameter).

        Returns:
            Parsed JSON response from Ollama.
        """
        # Handle both legacy string prompts and new dict prompts
        if isinstance(prompt, dict):
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
        else:
            # Legacy: treat entire prompt as user message
            messages = [{"role": "user", "content": prompt}]

        try:
            # Call Ollama with JSON format (official method)
            res = ollama.chat(
                model=self.model,
                messages=messages,
                format="json",  # Official JSON mode - ensures valid JSON
                options={
                    "temperature": 0.1,  # Low temperature for consistent extraction
                },
            )

            # Get response content
            raw_json = res["message"]["content"]

            # Parse JSON
            try:
                parsed = json.loads(raw_json)

                # Validate it's not empty
                if not parsed or (isinstance(parsed, dict) and not any(parsed.values())):
                    rich_print("[yellow]Warning:[/yellow] Ollama returned empty or all-null JSON")

                # Ensure return type matches Dict[str, Any]
                if isinstance(parsed, dict):
                    return cast(Dict[str, Any], parsed)
                else:
                    rich_print(
                        "[yellow]Warning:[/yellow] Expected a JSON object; got non-dict. Returning empty dict."
                    )
                    return {}

            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse Ollama response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {raw_json}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] Ollama API call failed: {type(e).__name__}: {e}")
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
