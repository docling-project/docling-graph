"""
vLLM (local LLM) client implementation.
Uses OpenAI-compatible API server from vLLM.
Cross-platform (Linux/Windows) via vLLM server mode.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, cast

from rich import print as rich_print

from .base import BaseLlmClient
from .config import get_context_limit

# Requires `pip install openai`
# Make the lazy import optional to satisfy type checkers when assigning None
_OpenAI: Any | None = None
try:
    from openai import OpenAI as OpenAI_module

    _OpenAI = OpenAI_module
except ImportError:
    rich_print(
        "[red]Error:[/red] `openai` package not found. "
        "Please run `pip install openai` to use the vLLM client."
    )
    _OpenAI = None

# Expose as Any to allow None fallback without mypy issues
OpenAI: Any = _OpenAI

if TYPE_CHECKING:  # Only imported for type checking; avoids runtime dependency at import
    from openai.types.chat import ChatCompletionMessageParam


class VllmClient(BaseLlmClient):
    """vLLM client implementation using OpenAI-compatible API."""

    def __init__(
        self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"
    ) -> None:
        """Initialize vLLM client."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # Use centralized config registry (vllm provider)
        self._context_limit = get_context_limit("vllm", model)

        # Test connection
        try:
            rich_print(
                f"[blue][VllmClient][/blue] Connecting to vLLM server at: [cyan]{self.base_url}[/cyan]"
            )
            self.client.models.list()
            rich_print("[blue][VllmClient][/blue] Connected successfully")
            rich_print(f"[blue][VllmClient][/blue] Using model: [blue]{self.model}[/blue]")
        except Exception as e:
            rich_print(f"[red]vLLM connection failed:[/red] {e}")
            rich_print("\n[yellow]Setup instructions:[/yellow]")
            rich_print("  1. Start vLLM server in a separate terminal:")
            rich_print(f"     [cyan]vllm serve {self.model}[/cyan]")
            rich_print("  2. Wait for server to load (may take 1-2 minutes)")
            rich_print(f"  3. Ensure server is accessible at: [cyan]{self.base_url}[/cyan]")
            rich_print("\n[dim]On Windows: Run vLLM server in WSL2 or Docker[/dim]")
            raise

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
        """
        Execute vLLM chat with JSON format using OpenAI-compatible API.

        Args:
            prompt: Either a string (legacy) or dict with 'system' and 'user' keys.
            schema_json: JSON schema (can be used for guided decoding).

        Returns:
            Parsed JSON response from vLLM.
        """
        # Handle both legacy string prompts and new dict prompts
        # Define once to avoid mypy no-redef when annotating in both branches
        messages: list[ChatCompletionMessageParam]
        if isinstance(prompt, dict):
            messages = [
                {"role": "system", "content": prompt.get("system", "")},
                {"role": "user", "content": prompt.get("user", "")},
            ]
        else:
            # Legacy: treat entire prompt as user message
            messages = [{"role": "user", "content": prompt}]

        try:
            # Call vLLM via OpenAI-compatible API with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"},  # Force JSON output
                # vLLM supports extra_body for additional parameters like guided_json
                # extra_body={"guided_json": schema_json}  # Uncomment for schema validation
            )

            # Get response content
            raw_json = response.choices[0].message.content

            # Parse JSON
            if not raw_json:
                rich_print("[red]Error:[/red] vLLM returned empty content")
                return {}

            try:
                parsed_json = json.loads(raw_json)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    rich_print("[yellow]Warning:[/yellow] vLLM returned empty or all-null JSON")

                # Ensure return type matches Dict[str, Any]
                if isinstance(parsed_json, dict):
                    return cast(Dict[str, Any], parsed_json)
                else:
                    rich_print(
                        "[yellow]Warning:[/yellow] Expected a JSON object; got non-dict. Returning empty dict."
                    )
                    return {}
            except json.JSONDecodeError as e:
                rich_print(f"[red]Error:[/red] Failed to parse vLLM response as JSON: {e}")
                rich_print(f"[yellow]Raw response:[/yellow] {raw_json}")
                return {}

        except Exception as e:
            rich_print(f"[red]Error:[/red] vLLM API call failed: {type(e).__name__}: {e}")
            return {}

    @property
    def context_limit(self) -> int:
        """Return context window size."""
        return self._context_limit
