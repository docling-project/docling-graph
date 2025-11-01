"""
vLLM (local LLM) client implementation.
Uses OpenAI-compatible API server from vLLM.
Cross-platform (Linux/Windows) via vLLM server mode.
"""

import json
from typing import Any, Dict

from rich import print

from .llm_base import BaseLlmClient

# Requires `pip install openai` (OpenAI Python client)
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class VllmClient(BaseLlmClient):
    """vLLM client implementation using OpenAI-compatible API."""

    def __init__(
        self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"
    ):
        """
        Initialize vLLM client.

        Args:
            model: Model name or path (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            base_url: vLLM server base URL (default: "http://localhost:8000/v1")
            api_key: API key (default: "EMPTY" for local servers)

        Setup:
            1. Start vLLM server: vllm serve meta-llama/Llama-3.1-8B-Instruct
            2. Server will run at http://localhost:8000
            3. Client will connect automatically
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI client is required for vLLM. Install it with: pip install openai"
            )

        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # Context limits for common models
        model_context_limits = {
            "meta-llama/Llama-3.1-8B": 128000,
            "mistralai/Mixtral-8x7B-v0.1": 32000,
            "qwen/Qwen2-7B": 128000,
            "ibm-granite/granite-4.0-h-tiny": 128000,
            "ibm-granite/granite-4.0-h-micro": 128000,
            "ibm-granite/granite-4.0-1b": 128000,
            "ibm-granite/granite-4.0-350m": 128000,
        }

        # Try to match model name to known context limits
        self._context_limit = model_context_limits.get(model, 32000)

        # Test connection
        try:
            print(
                f"[blue][VllmClient][/blue] Connecting to vLLM server at: [cyan]{self.base_url}[/cyan]"
            )
            # Test connection by listing models
            models = self.client.models.list()
            print("[blue][VllmClient][/blue] Connected successfully")
            print(f"[blue][VllmClient][/blue] Using model: [blue]{self.model}[/blue]")
        except Exception as e:
            print(f"[red]✗ vLLM connection failed:[/red] {e}")
            print("\n[yellow]Setup instructions:[/yellow]")
            print("  1. Start vLLM server in a separate terminal:")
            print(f"     [cyan]vllm serve {self.model}[/cyan]")
            print("  2. Wait for server to load (may take 1-2 minutes)")
            print(f"  3. Ensure server is accessible at: [cyan]{self.base_url}[/cyan]")
            print("\n[dim]On Windows: Run vLLM server in WSL2 or Docker[/dim]")
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
                print("[red]Error:[/red] vLLM returned empty content")
                return {}

            try:
                parsed_json = json.loads(raw_json)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    print("[yellow]Warning:[/yellow] vLLM returned empty or all-null JSON")

                return parsed_json
            except json.JSONDecodeError as e:
                print(f"[red]Error:[/red] Failed to parse vLLM response as JSON: {e}")
                print(f"[yellow]Raw response:[/yellow] {raw_json}")
                return {}

        except Exception as e:
            print(f"[red]Error:[/red] vLLM API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        """Return context window size."""
        return self._context_limit
