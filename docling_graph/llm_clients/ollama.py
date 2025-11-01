"""
Ollama (local LLM) client implementation.

Based on https://ollama.com/blog/structured-outputs
"""

import json
from typing import Any, Dict

from rich import print

from .llm_base import BaseLlmClient

# Requires `pip install ollama`
try:
    import ollama
except ImportError:
    print(
        "[red]Error:[/red] `ollama` package not found. Please run `pip install ollama` to use local LLMs."
    )
    ollama = None


class OllamaClient(BaseLlmClient):
    """Ollama (local LLM) implementation with proper message structure."""

    def __init__(self, model: str):
        if ollama is None:
            raise ImportError(
                "Ollama client could not be imported. Please install it with: pip install ollama"
            )

        self.model = model

        # Context limits for different models
        model_context_limits = {"llama3.1:8b": 128000, "llama3.2:3b": 128000, "mixtral:8x7b": 32000}

        # Extract base model name for lookup
        base_model = model.split(":")[0] + ":" + model.split(":")[1] if ":" in model else model
        self._context_limit = model_context_limits.get(base_model, 8000)

        try:
            print(f"[OllamaClient] Checking Ollama connection and model '{self.model}'...")
            ollama.show(self.model)
            print(f"[OllamaClient] Initialized with Ollama model: [blue]{self.model}[/blue]")
        except Exception as e:
            print(f"[red]Ollama connection error:[/red] {e}")
            print("Please ensure:")
            print("  1. Ollama is running: ollama serve")
            print(f"  2. Model is available: ollama pull {self.model}")
            raise

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
                parsed_json = json.loads(raw_json)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    print("[yellow]Warning:[/yellow] Ollama returned empty or all-null JSON")

                return parsed_json

            except json.JSONDecodeError as e:
                print(f"[red]Error:[/red] Failed to parse Ollama response as JSON: {e}")
                print(f"[yellow]Raw response:[/yellow] {raw_json}")
                return {}

        except Exception as e:
            print(f"[red]Error:[/red] Ollama API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
