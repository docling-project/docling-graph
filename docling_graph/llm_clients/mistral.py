"""
Mistral API client implementation.

Based on https://docs.mistral.ai/api/endpoint/chat
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from mistralai import Mistral
from rich import print

from .llm_base import BaseLlmClient

# Load environment variables
load_dotenv()


class MistralClient(BaseLlmClient):
    """Mistral API implementation matching official example."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "[MistralClient] [red]Error:[/red] MISTRAL_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)

        # Context limits for different models
        model_context_limits = {
            "mistral-large-latest": 128000,
            "mistral-medium-latest": 128000,
            "mistral-small-latest": 32000,
        }

        self._context_limit = model_context_limits.get(model, 32000)
        print(f"[MistralClient] Initialized for [blue]{self.model}[/blue]")

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
        # Build messages list - EXACTLY like official example
        messages = []

        if isinstance(prompt, dict):
            # Extract system and user content
            system_content = prompt.get("system", "")
            user_content = prompt.get("user", "")

            # Validate we have content
            if not system_content or not user_content:
                print("[yellow]Warning:[/yellow] Empty system or user prompt")
                print(f"  System: {bool(system_content)}")
                print(f"  User: {bool(user_content)}")

            # Add system message if present
            if system_content:
                messages.append(
                    {
                        "role": "system",
                        "content": system_content,
                    }
                )

            # Add user message
            messages.append(
                {
                    "role": "user",
                    "content": user_content or "Please provide a JSON response.",
                }
            )
        else:
            # Legacy string prompt
            if not prompt:
                print("[yellow]Warning:[/yellow] Empty prompt string")
                prompt = "Please provide a JSON response."

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

        try:
            # Call Mistral API
            res = self.client.chat.complete(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            # Get response content
            response_content = res.choices[0].message.content

            # Parse JSON
            if not response_content:
                print("[red]Error:[/red] Mistral returned empty content")
                return {}

            try:
                parsed_json = json.loads(response_content)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    print("[yellow]Warning:[/yellow] Mistral returned empty or all-null JSON")

                return parsed_json

            except json.JSONDecodeError as e:
                print(f"[red]Error:[/red] Failed to parse Mistral response as JSON: {e}")
                print(f"[yellow]Raw response:[/yellow] {response_content}")
                return {}

        except Exception as e:
            print(f"[red]Error:[/red] Mistral API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
