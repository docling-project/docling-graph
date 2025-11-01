"""
OpenAI API client implementation.

Based on https://platform.openai.com/docs/guides/structured-outputs
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from rich import print

from .llm_base import BaseLlmClient

# Load environment variables
load_dotenv()


class OpenAIClient(BaseLlmClient):
    """OpenAI API implementation with proper message structure."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "[OpenAIClient] [red]Error:[/red] OPENAI_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Context limits for different models
        model_context_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16000,
            "gpt-3.5-turbo-16k": 16000,
        }

        self._context_limit = model_context_limits.get(model, 128000)
        print(f"[OpenAIClient] Initialized for [blue]{self.model}[/blue]")

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
        # Handle both legacy string prompts and new dict prompts
        if isinstance(prompt, dict):
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ]
        else:
            # Legacy: add generic system message and use prompt as user message
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format.",
                },
                {"role": "user", "content": prompt},
            ]

        try:
            # Use chat completions with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},  # Official JSON mode
                temperature=0.1,  # Low temperature for consistent extraction
            )

            # Extract the JSON content from the response
            content = response.choices[0].message.content

            # Parse JSON
            try:
                parsed_json = json.loads(content)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    print("[yellow]Warning:[/yellow] OpenAI returned empty or all-null JSON")

                return parsed_json

            except json.JSONDecodeError as e:
                print(f"[red]Error:[/red] Failed to parse OpenAI response as JSON: {e}")
                print(f"[yellow]Raw response:[/yellow] {content}")
                return {}

        except Exception as e:
            print(f"[red]Error:[/red] OpenAI API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
