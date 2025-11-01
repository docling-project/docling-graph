"""
Google Gemini API client implementation.

Based on https://ai.google.dev/gemini-api/docs/structured-output
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich import print

from .llm_base import BaseLlmClient

# Load environment variables
load_dotenv()


class GeminiClient(BaseLlmClient):
    """Google Gemini API implementation with proper JSON response format."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "[GeminiClient] [red]Error:[/red] GEMINI_API_KEY not set. "
                "Please set it in your environment or .env file."
            )

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # Context limits for different models
        model_context_limits = {
            "gemini-1.5-flash": 1000000,
            "gemini-1.5-pro": 1000000,
            "gemini-2.0-flash": 1000000,
            "gemini-2.5-pro": 1000000,
        }

        self._context_limit = model_context_limits.get(model, 1000000)
        print(f"[GeminiClient] Initialized for [blue]{self.model}[/blue]")

    def get_json_response(self, prompt: str | dict, schema_json: str) -> Dict[str, Any]:
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
                parsed_json = json.loads(response_text)

                # Validate it's not empty
                if not parsed_json or (
                    isinstance(parsed_json, dict) and not any(parsed_json.values())
                ):
                    print("[yellow]Warning:[/yellow] Gemini returned empty or all-null JSON")

                return parsed_json

            except json.JSONDecodeError as e:
                print(f"[red]Error:[/red] Failed to parse Gemini response as JSON: {e}")
                print(f"[yellow]Raw response:[/yellow] {response_text}")
                return {}

        except Exception as e:
            print(f"[red]Error:[/red] Gemini API call failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    @property
    def context_limit(self) -> int:
        return self._context_limit
