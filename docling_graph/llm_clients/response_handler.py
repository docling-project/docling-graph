"""
Centralized JSON response parsing and validation for LLM clients.

This module eliminates 150+ lines of duplicated code across all LLM clients
by providing a single, well-tested implementation of JSON parsing, cleaning,
and validation logic.
"""

import json
import re
from typing import Any, Dict

from rich import print as rich_print

from ..exceptions import ClientError


class ResponseHandler:
    """
    Centralized response parsing - eliminates duplication across all clients.

    Handles:
    - Markdown code block removal
    - JSON extraction from mixed content
    - Aggressive cleaning for problematic providers
    - Structure validation
    - Consistent error reporting
    """

    @staticmethod
    def parse_json_response(
        raw_response: str,
        client_name: str,
        aggressive_clean: bool = False
    ) -> Dict[str, Any] | list[Any]:
        """
        Parse and validate JSON response from LLM.

        This is the main entry point used by all LLM clients. It handles
        all common response formats and edge cases.

        Args:
            raw_response: Raw string response from LLM
            client_name: Name of the client (for error messages)
            aggressive_clean: Apply more aggressive cleaning (for WatsonX, etc.)

        Returns:
            Parsed and validated JSON (dictionary or list)

        Raises:
            ClientError: If response cannot be parsed or is invalid
        """
        # Validate input
        if not raw_response or not raw_response.strip():
            raise ClientError(
                f"{client_name} returned empty response",
                details={"raw_response": raw_response}
            )

        # Clean response
        content = ResponseHandler._clean_response(raw_response, aggressive_clean)

        # Parse JSON
        try:
            parsed = json.loads(content)
            return ResponseHandler._validate_structure(parsed, client_name)

        except json.JSONDecodeError as e:
            # Provide detailed error information
            rich_print(f"[red]Error:[/red] {client_name} JSON parse failed: {e}")
            rich_print("[yellow]Raw response (first 500 chars):[/yellow]")
            rich_print(raw_response[:500])

            raise ClientError(
                f"{client_name}: Invalid JSON response",
                details={
                    "client_name": client_name,
                    "error": str(e),
                    "raw_response": raw_response[:500],
                    "cleaned_content_preview": content[:200]
                },
                cause=e
            ) from e

    @staticmethod
    def _clean_response(content: str, aggressive: bool) -> str:
        """
        Clean response by removing markdown and extracting JSON.

        Args:
            content: Raw response content
            aggressive: Whether to apply aggressive cleaning

        Returns:
            Cleaned content ready for JSON parsing
        """
        content = content.strip()

        # Remove markdown code blocks
        if "```" in content:
            content = ResponseHandler._extract_from_markdown(content)

        # Aggressive cleaning for problematic providers
        if aggressive:
            content = ResponseHandler._aggressive_clean(content)

        return content.strip()

    @staticmethod
    def _extract_from_markdown(content: str) -> str:
        """
        Extract JSON from markdown code blocks.

        Handles:
        - ```json ... ```
        - ``` ... ```
        - Plain JSON with text before/after

        Args:
            content: Content potentially containing markdown

        Returns:
            Extracted JSON content
        """
        # Pattern 1: ```json ... ```
        if "```json" in content:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Pattern 2: ``` ... ```
        if "```" in content:
            match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Pattern 3: Find JSON object or array start
        for char in ["{", "["]:
            idx = content.find(char)
            if idx != -1:
                return content[idx:]

        return content

    @staticmethod
    def _aggressive_clean(content: str) -> str:
        """
        Apply aggressive cleaning for problematic responses.

        This is used for providers like WatsonX that may include
        extra text before/after the JSON.

        Args:
            content: Content to clean

        Returns:
            Aggressively cleaned content
        """
        # Remove common prefixes
        prefixes = [
            "Here is the JSON:",
            "Here's the JSON:",
            "JSON:",
            "Response:",
            "Output:",
            "Result:",
        ]

        for prefix in prefixes:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()

        # Find first JSON object or array
        first_brace = content.find("{")
        first_bracket = content.find("[")

        # Determine which comes first
        if first_brace == -1 and first_bracket == -1:
            return content  # No JSON found

        if first_brace == -1:
            start_idx = first_bracket
            start_char = "["
            end_char = "]"
        elif first_bracket == -1:
            start_idx = first_brace
            start_char = "{"
            end_char = "}"
        else:
            start_idx = min(first_brace, first_bracket)
            start_char = "{" if start_idx == first_brace else "["
            end_char = "}" if start_char == "{" else "]"

        # Extract complete JSON object/array by counting braces/brackets
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(content)):
            char = content[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        # Found complete JSON
                        return content[start_idx:i + 1]

        # If we get here, JSON is incomplete - return from start to end
        return content[start_idx:]

    @staticmethod
    def _validate_structure(parsed: Any, client_name: str) -> Dict[str, Any] | list[Any]:
        """
        Validate and normalize response structure.

        Args:
            parsed: Parsed JSON object
            client_name: Name of client (for warnings)

        Returns:
            Validated JSON (dictionary or list)
        """
        # Allow lists to pass through
        if isinstance(parsed, list):
            return parsed

        # Handle other non-dict responses by wrapping
        if not isinstance(parsed, dict):
            rich_print(
                f"[yellow]Warning:[/yellow] {client_name} returned non-dict JSON, wrapping"
            )
            return {"result": parsed}

        # Warn about empty responses
        if not parsed or not any(parsed.values()):
            rich_print(
                f"[yellow]Warning:[/yellow] {client_name} returned empty or all-null JSON"
            )

        return parsed

