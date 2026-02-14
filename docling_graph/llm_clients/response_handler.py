"""
Centralized JSON response parsing and validation for LLM clients.

This module eliminates 150+ lines of duplicated code across all LLM clients
by providing a single, well-tested implementation of JSON parsing, cleaning,
and validation logic.

Enhanced with truncation detection and JSON repair capabilities.
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
        aggressive_clean: bool = False,
        truncated: bool = False,
        max_tokens: int | None = None,
    ) -> Dict[str, Any] | list[Any]:
        """
        Parse and validate JSON response from LLM.

        This is the main entry point used by all LLM clients. It handles
        all common response formats and edge cases, including truncated responses.

        Args:
            raw_response: Raw string response from LLM
            client_name: Name of the client (for error messages)
            aggressive_clean: Apply more aggressive cleaning (for WatsonX, etc.)
            truncated: Whether response was truncated (hit max_tokens limit)
            max_tokens: Max tokens limit (for warning messages)

        Returns:
            Parsed and validated JSON (dictionary or list)

        Raises:
            ClientError: If response cannot be parsed or is invalid
        """
        # Validate input
        if not raw_response or not raw_response.strip():
            raise ClientError(
                f"{client_name} returned empty response", details={"raw_response": raw_response}
            )

        # Clean response
        content = ResponseHandler._clean_response(raw_response, aggressive_clean)

        # Parse JSON
        try:
            parsed = json.loads(content)

            # Success! But warn if truncated
            if truncated:
                ResponseHandler._warn_truncation(client_name, max_tokens, recovered=True)

            return ResponseHandler._validate_structure(parsed, client_name)

        except json.JSONDecodeError as e:
            # Try to repair common JSON syntax errors (missing commas, etc.)
            repaired = ResponseHandler._attempt_json_repair(content)
            if repaired is not None:
                if truncated:
                    ResponseHandler._warn_truncation(client_name, max_tokens, recovered=True)
                return ResponseHandler._validate_structure(repaired, client_name)

            # If truncated and repair failed, show clear truncation error
            if truncated:
                ResponseHandler._warn_truncation(client_name, max_tokens, recovered=False)

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
                    "cleaned_content_preview": content[:200],
                    "truncated": truncated,
                    "max_tokens": max_tokens,
                },
                cause=e,
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

        # Normalize JSON whitespace - this is the critical fix for KeyError issues
        # LLMs often generate JSON with excessive whitespace/newlines that causes
        # Python's json.loads() to raise KeyError with keys like '\n  "slot_id"'
        content = ResponseHandler._normalize_json_whitespace(content)

        return content.strip()

    @staticmethod
    def _normalize_json_whitespace(content: str) -> str:
        """
        Normalize whitespace in JSON to prevent KeyError issues.

        LLMs often generate JSON with excessive whitespace/newlines that causes
        Python's json.loads() to raise KeyError with keys like '\n  "slot_id"'.

        This function normalizes all whitespace outside of string values while
        preserving whitespace inside strings.

        Args:
            content: JSON string with potentially problematic whitespace

        Returns:
            Normalized JSON string safe for json.loads()
        """
        # Strategy: Use a simple state machine to track if we're inside a string
        # and only normalize whitespace when outside strings

        result = []
        in_string = False
        escape_next = False

        for i, char in enumerate(content):
            # Handle escape sequences
            if escape_next:
                result.append(char)
                escape_next = False
                continue

            if char == "\\" and in_string:
                result.append(char)
                escape_next = True
                continue

            # Track string boundaries
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                continue

            # Inside strings: preserve everything
            if in_string:
                result.append(char)
                continue

            # Outside strings: normalize whitespace
            if char in " \t\n\r":
                # Only add a single space if the last char wasn't whitespace
                # and we're not adjacent to structural characters
                if result and result[-1] not in " \t\n\r{[,:}]":
                    # Look ahead to see if next non-whitespace is structural
                    next_idx = i + 1
                    while next_idx < len(content) and content[next_idx] in " \t\n\r":
                        next_idx += 1

                    if next_idx < len(content) and content[next_idx] not in "{}[],:":
                        result.append(" ")
                continue

            # Structural characters and other content: keep as-is
            result.append(char)

        return "".join(result)

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
                content = content[len(prefix) :].strip()

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
                        return content[start_idx : i + 1]

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
            rich_print(f"[yellow]Warning:[/yellow] {client_name} returned non-dict JSON, wrapping")
            return {"result": parsed}

        # Warn about empty responses
        if not parsed or not any(parsed.values()):
            rich_print(f"[yellow]Warning:[/yellow] {client_name} returned empty or all-null JSON")

        return parsed

    @staticmethod
    def _attempt_json_repair(content: str) -> Dict[str, Any] | list[Any] | None:
        """
        Attempt to repair truncated JSON.

        Strategies:
        1. Fix unterminated strings (for small LLMs)
        2. Fix missing commas between objects (for small LLMs)
        3. Truncate at last valid fact (for small LLMs)
        4. Find the last complete object/array before truncation
        5. Close unclosed brackets intelligently
        6. Remove incomplete trailing data

        Args:
            content: Potentially truncated JSON string

        Returns:
            Repaired JSON object/array, or None if unrepairable
        """
        content = content.strip()

        # Strategy 1: Fix unterminated strings (common with small LLMs)
        # Find the last complete object and truncate there
        try:
            # Try to parse to get the specific error
            json.loads(content)
        except json.JSONDecodeError as e:
            if "Unterminated string" in str(e):
                # Find the last complete object and truncate there
                last_complete = content.rfind("},")
                if last_complete > 0:
                    repaired = content[: last_complete + 1] + "\n]}"
                    try:
                        return json.loads(repaired)  # type: ignore[no-any-return]
                    except json.JSONDecodeError:
                        pass

            # Strategy 2: Fix missing commas between objects
            if "Expecting ',' delimiter" in str(e):
                # Add commas between consecutive closing braces
                repaired = re.sub(r"}\s*{", "},{", content)
                try:
                    return json.loads(repaired)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Truncate at last valid fact
        # Find the last complete fact object
        facts_match = re.search(r'"facts":\s*\[(.*)\]', content, re.DOTALL)
        if facts_match:
            facts_content = facts_match.group(1)
            # Find all complete fact objects
            fact_objects = re.findall(r"\{[^{}]*\}", facts_content)
            if fact_objects:
                # Reconstruct with only complete facts
                repaired = content[: facts_match.start(1)] + ",".join(fact_objects) + "]}"
                try:
                    return json.loads(repaired)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Fix common syntax errors (missing commas between fields)
        # This handles cases like: "field1": "value1"\n"field2": "value2"
        # Should be: "field1": "value1",\n"field2": "value2"
        #
        # Also handle missing values before comma/brace (common in small models):
        #   "parent": ,
        #   "tax": }
        # by inserting null as a safe placeholder for validation/retry loops.
        missing_value_fixed = re.sub(r":\s*(?=,|}|])", ": null", content)
        missing_value_fixed = re.sub(r",\s*([}\]])", r"\1", missing_value_fixed)
        if missing_value_fixed != content:
            try:
                result = json.loads(missing_value_fixed)
                return result if isinstance(result, dict | list) else None
            except json.JSONDecodeError:
                pass

        fixed_content = ResponseHandler._fix_missing_commas(content)
        if fixed_content != content:
            try:
                result = json.loads(fixed_content)
                return result if isinstance(result, dict | list) else None
            except json.JSONDecodeError:
                pass  # Continue with other strategies

        # Strategy 5: Try to find last complete structure by removing trailing incomplete data
        # Look for common truncation patterns
        truncation_patterns = [
            r',\s*"[^"]*$',  # Incomplete key: , "partial_key
            r':\s*"[^"]*$',  # Incomplete string value: : "partial_value
            r":\s*\d+\.?\d*$",  # Incomplete number: : 123.
            r",\s*$",  # Trailing comma
            r":\s*$",  # Trailing colon
        ]

        for pattern in truncation_patterns:
            cleaned = re.sub(pattern, "", content)
            if cleaned != content:
                # Try closing brackets
                repaired = ResponseHandler._close_brackets(cleaned)
                try:
                    result = json.loads(repaired)
                    return result if isinstance(result, dict | list) else None
                except json.JSONDecodeError:
                    continue

        # Strategy 6: Try to close brackets on original content
        repaired = ResponseHandler._close_brackets(content)
        try:
            result = json.loads(repaired)
            return result if isinstance(result, dict | list) else None
        except json.JSONDecodeError:
            pass

        # Strategy 7: Find last complete array/object element
        # For arrays: find last complete element before truncation
        if content.strip().startswith("["):
            last_complete_str = ResponseHandler._find_last_complete_array_element(content)
            if last_complete_str:
                try:
                    result = json.loads(last_complete_str)
                    return result if isinstance(result, dict | list) else None
                except json.JSONDecodeError:
                    pass

        # For objects: find last complete key-value pair
        if content.strip().startswith("{"):
            last_complete_str = ResponseHandler._find_last_complete_object(content)
            if last_complete_str:
                try:
                    result = json.loads(last_complete_str)
                    return result if isinstance(result, dict | list) else None
                except json.JSONDecodeError:
                    pass

        # Unable to repair
        return None

    @staticmethod
    def _fix_missing_commas(content: str) -> str:
        """
        Fix missing commas between JSON object fields.

        Common error pattern from LLMs:
        {
          "field1": "value1"
          "field2": "value2"  <- Missing comma after field1
        }

        Args:
            content: JSON string with potential missing commas

        Returns:
            Fixed JSON string with commas added where needed
        """
        # More conservative approach: only fix obvious missing commas
        # Pattern: Match end of a value followed by whitespace and a new field name
        # But be careful not to match within strings or after commas

        # First, try to parse as-is to see if it's already valid
        try:
            json.loads(content)
            return content  # Already valid, don't modify
        except (json.JSONDecodeError, KeyError) as e:
            # KeyError can happen if JSON has malformed keys with newlines
            # If we get a KeyError, apply normalization and try again
            if isinstance(e, KeyError):
                content = ResponseHandler._normalize_json_whitespace(content)
                try:
                    json.loads(content)
                    return content  # Normalization fixed it
                except (json.JSONDecodeError, KeyError):
                    pass  # Continue with other fixes
            # Continue with fixes for JSONDecodeError

        # Pattern: Match end of a value (string, number, boolean, null, object, array)
        # followed by whitespace and a new field name, without a comma
        # Use negative lookbehind to ensure we're not already after a comma
        patterns = [
            # After string value: "value"\n"field" (but not after comma)
            (r'(?<!,)("\s*)\n(\s*"[^"]+"\s*:)', r"\1,\n\2"),
            # After number value: 123\n"field" (but not after comma)
            (r'(?<!,)(\d+\.?\d*\s*)\n(\s*"[^"]+"\s*:)', r"\1,\n\2"),
            # After boolean/null: true\n"field" (but not after comma)
            (r'(?<!,)((?:true|false|null)\s*)\n(\s*"[^"]+"\s*:)', r"\1,\n\2"),
            # After closing brace: }\n"field" (but not after comma)
            (r'(?<!,)(}\s*)\n(\s*"[^"]+"\s*:)', r"\1,\n\2"),
            # After closing bracket: ]\n"field" (but not after comma)
            (r'(?<!,)(\]\s*)\n(\s*"[^"]+"\s*:)', r"\1,\n\2"),
        ]

        fixed = content
        for pattern, replacement in patterns:
            fixed = re.sub(pattern, replacement, fixed)

        return fixed

    @staticmethod
    def _close_brackets(content: str) -> str:
        """
        Intelligently close unclosed brackets in JSON.

        Args:
            content: JSON string with potentially unclosed brackets

        Returns:
            JSON string with brackets closed
        """
        # Count open/close brackets
        content.count("{")
        content.count("}")
        content.count("[")
        content.count("]")

        # Track what's open (accounting for strings)
        in_string = False
        escape_next = False
        stack = []

        for char in content:
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
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char == "}" and stack and stack[-1] == "}":
                    stack.pop()
                elif char == "]" and stack and stack[-1] == "]":
                    stack.pop()

        # Close remaining open structures
        return content + "".join(reversed(stack))

    @staticmethod
    def _find_last_complete_array_element(content: str) -> str | None:
        """
        Find the last complete element in a truncated array.

        Args:
            content: Truncated array JSON

        Returns:
            Array with last complete elements, or None
        """
        # Find all complete elements by tracking depth
        elements = []
        depth = 0
        in_string = False
        escape_next = False
        current_start = None

        for i, char in enumerate(content):
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
                if char in "{[":
                    if depth == 1 and current_start is None:
                        current_start = i
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 1 and current_start is not None:
                        # Complete element found
                        elements.append(content[current_start : i + 1])
                        current_start = None
                elif char == "," and depth == 1:
                    current_start = None

        if elements:
            return "[" + ",".join(elements) + "]"
        return None

    @staticmethod
    def _find_last_complete_object(content: str) -> str | None:
        """
        Find the last complete key-value pairs in a truncated object.

        Args:
            content: Truncated object JSON

        Returns:
            Object with last complete pairs, or None
        """
        # Similar to array but for objects
        # Find complete "key": value pairs
        pairs = []
        depth = 0
        in_string = False
        escape_next = False
        current_start = None

        for i, char in enumerate(content):
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
                if char in "{[":
                    if depth == 1 and current_start is None:
                        # Start of a value
                        current_start = i
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 1 and current_start is not None:
                        # Complete value found
                        pairs.append(content[current_start : i + 1])
                        current_start = None
                elif char == "," and depth == 1:
                    if current_start is not None:
                        pairs.append(content[current_start:i])
                    current_start = None

        if pairs:
            return "{" + ",".join(pairs) + "}"
        return None

    @staticmethod
    def _warn_truncation(client_name: str, max_tokens: int | None, recovered: bool) -> None:
        """
        Display clear warning about response truncation.

        Args:
            client_name: Name of the LLM client
            max_tokens: Maximum tokens limit that was hit
            recovered: Whether partial data was successfully recovered
        """
        max_tokens_str = str(max_tokens) if max_tokens else "unknown"

        if recovered:
            rich_print(f"[yellow]Response Truncated[/yellow] (hit max_tokens={max_tokens_str})")
            rich_print("[yellow]Partial data recovered - results may be incomplete[/yellow]")
        else:
            rich_print(f"[red]Response Truncated[/red] (hit max_tokens={max_tokens_str})")
            rich_print("[red]Unable to recover partial data - JSON too incomplete[/red]")
