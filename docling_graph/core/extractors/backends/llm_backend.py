"""
LLM (Language Model) extraction backend.

Performs direct full-document extraction via extract_from_markdown() in a single LLM call.
Best-effort: coerces QuantityWithUnit scalars and prunes invalid fields on validation errors.
"""

import copy
import gc
import json
import logging
import re
from functools import lru_cache
from typing import Any, Type

from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from ....protocols import LLMClientProtocol
from ..contracts import direct

logger = logging.getLogger(__name__)


class LlmBackend:
    """
    Backend for LLM-based extraction.

    Performs direct full-document extraction via extract_from_markdown().
    """

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        """
        Initialize LLM backend with a client.

        Args:
            llm_client: LLM client instance implementing LLMClientProtocol
        """
        self.client = llm_client

        # Get model identifier for logging
        model_attr = getattr(llm_client, "model", None) or getattr(llm_client, "model_id", None)

        logger.info("Initialized LlmBackend with client: %s", self.client.__class__.__name__)

        rich_print(
            f"[blue][LlmBackend][/blue] Initialized with:\n"
            f"  • Client: [cyan]{self.client.__class__.__name__}[/cyan]\n"
            f"  • Model: [cyan]{model_attr or 'unknown'}[/cyan]"
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_schema_json(template: Type[BaseModel]) -> str:
        """
        Get cached JSON schema for a Pydantic template.

        Uses LRU cache to avoid repeated serialization of the same template.
        This provides significant performance improvement when the same template
        is used multiple times.

        Args:
            template: Pydantic model class

        Returns:
            JSON string representation of the model schema
        """
        return json.dumps(template.model_json_schema(), indent=2)

    def _log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with consistent formatting."""
        formatted = f"[blue][LlmBackend][/blue] {message}"
        if kwargs:
            for key, value in kwargs.items():
                formatted += f" ([cyan]{value}[/cyan] {key})"
        rich_print(formatted)

    def _log_success(self, message: str) -> None:
        """Log success message."""
        rich_print(f"[blue][LlmBackend][/blue] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        rich_print(f"[yellow]Warning:[/yellow] {message}")

    def _log_error(self, message: str, exception: Exception | None = None) -> None:
        """Log error message with optional exception details."""
        error_text = f"[red]Error:[/red] {message}"
        if exception:
            error_text += f" {type(exception).__name__}: {exception}"
        rich_print(error_text)

    def _log_validation_error(
        self, context: str, error: ValidationError, raw_data: dict | list
    ) -> None:
        """Log detailed validation error information."""
        rich_print(f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]")
        rich_print("  The data extracted by the LLM does not match your Pydantic template.")
        rich_print("[red]Details:[/red]")
        for err in error.errors():
            loc = " -> ".join(map(str, err["loc"]))
            rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{err['msg']}[/red]")
        rich_print(f"\n[yellow]Extracted Data (raw):[/yellow]\n{raw_data}\n")

    @staticmethod
    def _is_quantity_with_unit_error(err: dict) -> bool:
        """True if this validation error is for a QuantityWithUnit expected type."""
        ctx = err.get("ctx") or {}
        if isinstance(ctx, dict) and ctx.get("class_name") == "QuantityWithUnit":
            return True
        msg = err.get("msg", "")
        return "QuantityWithUnit" in msg

    @staticmethod
    def _coerce_scalar_to_quantity_with_unit(v: Any) -> dict:
        """Coerce a scalar to a QuantityWithUnit-like object."""
        if isinstance(v, int | float):
            return {"numeric_value": float(v)}
        if isinstance(v, str):
            v_clean = re.sub(r"[^\d.\-eE]", "", v)
            try:
                return {"numeric_value": float(v_clean)}
            except ValueError:
                return {"text_value": v}
        return {"numeric_value": None, "text_value": str(v)}

    @staticmethod
    def _get_at_path(data: dict | list, loc: tuple) -> Any:
        """Get value at path (loc is tuple of keys/indices)."""
        if not loc:
            return data
        current: Any = data
        for key in loc:
            current = current[key]
        return current

    @staticmethod
    def _set_at_path(data: dict | list, loc: tuple, value: Any) -> None:
        """Set value at path (mutates data)."""
        if not loc:
            return
        parent = LlmBackend._get_at_path(data, loc[:-1])
        if parent is not None:
            parent[loc[-1]] = value

    @staticmethod
    def _delete_at_path(data: dict | list, loc: tuple) -> None:
        """Remove the leaf at loc (mutates data)."""
        if not loc:
            return
        parent = LlmBackend._get_at_path(data, loc[:-1])
        if parent is None:
            return
        leaf = loc[-1]
        if isinstance(parent, dict):
            parent.pop(leaf, None)
        elif isinstance(parent, list) and isinstance(leaf, int) and 0 <= leaf < len(parent):
            parent.pop(leaf)

    def _apply_quantity_coercion(self, data: dict | list, errors: list) -> bool:
        """
        Coerce scalar values at QuantityWithUnit error locations.
        Returns True if any coercion was applied.
        """
        changed = False
        for err in errors:
            if not self._is_quantity_with_unit_error(err):
                continue
            loc = tuple(err.get("loc", ()))
            if not loc:
                continue
            try:
                value = self._get_at_path(data, loc)
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(value, dict):
                continue
            coerced = self._coerce_scalar_to_quantity_with_unit(value)
            self._set_at_path(data, loc, coerced)
            changed = True
        return changed

    def _prune_invalid_fields(self, data: dict | list, errors: list) -> None:
        """
        Remove offending leaf fields indicated by validation error locs.
        Mutates data in place. For list element errors, removes the element.
        """
        # Sort by loc length descending so we prune deepest first (avoid index shift)
        sorted_errors = sorted(errors, key=lambda e: len(e.get("loc", ())), reverse=True)
        seen_locs: set[tuple] = set()
        for err in sorted_errors:
            loc = tuple(err.get("loc", ()))
            if not loc or loc in seen_locs:
                continue
            seen_locs.add(loc)
            self._delete_at_path(data, loc)

    def _validate_extraction(
        self, parsed_json: dict | list, template: Type[BaseModel], context: str
    ) -> BaseModel | None:
        """
        Validate parsed JSON against Pydantic template.

        Best-effort: on ValidationError, tries (1) QuantityWithUnit coercion,
        then (2) pruning invalid fields, then re-validates (up to 3 passes).
        """
        data: dict | list = copy.deepcopy(parsed_json)
        max_salvage_passes = 3

        for pass_num in range(max_salvage_passes):
            try:
                validated_model = template.model_validate(data)
                if pass_num > 0:
                    self._log_warning(
                        f"Extraction validated after best-effort salvage (pass {pass_num + 1})"
                    )
                self._log_success(f"Successfully extracted data from {context}")
                return validated_model
            except ValidationError as e:
                if pass_num == 0:
                    self._log_validation_error(context, e, parsed_json)

                # First pass: try QuantityWithUnit coercion
                if pass_num == 0:
                    if self._apply_quantity_coercion(data, e.errors()):
                        continue

                # Prune invalid fields and retry
                self._prune_invalid_fields(data, e.errors())

        self._log_warning("Validation failed after best-effort salvage")
        return None

    def _call_llm_for_extraction(
        self, markdown: str, schema_json: str, is_partial: bool, context: str
    ) -> dict | list | None:
        """
        Call LLM and return parsed JSON or None on failure.

        Args:
            markdown: Markdown content to extract from
            schema_json: JSON schema string
            is_partial: Whether this is partial extraction
            context: Context description for logging

        Returns:
            Parsed JSON (dict or list) or None if call failed
        """
        try:
            prompt = direct.get_extraction_prompt(
                markdown_content=markdown,
                schema_json=schema_json,
                is_partial=is_partial,
                model_config=None,  # No capability-based branching
            )

            parsed_json = self.client.get_json_response(prompt=prompt, schema_json=schema_json)

            if not parsed_json:
                self._log_warning(f"No valid JSON returned from LLM for {context}")
                return None

            return parsed_json

        except Exception as e:
            self._log_error(f"Error during LLM call for {context}", e)
            return None

    def _repair_json(self, raw_text: str) -> str:
        """
        Repair common JSON malformations from small LLMs.

        Applies the following fixes:
        1. Remove invalid control characters (except newlines, tabs, carriage returns)
        2. Remove trailing commas before closing brackets/braces
        3. Balance unmatched braces and brackets

        Args:
            raw_text: Raw JSON text from LLM

        Returns:
            Repaired JSON text

        Examples:
            >>> backend._repair_json('{"key": "value",}')
            '{"key": "value"}'
            >>> backend._repair_json('{"key": "value"')
            '{"key": "value"}'
        """
        # Step 1: Remove invalid control characters (keep \n, \t, \r)
        # Remove control chars in range 0x00-0x1F except \n (0x0A), \t (0x09), \r (0x0D)
        repaired = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", raw_text)

        # Step 2: Remove trailing commas before closing brackets
        # Match comma followed by optional whitespace and closing bracket/brace
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        # Step 3: Balance unmatched braces and brackets
        # Count opening and closing braces/brackets
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")

        # Add missing closing braces
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        # Add missing closing brackets
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)

        # Remove extra closing braces (trim from end)
        if close_braces > open_braces:
            excess = close_braces - open_braces
            # Remove excess closing braces from the end
            for _ in range(excess):
                repaired = repaired.rstrip()
                if repaired.endswith("}"):
                    repaired = repaired[:-1]

        # Remove extra closing brackets (trim from end)
        if close_brackets > open_brackets:
            excess = close_brackets - open_brackets
            # Remove excess closing brackets from the end
            for _ in range(excess):
                repaired = repaired.rstrip()
                if repaired.endswith("]"):
                    repaired = repaired[:-1]

        return repaired

    def extract_from_markdown(
        self,
        markdown: str,
        template: Type[BaseModel],
        context: str = "document",
        is_partial: bool = False,
    ) -> BaseModel | None:
        """
        Extract structured data from markdown content (direct mode).

        This is the "power user" mode that attempts full-document extraction
        in a single LLM call. Best-effort only, no retries or fallbacks.

        Args:
            markdown: Markdown content to extract from
            template: Pydantic model template
            context: Context description (e.g., "page 1", "full document")
            is_partial: If True, use partial/chunk-based prompt

        Returns:
            Extracted and validated Pydantic model instance, or None if failed
        """
        # Log extraction start
        self._log_info(f"Direct extraction from {context}", chars=len(markdown))

        # Early validation for empty markdown
        if not markdown or len(markdown.strip()) == 0:
            self._log_error(f"Markdown is empty for {context}. Cannot proceed.")
            return None

        # Get cached schema JSON
        schema_json = self._get_schema_json(template)

        # Call LLM
        parsed_json = self._call_llm_for_extraction(
            markdown=markdown,
            schema_json=schema_json,
            is_partial=is_partial,
            context=context,
        )

        if not parsed_json:
            return None

        # Validate and return
        return self._validate_extraction(parsed_json, template, context)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> Any:
        """
        Generate a text response from the LLM for consolidation.

        This method is used by LLM consolidation to generate patches.
        It returns a simple response object with a 'text' attribute.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Response object with 'text' attribute containing the LLM's response
        """
        # Build prompt dictionary
        prompt = {
            "system": system_prompt,
            "user": user_prompt,
        }

        # Call the LLM client
        # Note: We're using get_json_response but the consolidation prompt
        # should guide the LLM to return JSON format
        try:
            response = self.client.get_json_response(
                prompt=prompt,
                schema_json="{}",  # Empty schema for free-form response
            )

            # Wrap response in an object with 'text' attribute
            class Response:
                def __init__(self, data: Any) -> None:
                    if isinstance(data, dict):
                        self.text = json.dumps(data)
                    elif isinstance(data, str):
                        self.text = data
                    else:
                        self.text = str(data)

            return Response(response)

        except Exception as e:
            rich_print(f"[blue][LlmBackend][/blue] [red]Error in generate:[/red] {e}")

            # Return empty response on error
            class EmptyResponse:
                text = "{}"

            return EmptyResponse()

    def cleanup(self) -> None:
        """
        Clean up LLM client resources.

        Note: Most LLM clients use stateless HTTP APIs and don't require cleanup.
        This method is provided for consistency with VlmBackend and handles any
        clients that may have cleanup methods.
        """
        try:
            # Release the client reference
            if hasattr(self, "client"):
                # If the client has its own cleanup method, call it
                # Use getattr to avoid type checker issues with protocol
                cleanup_fn = getattr(self.client, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
                del self.client

            # Force garbage collection
            gc.collect()

            rich_print("[blue][LlmBackend][/blue] [green]Cleaned up resources[/green]")

        except Exception as e:
            rich_print(f"[blue][LlmBackend][/blue] [yellow]Warning during cleanup:[/yellow] {e}")
