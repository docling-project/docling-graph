"""
LLM (Language Model) extraction backend.

Performs direct full-document extraction via extract_from_markdown() in a single LLM call.
Best-effort: coerces QuantityWithUnit scalars and prunes invalid fields on validation errors.
"""

import ast
import copy
import gc
import hashlib
import json
import logging
import re
from functools import lru_cache
from typing import Any, Literal, Type, cast

from pydantic import BaseModel, ValidationError
from rich import print as rich_print

from ....exceptions import ClientError
from ....protocols import LLMClientProtocol
from ..contracts import direct
from ..contracts.delta.backend_ops import run_delta_orchestrator
from ..contracts.staged.backend_ops import run_staged_orchestrator
from ..gleaning import merge_gleaned_direct, run_gleaning_pass_direct

logger = logging.getLogger(__name__)


class LlmBackend:
    """
    Backend for LLM-based extraction.

    Performs direct full-document extraction via extract_from_markdown().
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        extraction_contract: Literal["direct", "staged", "delta"] = "direct",
        staged_config: dict[str, Any] | None = None,
        structured_output: bool = True,
        structured_sparse_check: bool = True,
    ) -> None:
        """
        Initialize LLM backend with a client.

        Args:
            llm_client: LLM client instance implementing LLMClientProtocol
        """
        self.client = llm_client
        self.extraction_contract: Literal["direct", "staged", "delta"] = extraction_contract
        self._staged_config_raw = staged_config or {}
        self.structured_output = structured_output
        self.structured_sparse_check = structured_sparse_check
        self.trace_data: Any = None  # Set by strategy when config.debug is True
        self.last_call_diagnostics: dict[str, Any] = {}
        self._retry_on_truncation = bool(self._staged_config_raw.get("retry_on_truncation", True))
        self._truncation_retry_multiplier = max(
            1.0,
            float(self._staged_config_raw.get("truncation_retry_max_tokens_multiplier", 2.0)),
        )
        self._truncation_retry_cap = 32768

        # Get model identifier for logging
        model_attr = getattr(llm_client, "model", None) or getattr(llm_client, "model_id", None)

        logger.info("Initialized LlmBackend with client: %s", self.client.__class__.__name__)

        rich_print(
            f"[yellow][LlmBackend][/yellow] Initialized with:\n"
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

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_schema_dict(template: Type[BaseModel]) -> dict[str, Any]:
        """Get cached schema dict for a Pydantic template."""
        schema = template.model_json_schema()
        return schema if isinstance(schema, dict) else {}

    def _log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message with consistent formatting."""
        formatted = f"[blue][LlmBackend][/blue] {message}"
        if kwargs:
            for key, value in kwargs.items():
                formatted += f" ([cyan]{value}[/cyan] {key})"
        rich_print(formatted)

    def _log_extraction(self, message: str, prefix: str) -> None:
        """Log extraction-phase message with contract-specific prefix (e.g. DirectExtraction)."""
        rich_print(f"[blue][{prefix}][/blue] {message}")

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
        """Log detailed validation error information. Raw data is stored in trace_data when present."""
        rich_print(f"[blue][LlmBackend][/blue] [yellow]Validation Error for {context}:[/yellow]")
        rich_print("  The data extracted by the LLM does not match your Pydantic template.")
        rich_print("[red]Details:[/red]")
        for err in error.errors():
            loc = " -> ".join(map(str, err["loc"]))
            rich_print(f"  - [bold magenta]{loc}[/bold magenta]: [red]{err['msg']}[/red]")
        if self.trace_data is not None:
            self.trace_data.emit(
                "validation_error_raw_data",
                "extraction",
                {"context": context, "raw_data": raw_data},
            )

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

    def _content_fingerprint(self, entity: dict, exclude_keys: set[str] | None = None) -> str:
        """Stable hash of entity content for deterministic synthetic IDs. Excludes given keys."""
        exclude = (exclude_keys or set()) | {"__class__"}
        stable = {k: v for k, v in entity.items() if k not in exclude}
        content = json.dumps(stable, sort_keys=True, default=str)
        return hashlib.blake2b(content.encode(), digest_size=8).hexdigest()

    @staticmethod
    def _resolve_schema_ref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
        """Resolve $ref in schema node; return resolved dict or node."""
        ref = node.get("$ref")
        if isinstance(ref, str) and (ref.startswith(("#/$defs/", "#/definitions/"))):
            return cast(dict[str, Any], defs.get(ref.split("/")[-1], node))
        return node

    @staticmethod
    def _schema_node_properties_or_any_of(
        node: dict[str, Any], defs: dict[str, Any]
    ) -> dict[str, Any]:
        """Return properties from node, or from first anyOf member if node uses anyOf."""
        props = node.get("properties")
        if isinstance(props, dict):
            return props
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and any_of:
            first = LlmBackend._resolve_schema_ref(any_of[0], defs)
            if isinstance(first, dict):
                return first.get("properties") or {}
        return {}

    @staticmethod
    def _get_field_schema_at_path(
        template: type[BaseModel], loc: tuple, defs: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Resolve JSON schema for the field at the given path (e.g. for enum default).
        Walks root schema: properties -> key, array -> items, $ref -> $defs.
        Handles anyOf (e.g. optional nested object) by using first branch for properties.
        """
        try:
            schema = template.model_json_schema()
        except Exception:
            return None
        root_defs = schema.get("$defs") or schema.get("definitions")
        defs = defs if defs is not None else (root_defs if isinstance(root_defs, dict) else {})
        node: dict[str, Any] = schema
        for key in loc:
            node = LlmBackend._resolve_schema_ref(node, defs)
            if isinstance(key, int):
                node = node.get("items") or {}
            else:
                props = LlmBackend._schema_node_properties_or_any_of(node, defs)
                node = props.get(key) or {}
            if not node:
                return None
        node = LlmBackend._resolve_schema_ref(node, defs)
        return node if isinstance(node, dict) else None

    @staticmethod
    def _enum_default_from_schema(field_schema: dict[str, Any]) -> Any:
        """Return a valid enum value from schema: prefer OTHER, else first."""
        enum_vals = field_schema.get("enum")
        if not isinstance(enum_vals, list) or not enum_vals:
            return None
        for v in enum_vals:
            if isinstance(v, str) and v.upper() == "OTHER":
                return v
        return enum_vals[0]

    def _fill_missing_required_fields(
        self,
        data: dict | list,
        errors: list,
        template: type[BaseModel] | None = None,
    ) -> bool:
        """
        Fill missing required fields with deterministic or template-derived values so
        validation can succeed when the LLM omits identity fields. Root-level
        "document identifier" fields (by naming convention) use template name when
        template is provided. Enum fields use schema enum default (e.g. OTHER).
        Returns True if any value was set.
        """
        changed = False
        missing_errors = [e for e in errors if e.get("type") == "missing"]
        sorted_errors = sorted(missing_errors, key=lambda e: len(e.get("loc", ())))
        seen_locs: set[tuple] = set()
        for err in sorted_errors:
            loc = tuple(err.get("loc", ()))
            if not loc or loc in seen_locs:
                continue
            field_name = loc[-1] if isinstance(loc[-1], str) else None
            if not field_name:
                continue
            try:
                parent = self._get_at_path(data, loc[:-1])
                if not (parent is not None and isinstance(parent, dict) and loc[-1] not in parent):
                    continue
            except (KeyError, IndexError, TypeError):
                continue
            # Root-level "document identifier" field (convention: name contains reference/document): use template name
            is_root = len(loc) == 1
            fn_lower = (field_name or "").lower()
            is_doc_id = ("reference" in fn_lower and "document" in fn_lower) or fn_lower.endswith(
                "_document"
            )
            if template is not None and is_root and is_doc_id:
                value = getattr(template, "__name__", None) or "Document"
            elif template is not None:
                # Enum-aware: use schema enum default (e.g. OTHER) so validation passes
                field_schema = self._get_field_schema_at_path(template, loc)
                if field_schema:
                    enum_default = self._enum_default_from_schema(field_schema)
                    if enum_default is not None:
                        value = enum_default
                    else:
                        # Identity-like fields (e.g. *_id) get stable generated IDs; other strings get ""
                        fingerprint = self._content_fingerprint(parent, exclude_keys={field_name})
                        if field_name.endswith("_id"):
                            prefix = field_name[:-3].upper()
                            prefix = prefix[:4] if len(prefix) > 4 else prefix
                            value = f"{prefix}-{fingerprint}"
                        else:
                            value = ""
                else:
                    fingerprint = self._content_fingerprint(parent, exclude_keys={field_name})
                    if field_name.endswith("_id"):
                        prefix = field_name[:-3].upper()
                        prefix = prefix[:4] if len(prefix) > 4 else prefix
                        value = f"{prefix}-{fingerprint}"
                    else:
                        value = ""
            else:
                # No template: identity-like fields get generated ID; other strings get ""
                fingerprint = self._content_fingerprint(parent, exclude_keys={field_name})
                if field_name.endswith("_id"):
                    prefix = field_name[:-3].upper()
                    prefix = prefix[:4] if len(prefix) > 4 else prefix
                    value = f"{prefix}-{fingerprint}"
                else:
                    value = ""
            self._set_at_path(data, loc, value)
            seen_locs.add(loc)
            changed = True
        return changed

    # Keys that indicate a dict is a guarantee/condition block (structural content),
    # not a simple label. Do not use such dicts' nom/name for string coercion
    # (avoids "Vol" as offer name). Excludes "description" so simple description+nom
    # still coerces; "conditions"/"texte" etc. mark full blocks.
    _COMPLEX_DICT_HINTS: frozenset[str] = frozenset(
        ("conditions", "texte", "exclusions_specifiques", "biens_couverts")
    )

    @staticmethod
    def _looks_like_complex_block(d: dict) -> bool:
        """True if dict looks like a guarantee/condition block, not a simple label."""
        if not isinstance(d, dict) or len(d) <= 1:
            return False
        hints = LlmBackend._COMPLEX_DICT_HINTS
        return bool(hints & set(d))

    @classmethod
    def _extract_string_from_list_or_dict(cls, value: Any) -> str | None:
        """
        Extract a single string from a list or dict when schema expected string.
        Domain-agnostic: uses common identity-like keys and first string element.
        Skips dicts that look like guarantee/condition blocks (description, conditions, etc.)
        so their inner nom is not used for a parent field (e.g. offer name).
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int | float | bool):
            return str(value)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    if cls._looks_like_complex_block(item):
                        continue
                    for key in ("nom", "name", "title", "id", "label"):
                        if key in item and item[key] is not None:
                            s = item[key]
                            if isinstance(s, str) and s.strip():
                                return s.strip()
                            if isinstance(s, int | float | bool):
                                return str(s)
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            return None
        if isinstance(value, dict):
            if cls._looks_like_complex_block(value):
                return None
            for key in ("nom", "name", "title", "id", "label"):
                if key in value and value[key] is not None:
                    s = value[key]
                    if isinstance(s, str) and s.strip():
                        return s.strip()
                    if isinstance(s, int | float | bool):
                        return str(s)
            for v in value.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None
        return None

    def _coerce_string_type_errors(self, data: dict | list, errors: list) -> bool:
        """
        Coerce values to string when validation expected string but got int/float/bool/list/dict.
        Returns True if any value was coerced (caller should retry validation).
        Preserves list items by extracting a string from list/dict instead of pruning.
        """
        # Pydantic v2: "string_type" = expected string, got other; "int_type" etc. = wrong scalar type
        coercible_types = ("int_type", "float_type", "bool_type", "string_type")
        changed = False
        seen_locs: set[tuple] = set()
        for err in errors:
            err_type = err.get("type")
            if err_type not in coercible_types:
                continue
            loc = tuple(err.get("loc", ()))
            if not loc or loc in seen_locs:
                continue
            try:
                value = self._get_at_path(data, loc)
            except (KeyError, IndexError, TypeError):
                continue
            if value is None:
                continue
            coerced: str | None = None
            if isinstance(value, int | float | bool):
                coerced = str(value)
            elif isinstance(value, list | dict):
                coerced = self._extract_string_from_list_or_dict(value)
                if coerced is None:
                    coerced = ""
            if coerced is None:
                continue
            try:
                self._set_at_path(data, loc, coerced)
                seen_locs.add(loc)
                changed = True
            except (KeyError, IndexError, TypeError):
                continue
        return changed

    def _coerce_list_type_errors(self, data: dict | list, errors: list) -> bool:
        """
        Coerce scalar values to single-element list when validation expected list (e.g. list[str])
        but got string or other scalar (Pydantic v2 type=list_type).
        Returns True if any value was coerced (caller should retry validation).
        """
        changed = False
        seen_locs: set[tuple] = set()
        for err in errors:
            if err.get("type") != "list_type":
                continue
            loc = tuple(err.get("loc", ()))
            if not loc or loc in seen_locs:
                continue
            try:
                value = self._get_at_path(data, loc)
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(value, list):
                continue
            try:
                # String that looks like a Python list literal (e.g. "['locataire']")
                if isinstance(value, str):
                    s = value.strip()
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, list):
                                list_value = list(parsed)
                                self._set_at_path(data, loc, list_value)
                                seen_locs.add(loc)
                                changed = True
                                continue
                        except (ValueError, SyntaxError):
                            pass
                    if "," in value:
                        list_value = [s.strip() for s in value.split(",") if s.strip()]
                    else:
                        list_value = [value]
                else:
                    list_value = [value]
                self._set_at_path(data, loc, list_value)
                seen_locs.add(loc)
                changed = True
            except (KeyError, IndexError, TypeError):
                continue
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
        (2) filling missing required fields with generated values,
        (3) coercing int/float/bool to string where string is expected,
        (4) coercing scalar to single-element list where list is expected,
        (5) pruning invalid fields, then re-validates (up to 3 passes).
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

                errors = e.errors()
                any_fixed = False

                # First pass: try QuantityWithUnit coercion
                if pass_num == 0 and self._apply_quantity_coercion(data, errors):
                    any_fixed = True

                # Fill missing required fields with enum or generated values
                if self._fill_missing_required_fields(data, errors, template=template):
                    any_fixed = True

                # Coerce int/float/bool to string where schema expects string
                if self._coerce_string_type_errors(data, errors):
                    any_fixed = True

                # Coerce scalar to list where schema expects list (comma-split when string)
                if self._coerce_list_type_errors(data, errors):
                    any_fixed = True

                if any_fixed:
                    continue

                # Prune invalid fields and retry
                self._prune_invalid_fields(data, errors)

        self._log_warning("Validation failed after best-effort salvage")
        return None

    @staticmethod
    def _count_non_empty_values(value: Any) -> int:
        """Count non-empty populated values recursively."""
        if value is None:
            return 0
        if isinstance(value, str):
            return 1 if value.strip() else 0
        if isinstance(value, int | float | bool):
            return 1
        if isinstance(value, list):
            return sum(LlmBackend._count_non_empty_values(v) for v in value)
        if isinstance(value, dict):
            return sum(LlmBackend._count_non_empty_values(v) for v in value.values())
        return 1

    @staticmethod
    def _count_schema_leaf_fields(schema: dict[str, Any]) -> int:
        """Approximate number of schema leaf fields for sparsity checks."""
        _defs = schema.get("$defs")
        defs: dict[str, Any] = _defs if isinstance(_defs, dict) else {}

        def _resolve(node: dict[str, Any]) -> dict[str, Any]:
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                resolved = defs.get(key)
                if isinstance(resolved, dict):
                    return resolved
            return node

        def _walk(node: dict[str, Any], depth: int) -> int:
            if depth > 6:
                return 0
            node = _resolve(node)
            props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
            if not props:
                return 1
            total = 0
            for raw in props.values():
                if not isinstance(raw, dict):
                    continue
                item = _resolve(raw)
                if item.get("type") == "array" and isinstance(item.get("items"), dict):
                    total += _walk(item["items"], depth + 1)
                else:
                    total += _walk(item, depth + 1)
            return max(total, 1)

        return _walk(schema, 1)

    def _is_sparse_structured_result(
        self, parsed_json: dict | list, schema_dict: dict[str, Any], markdown: str
    ) -> bool:
        """Detect obvious structured-output under-extraction before graph conversion."""
        if len(markdown) < 400:
            return False
        non_empty = self._count_non_empty_values(parsed_json)
        schema_leafs = self._count_schema_leaf_fields(schema_dict)
        if schema_leafs < 10:
            return False
        ratio = non_empty / max(schema_leafs, 1)
        return ratio < 0.40

    def _call_llm_for_extraction(
        self,
        markdown: str,
        schema_json: str,
        schema_dict: dict[str, Any],
        is_partial: bool,
        context: str,
        template: Type[BaseModel] | None = None,
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
            if self.extraction_contract == "delta" and not is_partial:
                delta_result = self._extract_with_delta_contract(
                    markdown=markdown,
                    context=context,
                    template=template,
                    trace_data=getattr(self, "trace_data", None),
                )
                if delta_result:
                    return delta_result
                self._log_warning(
                    f"Delta extraction produced no JSON for {context}; falling back to direct."
                )

            if self.extraction_contract == "staged" and not is_partial:
                staged_result = self._extract_with_staged_contract(
                    markdown=markdown,
                    schema_json=schema_json,
                    context=context,
                    template=template,
                    trace_data=getattr(self, "trace_data", None),
                )
                if staged_result:
                    return staged_result
                self._log_warning(
                    f"Staged extraction produced no JSON for {context}; falling back to direct."
                )

            prompt = direct.get_extraction_prompt(
                markdown_content=markdown,
                schema_json=schema_json,
                is_partial=is_partial,
                model_config=None,  # No capability-based branching
                structured_output=self.structured_output,
                schema_dict=schema_dict,
            )
            self.last_call_diagnostics = {
                "structured_attempted": bool(self.structured_output),
                "structured_failed": False,
                "fallback_used": False,
                "fallback_error_class": None,
            }
            structured_primary_attempt_parsed_json: dict | list | None = None
            structured_primary_attempt_raw: str | None = None
            try:
                parsed_json = self.client.get_json_response(
                    prompt=prompt,
                    schema_json=schema_json,
                    structured_output=self.structured_output,
                    response_top_level="object",
                    response_schema_name="direct_extraction",
                )
                if self.structured_output and self.trace_data is not None:
                    structured_primary_attempt_parsed_json = copy.deepcopy(parsed_json)
                    primary_diag = getattr(self.client, "last_call_diagnostics", None)
                    if isinstance(primary_diag, dict):
                        raw_value = primary_diag.get("raw_response")
                        if isinstance(raw_value, str):
                            structured_primary_attempt_raw = raw_value
            except ClientError as e:
                if not self.structured_output:
                    raise
                if self.trace_data is not None:
                    self.trace_data.emit(
                        "structured_output_fallback_triggered",
                        "extraction",
                        {
                            "context": context,
                            "reason": type(e).__name__,
                            "provider": getattr(self.client, "provider", "unknown"),
                            "model": getattr(
                                self.client, "model", getattr(self.client, "model_id", "unknown")
                            ),
                        },
                    )
                if self.trace_data is not None:
                    primary_diag = getattr(self.client, "last_call_diagnostics", None)
                    if isinstance(primary_diag, dict):
                        raw_value = primary_diag.get("raw_response")
                        if isinstance(raw_value, str):
                            structured_primary_attempt_raw = raw_value
                self._log_warning(
                    f"Structured output failed for {context}; falling back to legacy prompt-schema mode."
                )
                self.last_call_diagnostics = {
                    "structured_attempted": True,
                    "structured_failed": True,
                    "fallback_used": True,
                    "fallback_error_class": type(e).__name__,
                }
                logger.warning(
                    "Structured output failed (provider=%s, model=%s): %s",
                    getattr(self.client, "provider", "unknown"),
                    getattr(self.client, "model", getattr(self.client, "model_id", "unknown")),
                    str(e),
                )
                legacy_prompt = direct.get_extraction_prompt(
                    markdown_content=markdown,
                    schema_json=schema_json,
                    is_partial=is_partial,
                    model_config=None,
                    structured_output=True,
                    schema_dict=schema_dict,
                    force_legacy_prompt_schema=True,
                )
                parsed_json = self.client.get_json_response(
                    prompt=legacy_prompt,
                    schema_json=schema_json,
                    structured_output=False,
                    response_top_level="object",
                    response_schema_name="direct_extraction",
                )
            if not parsed_json:
                self._log_warning(f"No valid JSON returned from LLM for {context}")
                return None
            if (
                self.structured_output
                and self.structured_sparse_check
                and not self.last_call_diagnostics.get("fallback_used")
                and self._is_sparse_structured_result(parsed_json, schema_dict, markdown)
            ):
                if self.trace_data is not None:
                    self.trace_data.emit(
                        "structured_output_fallback_triggered",
                        "extraction",
                        {
                            "context": context,
                            "reason": "SparseStructuredOutput",
                            "provider": getattr(self.client, "provider", "unknown"),
                            "model": getattr(
                                self.client, "model", getattr(self.client, "model_id", "unknown")
                            ),
                        },
                    )
                self._log_warning(
                    f"Structured output appears sparse for {context}; retrying legacy prompt-schema mode."
                )
                self.last_call_diagnostics = {
                    "structured_attempted": True,
                    "structured_failed": True,
                    "fallback_used": True,
                    "fallback_error_class": "SparseStructuredOutput",
                }
                legacy_prompt = direct.get_extraction_prompt(
                    markdown_content=markdown,
                    schema_json=schema_json,
                    is_partial=is_partial,
                    model_config=None,
                    structured_output=True,
                    schema_dict=schema_dict,
                    force_legacy_prompt_schema=True,
                )
                legacy_json = self.client.get_json_response(
                    prompt=legacy_prompt,
                    schema_json=schema_json,
                    structured_output=False,
                    response_top_level="object",
                    response_schema_name="direct_extraction",
                )
                if legacy_json:
                    parsed_json = legacy_json
            if self.trace_data is not None:
                if structured_primary_attempt_parsed_json is not None:
                    self.last_call_diagnostics["structured_primary_attempt_parsed_json"] = (
                        structured_primary_attempt_parsed_json
                    )
                if structured_primary_attempt_raw is not None:
                    self.last_call_diagnostics["structured_primary_attempt_raw"] = (
                        structured_primary_attempt_raw
                    )
            client_diag = getattr(self.client, "last_call_diagnostics", None)
            if isinstance(client_diag, dict) and client_diag:
                self.last_call_diagnostics["structured_attempted"] = bool(
                    self.last_call_diagnostics.get("structured_attempted")
                    or client_diag.get("structured_attempted")
                )
                self.last_call_diagnostics["structured_failed"] = bool(
                    self.last_call_diagnostics.get("structured_failed")
                    or client_diag.get("structured_failed")
                )
                self.last_call_diagnostics["fallback_used"] = bool(
                    self.last_call_diagnostics.get("fallback_used")
                    or client_diag.get("fallback_used")
                )
                if not self.last_call_diagnostics.get("fallback_error_class"):
                    self.last_call_diagnostics["fallback_error_class"] = client_diag.get(
                        "fallback_error_class"
                    )
                for passthrough_key in ("provider", "model"):
                    if passthrough_key in client_diag:
                        self.last_call_diagnostics[passthrough_key] = client_diag[passthrough_key]

            # Optional gleaning pass (direct only, full-doc)
            if (
                not is_partial
                and template is not None
                and isinstance(parsed_json, dict)
                and self._staged_config_raw.get("gleaning_enabled")
                and int(self._staged_config_raw.get("gleaning_max_passes", 1) or 1) >= 1
            ):

                def _gleaning_llm_call(prompt_dict: dict) -> dict | list | None:
                    return self.client.get_json_response(
                        prompt=prompt_dict,
                        schema_json=schema_json,
                        structured_output=self.structured_output,
                        response_top_level="object",
                        response_schema_name="direct_extraction",
                    )

                gleaned = run_gleaning_pass_direct(
                    markdown=markdown,
                    existing_result=parsed_json,
                    schema_json=schema_json,
                    llm_call_fn=_gleaning_llm_call,
                )
                if isinstance(gleaned, dict) and gleaned:
                    parsed_json = merge_gleaned_direct(
                        parsed_json,
                        gleaned,
                        description_merge_fields=frozenset({"description", "summary"}),
                        description_merge_max_length=4096,
                    )
            return parsed_json

        except Exception as e:
            self._log_error(f"Error during LLM call for {context}", e)
            return None

    def _get_client_max_tokens(self) -> int | None:
        """Get current max_tokens from client (for retry calculation)."""
        gen = getattr(self.client, "_generation", None)
        if gen is not None and getattr(gen, "max_tokens", None) is not None:
            return int(gen.max_tokens)
        max_tok = getattr(self.client, "max_tokens", None)
        if max_tok is not None:
            return int(max_tok)
        return None

    def _retry_max_tokens_for_truncation(self, context_max: int | None = None) -> int | None:
        """Compute max_tokens for one retry after truncation (current * multiplier, cap 32k)."""
        if not self._retry_on_truncation:
            return None
        current = context_max or self._get_client_max_tokens()
        if current is None:
            return None
        retry_max = min(
            int(current * self._truncation_retry_multiplier),
            self._truncation_retry_cap,
        )
        return retry_max if retry_max > current else None

    def _get_staged_call_max_tokens(self, context: str) -> int | None:
        """Select staged max_tokens override by call type. Delta batch calls use client default (no override)."""
        cfg = self._staged_config_raw or {}
        if "catalog_id_pass" in context:
            v = cfg.get("id_max_tokens")
            return int(v) if v is not None else None
        if "fill_call_" in context:
            v = cfg.get("fill_max_tokens")
            return int(v) if v is not None else None
        return None

    def _call_with_optional_max_tokens(
        self,
        prompt: dict[str, str],
        schema_json: str,
        max_tokens: int | None,
        response_top_level: Literal["object", "array"] = "object",
        response_schema_name: str = "staged_extraction",
        structured_output_override: bool | None = None,
    ) -> dict | list | None:
        """Invoke client with temporary max_tokens override when supported."""
        structured_output = (
            self.structured_output
            if structured_output_override is None
            else structured_output_override
        )
        generation = getattr(self.client, "_generation", None)
        if generation is None or max_tokens is None:
            return self.client.get_json_response(
                prompt=prompt,
                schema_json=schema_json,
                structured_output=structured_output,
                response_top_level=response_top_level,
                response_schema_name=response_schema_name,
            )
        original = getattr(generation, "max_tokens", None)
        try:
            generation.max_tokens = max_tokens
            return self.client.get_json_response(
                prompt=prompt,
                schema_json=schema_json,
                structured_output=structured_output,
                response_top_level=response_top_level,
                response_schema_name=response_schema_name,
            )
        finally:
            generation.max_tokens = original

    def _call_prompt(
        self,
        prompt: dict[str, str],
        schema_json: str,
        context: str,
        response_top_level: Literal["object", "array"] = "object",
        response_schema_name: str = "staged_extraction",
    ) -> dict | list | None:
        """Call LLM with explicit prompt/schema and staged truncation recovery."""
        try:
            max_tokens = self._get_staged_call_max_tokens(context)
            self.last_call_diagnostics = {
                "structured_attempted": bool(self.structured_output),
                "structured_failed": False,
                "fallback_used": False,
                "fallback_error_class": None,
            }
            try:
                parsed_json = self._call_with_optional_max_tokens(
                    prompt=prompt,
                    schema_json=schema_json,
                    max_tokens=max_tokens,
                    response_top_level=response_top_level,
                    response_schema_name=response_schema_name,
                )
            except ClientError as e:
                if not self.structured_output:
                    raise
                if self.trace_data is not None:
                    self.trace_data.emit(
                        "structured_output_fallback_triggered",
                        "extraction",
                        {
                            "context": context,
                            "reason": type(e).__name__,
                            "provider": getattr(self.client, "provider", "unknown"),
                            "model": getattr(
                                self.client, "model", getattr(self.client, "model_id", "unknown")
                            ),
                        },
                    )
                self._log_warning(
                    f"Structured output failed for {context}; retrying with legacy prompt-schema mode."
                )
                self.last_call_diagnostics = {
                    "structured_attempted": True,
                    "structured_failed": True,
                    "fallback_used": True,
                    "fallback_error_class": type(e).__name__,
                }
                legacy_prompt = {
                    "system": prompt["system"],
                    "user": (
                        prompt["user"]
                        + "\n\n=== TARGET SCHEMA ===\n"
                        + schema_json
                        + "\n=== END SCHEMA ===\n\n"
                    ),
                }
                parsed_json = self._call_with_optional_max_tokens(
                    prompt=legacy_prompt,
                    schema_json=schema_json,
                    max_tokens=max_tokens,
                    response_top_level=response_top_level,
                    response_schema_name=response_schema_name,
                    structured_output_override=False,
                )
            if not parsed_json:
                self._log_warning(f"No valid JSON returned from LLM for {context}")
                return None
            client_diag = getattr(self.client, "last_call_diagnostics", None)
            if isinstance(client_diag, dict) and client_diag:
                self.last_call_diagnostics["structured_attempted"] = bool(
                    self.last_call_diagnostics.get("structured_attempted")
                    or client_diag.get("structured_attempted")
                )
                self.last_call_diagnostics["structured_failed"] = bool(
                    self.last_call_diagnostics.get("structured_failed")
                    or client_diag.get("structured_failed")
                )
                self.last_call_diagnostics["fallback_used"] = bool(
                    self.last_call_diagnostics.get("fallback_used")
                    or client_diag.get("fallback_used")
                )
                if not self.last_call_diagnostics.get("fallback_error_class"):
                    self.last_call_diagnostics["fallback_error_class"] = client_diag.get(
                        "fallback_error_class"
                    )
                for passthrough_key in ("provider", "model"):
                    if passthrough_key in client_diag:
                        self.last_call_diagnostics[passthrough_key] = client_diag[passthrough_key]
            return parsed_json
        except Exception as e:
            details = getattr(e, "details", {}) if isinstance(e, ClientError) else {}
            truncated = bool(details.get("truncated")) if isinstance(details, dict) else False
            if truncated and self._retry_on_truncation:
                context_max = self._get_staged_call_max_tokens(context)
                if (
                    context_max is None
                    and isinstance(details, dict)
                    and isinstance(details.get("max_tokens"), int)
                ):
                    context_max = int(details["max_tokens"])
                retry_max = self._retry_max_tokens_for_truncation(context_max)
                if retry_max is not None:
                    try:
                        self._log_warning(
                            f"Retrying truncated call for {context} with max_tokens={retry_max}"
                        )
                        parsed_json = self._call_with_optional_max_tokens(
                            prompt=prompt,
                            schema_json=schema_json,
                            max_tokens=retry_max,
                            response_top_level=response_top_level,
                            response_schema_name=response_schema_name,
                        )
                        if parsed_json:
                            return parsed_json
                    except Exception as retry_e:
                        self._log_error(f"Retry after truncation failed for {context}", retry_e)
            self._log_error(f"Error during LLM call for {context}", e)
            return None

    def _extract_with_delta_contract(
        self,
        markdown: str,
        context: str,
        template: Type[BaseModel] | None = None,
        trace_data: Any = None,
    ) -> dict | list | None:
        """Delta extraction from a single markdown payload (fallback path)."""
        chunks = [markdown]
        chunk_metadata = [
            {
                "chunk_id": 0,
                "page_numbers": [0],
                "token_count": max(1, len(markdown.split())),
            }
        ]
        return self._run_delta_orchestrator(
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            context=context,
            template=template,
            trace_data=trace_data,
        )

    def _run_delta_orchestrator(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        context: str,
        template: Type[BaseModel] | None,
        trace_data: Any,
    ) -> dict | list | None:
        return run_delta_orchestrator(
            llm_call_fn=self._call_prompt,
            staged_config_raw=self._staged_config_raw,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            context=context,
            template=template,
            trace_data=trace_data,
            structured_output=self.structured_output,
        )

    def extract_from_chunk_batches(
        self,
        *,
        chunks: list[str],
        chunk_metadata: list[dict[str, Any]] | None,
        template: Type[BaseModel],
        context: str = "document",
    ) -> BaseModel | None:
        """Run delta extraction from pre-chunked content and validate final model."""
        self._log_extraction(
            f"Running delta extraction ([cyan]{len(chunks)}[/cyan] chunks)...",
            "DeltaExtraction",
        )
        self._log_extraction("Calling LLM (batch mode)...", "DeltaExtraction")
        schema_dict = self._get_schema_dict(template)
        parsed_json = self._run_delta_orchestrator(
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            context=context,
            template=template,
            trace_data=getattr(self, "trace_data", None),
        )
        if not parsed_json:
            return None
        if (
            self.structured_sparse_check
            and self.extraction_contract != "delta"
            and self._is_sparse_structured_result(
                parsed_json,
                schema_dict,
                "\n".join(chunks),
            )
        ):
            self._log_warning(
                f"Delta extraction appears sparse for {context}; falling back to direct extraction."
            )
            return None
        return self._validate_extraction(parsed_json, template, context)

    def _extract_with_staged_contract(
        self,
        markdown: str,
        schema_json: str,
        context: str,
        template: Type[BaseModel] | None = None,
        trace_data: Any = None,
    ) -> dict | list | None:
        """3-pass catalog extraction: ID discovery, fill nodes, assemble edges."""
        return run_staged_orchestrator(
            llm_call_fn=self._call_prompt,
            staged_config_raw=self._staged_config_raw,
            markdown=markdown,
            schema_json=schema_json,
            context=context,
            template=template,
            trace_data=trace_data,
            structured_output=self.structured_output,
        )

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
        # Log extraction start with contract-specific prefix
        mode = (
            "Delta"
            if (self.extraction_contract == "delta" and not is_partial)
            else (
                "Staged" if (self.extraction_contract == "staged" and not is_partial) else "Direct"
            )
        )
        prefix = f"{mode}Extraction"
        self._log_extraction(
            f"{mode} extraction from {context} ([cyan]{len(markdown)}[/cyan] chars)", prefix
        )

        # Early validation for empty markdown
        if not markdown or len(markdown.strip()) == 0:
            self._log_error(f"Markdown is empty for {context}. Cannot proceed.")
            return None

        # Get cached schema JSON
        schema_json = self._get_schema_json(template)
        schema_dict = self._get_schema_dict(template)

        self._log_extraction("Calling LLM...", prefix)

        # Call LLM
        parsed_json = self._call_llm_for_extraction(
            markdown=markdown,
            schema_json=schema_json,
            schema_dict=schema_dict,
            is_partial=is_partial,
            context=context,
            template=template,
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
                structured_output=False,
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
