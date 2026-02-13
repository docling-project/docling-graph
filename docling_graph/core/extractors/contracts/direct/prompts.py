"""
Prompt templates for LLM document extraction.

This module provides optimized prompts for structured data extraction
from document markdown using LLMs.

Design goals:
- Domain-agnostic (works across domains).
- Chunk-friendly (partial extraction).
- Relationship-friendly (avoid default empty arrays that kill edges).
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel

from .....llm_clients.schema_utils import build_compact_semantic_guide


class PromptDict(TypedDict):
    """Type definition for prompt dictionaries."""

    system: str
    user: str


# ---------------------------------------------------------------------------
# Standard extraction instructions (domain-agnostic)
# ---------------------------------------------------------------------------

# IMPORTANT: do NOT default to "" / [] / {} for missing values.
# This is critical for chunk extraction + graph edges:
# - If a relationship is implied, produce a minimal reference instead of [].
_EXTRACTION_INSTRUCTIONS = (
    "1. Read the provided document text carefully.\n"
    "2. Extract ALL information that matches the provided schema AND is supported by the text.\n"
    "3. Return ONLY valid JSON that matches the schema (no extra keys).\n"
    "4. If a field is not evidenced in the provided text chunk, OMIT the field (preferred).\n"
    '5. Do NOT use empty strings "" for missing text; omit the field or use null for optional scalars.\n'
    "6. For arrays/objects: ONLY output [] or {} when the text explicitly indicates "
    '"none / not applicable / no items". Otherwise omit the field.\n'
    "7. For relationship-like arrays (lists of nested objects): if the relationship is stated "
    "but details are incomplete, output a minimal reference object using identifier fields "
    '(e.g., {"name": "..."} or {"id": "..."}) rather than outputting an empty list.\n'
    "8. Keep identifiers consistent across references (same entity => same identifier value).\n"
    "9. If the schema expects an object, do not output a scalar. For quantity/measurement "
    "objects, output an object with numeric_value and unit (if known), or text_value.\n"
)


_USER_PROMPT_TEMPLATE = (
    "Extract information from this {document_type}:\n\n"
    "=== {delimiter} ===\n"
    "{markdown_content}\n"
    "=== END {delimiter} ===\n\n"
    "=== TARGET SCHEMA ===\n"
    "{schema_json}\n"
    "=== END SCHEMA ===\n\n"
    "Return ONLY a JSON object that follows the target schema."
)

_USER_PROMPT_TEMPLATE_COMPACT = (
    "Extract information from this {document_type}:\n\n"
    "=== {delimiter} ===\n"
    "{markdown_content}\n"
    "=== END {delimiter} ===\n\n"
    "=== SEMANTIC FIELD GUIDANCE ===\n"
    "{semantic_guide}\n"
    "=== END GUIDANCE ===\n\n"
    "Return ONLY a JSON object that follows the API-enforced schema."
)


# ---------------------------------------------------------------------------
# Public API: Prompt generation
# ---------------------------------------------------------------------------


def get_extraction_prompt(
    markdown_content: str,
    schema_json: str,
    is_partial: bool = False,
    model_config: Any | None = None,  # Kept for backward compatibility, not used
    structured_output: bool = True,
    schema_dict: dict[str, Any] | None = None,
    force_legacy_prompt_schema: bool = False,
) -> dict[str, str]:
    """
    Generate system and user prompts for LLM extraction.

    Simplified version without capability branching - always uses standard instructions.
    The model_config parameter is kept for backward compatibility but not used.

    Args:
        markdown_content: Document text in markdown format
        schema_json: JSON schema for extraction target
        is_partial: Whether this is a partial document (page) or complete
        model_config: Deprecated, kept for backward compatibility

    Returns:
        Dictionary with 'system' and 'user' prompt strings
    """
    instructions = _EXTRACTION_INSTRUCTIONS

    if is_partial:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from document pages.\n\n"
            f"Instructions:\n{instructions}\n"
            "Note: This is a partial page; incomplete data is expected.\n\n"
            "Important: Your response MUST be valid JSON."
        )
    else:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from complete documents.\n\n"
            f"Instructions:\n{instructions}\n"
            "Be thorough: Extract all available information.\n\n"
            "Important: Your response MUST be valid JSON."
        )

    document_type = "document page" if is_partial else "complete document"
    delimiter = "DOCUMENT PAGE" if is_partial else "COMPLETE DOCUMENT"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        document_type=document_type,
        delimiter=delimiter,
        markdown_content=markdown_content,
        schema_json=schema_json,
    )
    if structured_output and not force_legacy_prompt_schema:
        semantic_guide = build_compact_semantic_guide(schema_dict or {})
        user_prompt = _USER_PROMPT_TEMPLATE_COMPACT.format(
            document_type=document_type,
            delimiter=delimiter,
            markdown_content=markdown_content,
            semantic_guide=semantic_guide,
        )

    return {"system": system_prompt, "user": user_prompt}
