"""Prompt templates for LLM document extraction.

This module provides optimized prompts for structured data extraction
from document markdown using LLMs.
"""

from typing import TypedDict


class PromptDict(TypedDict):
    """Type definition for prompt dictionaries."""
    system: str
    user: str


# Constants for reusable prompt components
_EXTRACTION_INSTRUCTIONS = (
    "1. Read the document text carefully.\n"
    "2. Extract ALL information that matches the provided schema.\n"
    "3. Return ONLY valid JSON that matches the schema.\n"
    '4. Use empty strings "" for missing text fields.\n'
    "5. Use [] for missing array fields.\n"
    "6. Use {} for missing nested objects.\n"
)

_SYSTEM_PROMPT_PARTIAL = (
    "You are an expert data extraction assistant. Your task is to extract "
    "structured information from document pages.\n\n"
    f"Instructions:\n{_EXTRACTION_INSTRUCTIONS}"
    "7. It's okay if the page only contains partial information.\n\n"
    "Important: Your response MUST be valid JSON that can be parsed."
)

_SYSTEM_PROMPT_COMPLETE = (
    "You are an expert data extraction assistant. Your task is to extract "
    "structured information from complete documents.\n\n"
    f"Instructions:\n{_EXTRACTION_INSTRUCTIONS}"
    "7. Be thorough: This is the complete document; try to extract all information.\n\n"
    "Important: Your response MUST be valid JSON that can be parsed."
)

_USER_PROMPT_TEMPLATE = (
    "Extract information from this {document_type}:\n\n"
    "=== {delimiter} ===\n"
    "{markdown_content}\n"
    "=== END {delimiter} ===\n\n"
    "=== TARGET SCHEMA ===\n"
    "{schema_json}\n"
    "=== END SCHEMA ===\n\n"
    "Extract ALL relevant data from the {document_type_lower} and return it as JSON "
    "following the schema above."
)

_CONSOLIDATION_PROMPT = """You are a data consolidation expert. Your task is to merge multiple \
partial JSON objects from a document into one single, accurate, and complete JSON object that \
strictly adheres to the provided schema.

You will be given three pieces of information:
1.  **SCHEMA**: A JSON schema that the final object MUST validate against.
2.  **RAW_JSONS**: A list of partial JSON objects extracted from different document chunks.
3.  **DRAFT_JSON**: A JSON object created by a programmatic (non-LLM) merge. This is a \
"first draft" for you to review and correct.

Your job is to act as a final reviewer. Use the DRAFT_JSON as a starting point, but \
critically evaluate it against the RAW_JSONS to fix any errors and ensure all data is captured.

**Your Instructions:**
1.  **Merge & Deduplicate**: Intelligently merge all entities from the RAW_JSONS. If the same \
entity (e.g., a "Material" with the same name) appears in multiple raw objects, it must be \
represented only ONCE in the final output.
2.  **Remove Phantoms**: If the DRAFT_JSON contains "phantom" or "empty" objects (e.g., a \
"Component" with no name or material), you MUST remove them unless they are fully specified in \
one of the RAW_JSONS.
3.  **Ensure Completeness**: Make sure all non-duplicate, valid data from all RAW_JSONS is \
present in the final object.
4.  **Validate Schema**: The final JSON object you output MUST strictly follow the provided SCHEMA.

Output *only* the final, consolidated JSON object. Do not add any other text, preambles, or explanations.

**SCHEMA:**
{schema_json}

**RAW_JSONS:**
{raw_jsons}

**DRAFT_JSON:**
{programmatic_json}

**FINAL CONSOLIDATED JSON:**
"""


# Methods for formatting and serving the prompts
def get_extraction_prompt(
    markdown_content: str,
    schema_json: str,
    is_partial: bool = False,
) -> PromptDict:
    """Generate system and user prompts for LLM extraction.

    Args:
        markdown_content: The document content in markdown format.
        schema_json: JSON schema of the Pydantic model.
        is_partial: Whether to expect partial data (for single page extraction).

    Returns:
        Dictionary with 'system' and 'user' keys containing the prompts.
    """
    system_prompt = _SYSTEM_PROMPT_PARTIAL if is_partial else _SYSTEM_PROMPT_COMPLETE

    document_type = "document page" if is_partial else "complete document"
    delimiter = "DOCUMENT PAGE" if is_partial else "COMPLETE DOCUMENT"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        document_type=document_type,
        document_type_lower=document_type.lower(),
        delimiter=delimiter,
        markdown_content=markdown_content,
        schema_json=schema_json,
    )

    return {"system": system_prompt, "user": user_prompt}


def get_consolidation_prompt(
    schema_json: str,
    raw_models: list,
    programmatic_model: object | None = None,
) -> str:
    """Generate the prompt for LLM-based consolidation.

    Args:
        schema_json: The Pydantic model schema.
        raw_models: List of Pydantic models from each extraction batch.
        programmatic_model: Result of the programmatic merge (optional).

    Returns:
        The formatted consolidation prompt.
    """
    raw_jsons = "\n\n---\n\n".join(
        m.model_dump_json(indent=2) for m in raw_models
    )

    programmatic_json = (
        programmatic_model.model_dump_json(indent=2)
        if programmatic_model
        else "No programmatic merge available."
    )

    return _CONSOLIDATION_PROMPT.format(
        schema_json=schema_json,
        raw_jsons=raw_jsons,
        programmatic_json=programmatic_json,
    )
