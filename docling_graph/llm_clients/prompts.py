"""Prompt templates for LLM document extraction.

This module provides optimized prompts for structured data extraction
from document markdown using LLMs with model-aware adaptive prompting.
"""

from typing import TypedDict

from pydantic import BaseModel, Field

from .config import ModelCapability, ModelConfig


class PromptDict(TypedDict):
    """Type definition for prompt dictionaries."""

    system: str
    user: str


# Instruction variants for different model capabilities
_EXTRACTION_INSTRUCTIONS_SIMPLE = (
    "1. Read the document carefully.\n"
    "2. Extract information matching the schema.\n"
    "3. Return valid JSON only.\n"
    "4. Omit fields with no data.\n"
)

_EXTRACTION_INSTRUCTIONS_STANDARD = (
    "1. Read the document text carefully.\n"
    "2. Extract ALL information that matches the provided schema.\n"
    "3. Return ONLY valid JSON that matches the schema.\n"
    '4. Use empty strings "" for missing text fields.\n'
    "5. Use [] for missing array fields.\n"
    "6. Use {} for missing nested objects.\n"
)

_EXTRACTION_INSTRUCTIONS_ADVANCED = (
    "1. Carefully analyze the document text.\n"
    "2. Extract ALL information that matches the provided schema.\n"
    "3. Return ONLY valid JSON that strictly adheres to the schema.\n"
    '4. Use empty strings "" for missing text fields.\n'
    "5. Use [] for missing array fields.\n"
    "6. Use {} for missing nested objects.\n"
    "7. Preserve exact values and relationships from the source.\n"
    "8. Maintain data consistency across related fields.\n"
)

# Legacy constant for backward compatibility
_EXTRACTION_INSTRUCTIONS = _EXTRACTION_INSTRUCTIONS_STANDARD

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
    model_config: ModelConfig | None = None,
) -> dict[str, str]:
    """Generate system and user prompts for LLM extraction with model-aware adaptation.

    Args:
        markdown_content: The document content in markdown format.
        schema_json: JSON schema of the Pydantic model.
        is_partial: Whether to expect partial data (for single page extraction).
        model_config: Optional model configuration for adaptive prompting.

    Returns:
        Dictionary with 'system' and 'user' keys containing the prompts.
    """
    # Select instructions based on model capability
    if model_config:
        if model_config.capability == ModelCapability.SIMPLE:
            instructions = _EXTRACTION_INSTRUCTIONS_SIMPLE
        elif model_config.capability == ModelCapability.ADVANCED:
            instructions = _EXTRACTION_INSTRUCTIONS_ADVANCED
        else:
            instructions = _EXTRACTION_INSTRUCTIONS_STANDARD
    else:
        # Default to standard if no config provided (backward compatibility)
        instructions = _EXTRACTION_INSTRUCTIONS_STANDARD

    # Build system prompt
    if is_partial:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from document pages.\n\n"
            f"Instructions:\n{instructions}"
            "Note: This is a partial page; incomplete data is expected.\n\n"
            "Important: Your response MUST be valid JSON."
        )
    else:
        system_prompt = (
            "You are an expert data extraction assistant. "
            "Extract structured information from complete documents.\n\n"
            f"Instructions:\n{instructions}"
            "Be thorough: Extract all available information.\n\n"
            "Important: Your response MUST be valid JSON."
        )

    # Build user prompt (same for all models)
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
    programmatic_model: BaseModel | None = None,
    model_config: ModelConfig | None = None,
) -> str | list[str]:
    """Generate the prompt(s) for LLM-based consolidation with model-aware adaptation.

    Args:
        schema_json: The Pydantic model schema.
        raw_models: List of Pydantic models from each extraction batch.
        programmatic_model: Result of the programmatic merge (optional).
        model_config: Optional model configuration for adaptive prompting.

    Returns:
        - str: Single prompt for simple/standard models
        - list[str]: Multiple prompts for advanced models (Chain of Density)
    """
    raw_jsons = "\n\n---\n\n".join(m.model_dump_json(indent=2) for m in raw_models)

    # Simple models: basic merge only
    if model_config and model_config.capability == ModelCapability.SIMPLE:
        return f"""Merge these JSON objects into one, removing duplicates.

SCHEMA:
{schema_json}

OBJECTS TO MERGE:
{raw_jsons}

Output the merged JSON only."""

    # Advanced models: Chain of Density (multi-turn)
    elif model_config and model_config.supports_chain_of_density:
        # Stage 1: Initial merge
        stage1 = f"""Merge these JSON objects, removing duplicates.

SCHEMA: {schema_json}
OBJECTS: {raw_jsons}

Output merged JSON."""

        # Stage 2: Refinement (will be formatted with stage1 result)
        stage2_template = """Review and refine this merged JSON.

SCHEMA: {schema}
MERGED: {{stage1_result}}
ORIGINALS: {originals}

Fix any missing data or errors. Output final JSON."""

        # Return list of prompts for multi-turn execution
        return [stage1, stage2_template]

    # Standard models: single-pass with draft (default)
    else:
        programmatic_json = (
            programmatic_model.model_dump_json(indent=2)
            if programmatic_model
            else "No draft available."
        )

        return f"""Merge multiple JSON objects into one accurate result.

SCHEMA:
{schema_json}

RAW EXTRACTIONS:
{raw_jsons}

DRAFT MERGE:
{programmatic_json}

Review the draft against raw extractions. Fix errors, remove duplicates, ensure completeness.
Output final JSON only."""
