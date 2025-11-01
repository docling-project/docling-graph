"""
Prompt templates for LLM extraction.

This module provides improved prompts for structured data extraction
from document markdown using LLMs.
"""


def get_prompt(markdown_content: str, schema_json: str, is_partial: bool = False) -> dict:
    """
    Generates system and user prompts for LLM extraction.

    Args:
        markdown_content: The document content in markdown format.
        schema_json: JSON schema of the Pydantic model.
        is_partial: Whether to expect partial data (for single page extraction).

    Returns:
        Dict with 'system' and 'user' keys containing the prompts.
    """
    if is_partial:
        system_prompt = (
            "You are an expert data extraction assistant. Your task is to extract "
            "structured information from document pages.\n\n"
            "Instructions:\n"
            "1. Read the document page text carefully.\n"
            "2. Extract ALL information that matches the provided schema.\n"
            "3. Return ONLY valid JSON that matches the schema.\n"
            '4. Use empty strings "" for missing fields.\n'
            "5. Use [] for missing array fields.\n"
            "6. Use {} for missing nested objects.\n"
            "7. It's okay if the page only contains partial information.\n\n"
            "Important: Your response MUST be valid JSON that can be parsed."
        )

        user_prompt = (
            f"Extract information from this document page:\n\n"
            f"=== DOCUMENT PAGE ===\n"
            f"{markdown_content}\n"
            f"=== END PAGE ===\n\n"
            f"=== TARGET SCHEMA ===\n"
            f"{schema_json}\n"
            f"=== END SCHEMA ===\n\n"
            "Extract ALL relevant data from the page and return it as JSON "
            "following the schema above."
        )

    else:
        system_prompt = (
            "You are an expert data extraction assistant. Your task is to extract "
            "structured information from complete documents.\n\n"
            "Instructions:\n"
            "1. Read the ENTIRE document text carefully.\n"
            "2. Extract ALL information that matches the provided schema.\n"
            "3. Return ONLY valid JSON that strictly matches the schema.\n"
            '4. Use empty strings "" for missing text fields.\n'
            "5. Use [] for missing array fields.\n"
            "6. Use {} for missing nested objects.\n"
            "7. Be thorough: This is the complete document; try to extract all information.\n\n"
            "Important: Your response MUST be valid JSON that can be parsed."
        )

        user_prompt = (
            f"Extract information from this complete document:\n\n"
            f"=== COMPLETE DOCUMENT ===\n"
            f"{markdown_content}\n"
            f"=== END DOCUMENT ===\n\n"
            f"=== TARGET SCHEMA ===\n"
            f"{schema_json}\n"
            f"=== END SCHEMA ===\n\n"
            "Extract ALL relevant data from the document and return it as JSON "
            "following the schema above."
        )

    return {"system": system_prompt, "user": user_prompt}


def get_legacy_prompt(markdown_content: str, schema_json: str, is_partial: bool = False) -> str:
    """
    Legacy single-prompt version (for backwards compatibility).

    Returns a single string combining system and user messages.
    """
    prompts = get_prompt(markdown_content, schema_json, is_partial)
    return f"{prompts['system']}\n\n{prompts['user']}"
