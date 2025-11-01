"""
Utility functions for document extraction.
"""

import copy
from typing import Any, Dict, List

# Heuristic for token calculation (chars / 3.5 is a rough proxy)
# You can replace this with a proper tokenizer like tiktoken if you want more accuracy.
TOKEN_CHAR_RATIO = 3.5


def deep_merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges a 'source' dict into a 'target' dict.

    Merge behavior:
    - New keys from source are added to target
    - Lists are concatenated with smart deduplication
    - Nested dicts are recursively merged
    - Scalars are overwritten only if source has non-empty value
    - None, empty strings, empty lists, and empty dicts are ignored

    Args:
        target (Dict[str, Any]): The dictionary to merge into (modified in place)
        source (Dict[str, Any]): The dictionary to merge from

    Returns:
        Dict[str, Any]: The merged dictionary (same as target)
    """
    for key, source_value in source.items():
        # Skip empty values
        if source_value in (None, "", [], {}):
            continue

        if key not in target:
            # New key, add it
            target[key] = copy.deepcopy(source_value)
        else:
            target_value = target[key]

            # If both are dicts, merge recursively
            if isinstance(target_value, dict) and isinstance(source_value, dict):
                deep_merge_dicts(target_value, source_value)

            # If both are lists, concatenate and deduplicate
            elif isinstance(target_value, list) and isinstance(source_value, list):
                # Simple concatenation; you could add deduplication logic here
                # For objects, deduplication is trickier (compare by serialized JSON?)
                for item in source_value:
                    if item not in target_value:
                        target_value.append(item)

            # Otherwise, overwrite (prefer non-empty source)
            else:
                target[key] = copy.deepcopy(source_value)

    return target


def merge_pydantic_models(models: List[Any], template_class: type) -> Any:
    """
    Merge multiple Pydantic model instances into a single model.

    This function takes a list of Pydantic model instances and merges them
    into a single instance by deeply merging their dict representations.

    Args:
        models: List of Pydantic model instances to merge
        template_class: The Pydantic model class to use for the result

    Returns:
        A single merged Pydantic model instance, or None if models list is empty
    """
    # Return None for empty list (test expects None, not instance)
    if not models:
        return None

    if len(models) == 1:
        return models[0]

    # Convert all models to dicts
    dicts = [model.model_dump() for model in models]

    # Start with first model as base
    merged = copy.deepcopy(dicts[0])

    # Merge remaining models
    for d in dicts[1:]:
        deep_merge_dicts(merged, d)

    # Convert back to Pydantic model
    try:
        return template_class(**merged)
    except Exception as e:
        # If merge fails, return first model
        print(f"Warning: Failed to merge models: {e}")
        return models[0]


def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    """
    Split text into chunks that don't exceed max_tokens.

    This is a simple character-based approximation.
    For more accuracy, use a proper tokenizer (e.g., tiktoken).

    Args:
        text: The text to chunk
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of text chunks
    """
    max_chars = int(max_tokens * TOKEN_CHAR_RATIO)

    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Take a chunk
        end_pos = min(current_pos + max_chars, len(text))

        # Try to break at a sentence boundary if not at end
        if end_pos < len(text):
            # Look for last period, exclamation, or question mark
            for delimiter in [". ", "! ", "? ", "\n\n", "\n"]:
                last_break = text.rfind(delimiter, current_pos, end_pos)
                if last_break != -1:
                    end_pos = last_break + len(delimiter)
                    break

        chunk = text[current_pos:end_pos].strip()
        if chunk:
            chunks.append(chunk)

        current_pos = end_pos

    return chunks


def consolidate_extracted_data(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consolidate multiple extracted data dictionaries into one.

    Args:
        data_list: List of dictionaries to consolidate

    Returns:
        Single consolidated dictionary
    """
    if not data_list:
        return {}

    if len(data_list) == 1:
        return data_list[0]

    # Start with first dict
    consolidated = copy.deepcopy(data_list[0])

    # Merge remaining dicts
    for data in data_list[1:]:
        deep_merge_dicts(consolidated, data)

    return consolidated
