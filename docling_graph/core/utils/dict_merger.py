"""
Utility functions for document extraction.
"""

import copy
from typing import Any, Dict, List


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
    # Return None for empty list
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


def deep_merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dicts with smart list deduplication.

    For lists of dicts (entities), uses content-based deduplication
    to avoid creating duplicate references.
    """
    for key, source_value in source.items():
        # Skip empty values
        if source_value in (None, "", [], {}):
            continue

        if key not in target:
            target[key] = copy.deepcopy(source_value)
        else:
            target_value = target[key]

            # Both dicts: recursive merge
            if isinstance(target_value, dict) and isinstance(source_value, dict):
                deep_merge_dicts(target_value, source_value)

            # Both lists: smart merge
            elif isinstance(target_value, list) and isinstance(source_value, list):
                # Check if this is a list of entities (dicts with identity)
                if target_value and isinstance(target_value[0], dict):
                    target[key] = _merge_entity_lists(target_value, source_value)
                else:
                    # Simple list: concatenate and deduplicate
                    for item in source_value:
                        if item not in target_value:
                            target_value.append(item)

            # Overwrite
            else:
                target[key] = copy.deepcopy(source_value)

    return target


def _merge_entity_lists(
    target_list: List[Dict],
    source_list: List[Dict],
) -> List[Dict]:
    """
    Merge two lists of entity dicts, avoiding duplicates.

    Uses content-based hashing to identify duplicates.
    """
    import hashlib
    import json

    def entity_hash(entity: Dict) -> str:
        """Compute content hash for entity."""
        # Use stable fields for identity
        stable_fields = {
            k: v for k, v in entity.items() if k not in {"id", "__class__"} and v is not None
        }
        content = json.dumps(stable_fields, sort_keys=True, default=str)
        return hashlib.blake2b(content.encode()).hexdigest()[:16]

    # Build hash -> entity map for target
    seen_hashes = {entity_hash(e): e for e in target_list}

    # Add source entities if not seen
    merged = list(seen_hashes.values())

    for source_entity in source_list:
        s_hash = entity_hash(source_entity)
        if s_hash not in seen_hashes:
            merged.append(source_entity)
            seen_hashes[s_hash] = source_entity

    return merged


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
