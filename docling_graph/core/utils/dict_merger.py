"""
Utility functions for document extraction.
"""

import copy
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Minimum Jaccard similarity (child overlap) to merge entities when similarity fallback is enabled
_MERGE_SIMILARITY_THRESHOLD = 0.5


def merge_pydantic_models(
    models: List[Any], template_class: type, context_tag: str | None = None
) -> Any:
    """
    Merge multiple Pydantic model instances into a single model.

    Args:
        models: List of Pydantic model instances to merge
        template_class: The Pydantic model class to use for the result

    Returns:
        A single merged Pydantic model instance
    """
    # Return default instance for empty list
    if not models:
        return template_class()

    if len(models) == 1:
        return models[0]

    # Convert all models to dicts
    dicts = [model.model_dump() for model in models]

    # Start with first model as base
    merged = copy.deepcopy(dicts[0])

    # Merge remaining models
    for d in dicts[1:]:
        deep_merge_dicts(merged, d, context_tag=context_tag)

    # Convert back to Pydantic model
    try:
        return template_class(**merged)
    except Exception as e:
        # If merge fails, return first model
        print(f"Warning: Failed to merge models: {e}")
        return models[0]


def deep_merge_dicts(
    target: Dict[str, Any],
    source: Dict[str, Any],
    context_tag: str | None = None,
    identity_fields_map: dict[str, list[str]] | None = None,
    override_roots: set[str] | None = None,
    parent_path: str = "",
    merge_similarity_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Recursively merge dicts with smart list deduplication.

    For lists of dicts (entities), uses path-based identity_fields_map
    (e.g. "studies", "studies.experiments") for content-based deduplication.
    """
    for key, source_value in source.items():
        if override_roots and key in override_roots and source_value not in (None, "", [], {}):
            target[key] = copy.deepcopy(source_value)
            continue

        # Skip empty values
        if source_value in (None, "", [], {}):
            continue

        if key not in target:
            target[key] = copy.deepcopy(source_value)
        else:
            target_value = target[key]

            # Both dicts: recursive merge (path for nested lists)
            if isinstance(target_value, dict) and isinstance(source_value, dict):
                child_path = f"{parent_path}.{key}" if parent_path else key
                deep_merge_dicts(
                    target_value,
                    source_value,
                    context_tag=context_tag,
                    identity_fields_map=identity_fields_map,
                    override_roots=override_roots,
                    parent_path=child_path,
                    merge_similarity_fallback=merge_similarity_fallback,
                )

            # Both lists: smart merge with path-based identity
            elif isinstance(target_value, list) and isinstance(source_value, list):
                list_path = f"{parent_path}.{key}" if parent_path else key
                if target_value and isinstance(target_value[0], dict):
                    target[key] = _merge_entity_lists(
                        target_value,
                        source_value,
                        context_tag=context_tag,
                        identity_fields=(identity_fields_map or {}).get(list_path),
                        parent_path=list_path,
                        identity_fields_map=identity_fields_map,
                        merge_similarity_fallback=merge_similarity_fallback,
                    )
                else:
                    # Simple list: concatenate and deduplicate
                    for item in source_value:
                        if item not in target_value:
                            target_value.append(item)

            # Overwrite
            else:
                target[key] = copy.deepcopy(source_value)

    return target


def _child_fingerprints(entity: Dict) -> set[str]:
    """Set of hashes of child list items (and key scalars) for similarity comparison."""
    import hashlib
    import json

    fingerprints: set[str] = set()
    for k, v in entity.items():
        if k in ("id", "__class__"):
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                content = json.dumps(item, sort_keys=True, default=str)
                fingerprints.add(hashlib.blake2b(content.encode(), digest_size=8).hexdigest())
        elif v is not None and not isinstance(v, (dict, list)):
            fingerprints.add(f"{k}:{str(v)}")
    return fingerprints


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _merge_entity_lists(
    target_list: List[Dict],
    source_list: List[Dict],
    context_tag: str | None = None,
    identity_fields: list[str] | None = None,
    parent_path: str = "",
    identity_fields_map: dict[str, list[str]] | None = None,
    merge_similarity_fallback: bool = False,
) -> List[Dict]:
    """
    Merge two lists of entity dicts, avoiding duplicates.

    Uses path-based identity fields or content-based hashing. When merging
    two entities, passes parent_path so nested lists use path-aware identity.
    If merge_similarity_fallback is True and no id/hash match, may merge by
    child-overlap similarity (logs warning when used).
    """
    import hashlib
    import json

    def entity_hash(entity: Dict) -> str:
        """Compute content hash for entity. Same policy for target and source so duplicates match."""
        if identity_fields:
            identity_data = {field: entity.get(field) for field in identity_fields}
            if any(value not in (None, "") for value in identity_data.values()):
                content = json.dumps(identity_data, sort_keys=True, default=str)
                return hashlib.blake2b(content.encode()).hexdigest()[:16]

        # Use stable fields for identity (no context so source/target hashes match)
        stable_fields = {
            k: v for k, v in entity.items() if k not in {"id", "__class__"} and v is not None
        }
        content = json.dumps(stable_fields, sort_keys=True, default=str)
        return hashlib.blake2b(content.encode()).hexdigest()[:16]

    merged: List[Dict] = []
    id_map: Dict[str, Dict] = {}
    seen_hashes: Dict[str, Dict] = {}

    for entity in target_list:
        entity_id = entity.get("id")
        if entity_id:
            id_map[entity_id] = entity
            merged.append(entity)
        else:
            e_hash = entity_hash(entity)
            seen_hashes[e_hash] = entity
            merged.append(entity)

    for source_entity in source_list:
        source_id = source_entity.get("id")
        if source_id and source_id in id_map:
            deep_merge_dicts(
                id_map[source_id],
                source_entity,
                context_tag=context_tag,
                identity_fields_map=identity_fields_map,
                parent_path=parent_path,
                merge_similarity_fallback=merge_similarity_fallback,
            )
        elif source_id:
            merged.append(source_entity)
            id_map[source_id] = source_entity
        else:
            s_hash = entity_hash(source_entity)
            if s_hash in seen_hashes:
                deep_merge_dicts(
                    seen_hashes[s_hash],
                    source_entity,
                    context_tag=context_tag,
                    identity_fields_map=identity_fields_map,
                    parent_path=parent_path,
                    merge_similarity_fallback=merge_similarity_fallback,
                )
            elif merge_similarity_fallback:
                src_fp = _child_fingerprints(source_entity)
                best_score = 0.0
                best_entity: Dict | None = None
                for existing in merged:
                    if existing.get("id") and existing["id"] != source_entity.get("id"):
                        continue
                    existing_fp = _child_fingerprints(existing)
                    score = _jaccard(src_fp, existing_fp)
                    if score > best_score:
                        best_score = score
                        best_entity = existing
                if best_entity is not None and best_score >= _MERGE_SIMILARITY_THRESHOLD:
                    logger.warning(
                        "merge_similarity_fallback: merging entity by child overlap path=%s score=%.2f",
                        parent_path or "root",
                        best_score,
                    )
                    deep_merge_dicts(
                        best_entity,
                        source_entity,
                        context_tag=context_tag,
                        identity_fields_map=identity_fields_map,
                        parent_path=parent_path,
                        merge_similarity_fallback=merge_similarity_fallback,
                    )
                else:
                    merged.append(source_entity)
                    seen_hashes[s_hash] = source_entity
            else:
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
