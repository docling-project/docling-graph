import os
import json
from typing import Dict, Any

# Heuristic for token calculation (chars / 3.5 is a rough proxy)
# You can replace this with a proper tokenizer like tiktoken
# if you want more accuracy.
TOKEN_CHAR_RATIO = 3.5

def _deep_merge_dicts(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges a 'source' dict into a 'target' dict.
    
    - Concatenates lists, avoiding simple duplicates.
    - Recursively merges nested dicts.
    - Overwrites scalars (str, int, bool) in 'target' with non-None/non-empty 
      values from 'source'.
    """
    for key, source_value in source.items():
        if key not in target:
            # New key, just add it
            if source_value is not None:
                target[key] = source_value
        else:
            # Key exists, merge
            target_value = target[key]
            
            if isinstance(target_value, list) and isinstance(source_value, list):
                # --- Merge Lists ---
                # Add items from source if they are not already in target
                # This provides basic deduplication for simple lists/scalars
                try:
                    existing_items = set(target[key])
                    for item in source_value:
                        if item not in existing_items:
                             target[key].append(item)
                             existing_items.add(item)
                except TypeError: 
                    # Handle unhashable types (like dicts)
                    # This is a simple merge; more complex logic may be needed
                    for item in source_value:
                        if item not in target[key]:
                            target[key].append(item)
            
            elif isinstance(target_value, dict) and isinstance(source_value, dict):
                # --- Recurse for Dictionaries ---
                _deep_merge_dicts(target_value, source_value)
            
            elif source_value is not None and source_value != "" and source_value != [] and source_value != {}:
                # --- Overwrite Scalar/Other ---
                # Only overwrite if the new value is non-empty. This prevents
                # a page with '{"policy_holder": null}' from erasing a
                # page with '{"policy_holder": "John Doe"}'
                target[key] = source_value
                
    return target

