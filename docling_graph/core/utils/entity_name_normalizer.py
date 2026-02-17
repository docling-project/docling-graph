"""
Entity name normalization for consistent deduplication across extraction contracts.

Normalizes display names to a canonical form (e.g. UPPER_SNAKE) so that
"John Doe", "john doe", and "The John Doe" resolve to the same key.
Used by dict_merger, delta resolvers, staged merge, and node_id_registry.
"""
from __future__ import annotations

import unicodedata


def normalize_entity_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKD", raw)
    trimmed = text.strip()
    if not trimmed:
        return ""
    for prefix in ("The ", "the ", "A ", "a ", "An ", "an "):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):].strip()
            break
    if trimmed in ("The", "the", "A", "a", "An", "an"):
        trimmed = ""
    if not trimmed:
        return ""
    words = []
    for word in trimmed.split():
        if not word:
            continue
        if word.endswith("'s"):
            word = word[:-2]
        elif len(word) >= 2 and word[-2:] == "\u2019s":
            word = word[:-2]
        if word:
            words.append(word)
    if not words:
        return ""
    return "_".join(words).upper()
