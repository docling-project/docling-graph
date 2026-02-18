"""
Entity name normalization for consistent deduplication across extraction contracts.

Normalizes display names to a canonical form (e.g. UPPER_SNAKE) so that
"John Doe", "john doe", and "The John Doe" resolve to the same key.
Used by dict_merger, delta resolvers, staged merge, and node_id_registry.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_entity_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKD", raw)
    trimmed = text.strip()
    if not trimmed:
        return ""
    for prefix in ("The ", "the ", "A ", "a ", "An ", "an "):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix) :].strip()
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


# Identity fields that use name-style normalization (same as staged _NAME_IDENTITY_FIELDS).
_NAME_DEDUP_FIELDS: frozenset[str] = frozenset({"name", "title", "nom"})


def canonicalize_identity_for_dedup(field_name: str, value: Any) -> str:
    """
    Canonicalize an identity field value for dedup key computation only.
    - For name/title/nom: use normalize_entity_name so "John Doe" and "john doe" match.
    - For other id fields (run_id, batch_id, etc.): lowercase and keep only
      alphanumeric characters so "Run-1", "run_1", "run1" all become "run1".
    """
    if value is None:
        return ""
    if field_name in _NAME_DEDUP_FIELDS and isinstance(value, str):
        return normalize_entity_name(value)
    text = str(value).strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    lower = normalized.casefold()
    return re.sub(r"[^a-z0-9]", "", lower)
