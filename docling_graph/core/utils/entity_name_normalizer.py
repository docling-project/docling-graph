"""
Entity name normalization for consistent deduplication across extraction contracts.

Normalizes display names to a canonical form (e.g. UPPER_SNAKE) so that
"John Doe", "john doe", and "The John Doe" resolve to the same key.
Used by dict_merger, dense resolvers, and node_id_registry.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Folding these blocks is what makes "\u00e9" match "e". Script-specific marks
# (kana voicing, Indic vowel signs, Thai tone marks, Arabic harakat) belong to the letter and
# are kept, otherwise distinct words such as "\u30ac\u30b9" and "\u30ab\u30b9" collide.
_FOLDABLE_MARK_RANGES = (
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1AB0, 0x1AFF),  # Combining Diacritical Marks Extended
    (0x1DC0, 0x1DFF),  # Combining Diacritical Marks Supplement
    (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols
    (0xFE20, 0xFE2F),  # Combining Half Marks
)


def _fold_diacritics(text: str) -> str:
    """Drop generic diacritics, keep script-specific marks, then recompose."""
    decomposed = unicodedata.normalize("NFKD", text)
    kept = "".join(
        ch for ch in decomposed if not any(lo <= ord(ch) <= hi for lo, hi in _FOLDABLE_MARK_RANGES)
    )
    return unicodedata.normalize("NFC", kept)


def normalize_entity_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    # Fold accents: without it "PROPRI\u00c9TAIRE" differs from "PROPRIETAIRE"
    # and the same offer extracted with/without accents becomes two nodes.
    text = _fold_diacritics(raw)
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
    # Underscores and hyphens are word separators here: models emit the same
    # name as "PROPRIETAIRE_NON_OCCUPANT", "Propri\u00e9taire non occupant" or
    # "PROPRIETAIRE-NON-OCCUPANT" depending on the batch, and a dedup key must
    # not depend on that choice.
    for word in re.split(r"[\s_\-]+", trimmed):
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


# Identity fields that use name-style normalization.
_NAME_DEDUP_FIELDS: frozenset[str] = frozenset({"name", "title", "nom"})


def canonicalize_identity_for_dedup(field_name: str, value: Any) -> str:
    """
    Canonicalize an identity field value for dedup key computation only.
    - For name/title/nom: use normalize_entity_name so "John Doe" and "john doe" match.
    - For other id fields (run_id, batch_id, etc.): casefold, fold generic
      diacritics and keep only letters, digits and marks (in any script) so "Run-1", "run_1",
      "run1" all become "run1" while "北京" and "上海" stay distinct.
    """
    if value is None:
        return ""
    if field_name in _NAME_DEDUP_FIELDS and isinstance(value, str):
        return normalize_entity_name(value)
    text = str(value).strip()
    if not text:
        return ""
    folded = _fold_diacritics(text).casefold()
    # Unicode-aware: an ASCII-only [a-z0-9] filter erased every non-Latin
    # value to "", so distinct Chinese/Cyrillic/Greek ids shared one key.
    # Keep letters, numbers and the marks that survive folding; drop
    # punctuation, symbols and separators.
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] in "LNM")
