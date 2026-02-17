"""
Merge text descriptions with sentence-level dedup and optional truncation.

Used when the same entity is seen in multiple chunks/documents: add only
new sentences from the incoming description, then truncate at sentence boundary.
Optional LLM summarizer: when total length exceeds a threshold, a callable can
produce one concise summary; fallback to sentence-dedup + truncate on failure.
Used by dict_merger, delta merge_delta_graphs, and staged merge.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)


def truncate_at_sentence_boundary(text: str, max_length: int) -> str:
    """
    Truncate text at the last sentence boundary (. ! ?) before max_length.

    If no boundary found before max_length, truncates at max_length.
    If text is shorter than max_length, returns text unchanged.

    Args:
        text: Input text.
        max_length: Maximum length in characters.

    Returns:
        Truncated text ending at a sentence boundary when possible.
    """
    if not text or max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    end = max_length
    for i, c in enumerate(text[:max_length]):
        if c in ".!?":
            end = i + 1
    return text[:end]


def merge_descriptions(
    existing: str,
    new: str,
    max_length: int = 4096,
    summarizer: Callable[[str, list[str]], str] | None = None,
    summarizer_min_total_length: int = 0,
) -> str:
    """
    Merge two descriptions: add only sentences from new that are not in existing.

    Splits on sentence boundaries (. ! ?), deduplicates, then truncates
    at sentence boundary. Empty or duplicate new content leaves existing unchanged.

    When summarizer is set and combined length >= summarizer_min_total_length,
    the summarizer is called to produce one concise summary; on failure or
    if below threshold, falls back to sentence-dedup + truncate.

    Args:
        existing: Current description.
        new: Incoming description to merge.
        max_length: Maximum length of merged result (default 4096).
        summarizer: Optional callable (existing, list of new strings) -> summary.
        summarizer_min_total_length: Use summarizer when len(existing)+len(new) >= this (0 = off).

    Returns:
        Merged description with no duplicate sentences, truncated at boundary.
    """
    if not existing or not existing.strip():
        return truncate_at_sentence_boundary((new or "").strip(), max_length)
    existing = existing.strip()
    if not new or not new.strip():
        return truncate_at_sentence_boundary(existing, max_length)
    new = new.strip()

    if (
        summarizer is not None
        and summarizer_min_total_length > 0
        and (len(existing) + len(new)) >= summarizer_min_total_length
    ):
        try:
            summary = summarizer(existing, [new])
            if isinstance(summary, str) and summary.strip():
                return truncate_at_sentence_boundary(summary.strip(), max_length)
        except Exception as e:
            logger.warning("Description summarizer failed, using sentence-dedup fallback: %s", e)

    if existing.find(new) != -1:
        return truncate_at_sentence_boundary(existing, max_length)
    # Split on sentence boundaries (. ! ? followed by space or end)
    parts = re.split(r"(?<=[.!?])\s+", new)
    additions = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if s not in existing:
            additions.append(s)
    if not additions:
        return truncate_at_sentence_boundary(existing, max_length)
    combined = existing + " " + " ".join(additions)
    return truncate_at_sentence_boundary(combined, max_length)
