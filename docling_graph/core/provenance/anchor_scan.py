"""Deterministic verbatim locator — the precise node-to-source mechanism.

Given an identifier string and the chunk texts, finds the exact chunk(s) and
character span(s) where it literally appears — pure string search, never an LLM.

This is what turns a vague batch-level ``observed`` set into an exact
chunk/page. The binder drives it with each graph node's *final* identifier
values (post-fill/post-validation), which is essential: the dense skeleton may
capture a generic placeholder (e.g. "SlurryComponent") while the real value the
node ends up with ("LiFePO4") is what actually appears in the document.

Guards (fail-empty, never fail-wrong):
- values shorter than 3 characters, or short purely-numeric values, are skipped;
- an identifier matching more than ``_MAX_VERBATIM_CHUNKS`` chunks is treated as
  a non-distinctive common term and skipped, so it never re-smears the node;
- one location per matching chunk (first occurrence in that chunk); a casefold
  length drift (rare unicode) skips the match rather than emitting a misaligned
  span.
"""

from __future__ import annotations

import logging
from typing import Mapping

from .models import ProvenanceLedger, SourceAnchor

logger = logging.getLogger(__name__)

_MIN_TEXT_LEN = 3
_MIN_DIGIT_ONLY_LEN = 4
# An identifier literally appearing in more chunks than this is not a
# distinctive locator (e.g. a common term); skip it rather than re-smear.
_MAX_VERBATIM_CHUNKS = 6


def _first_occurrence(needle: str, haystack: str) -> tuple[int, int] | None:
    """Case-insensitive first occurrence of needle in haystack, or None."""
    folded_hay = haystack.casefold()
    folded_needle = needle.casefold()
    if len(folded_hay) != len(haystack):
        # Casefold changed offsets (rare unicode expansion); use exact match only.
        folded_hay = haystack
        folded_needle = needle
    start = folded_hay.find(folded_needle)
    if start == -1:
        return None
    end = start + len(needle)
    if haystack[start:end].casefold() != needle.casefold():
        return None  # offset drift safety net
    return (start, end)


def locate_identifier(
    value: str, chunk_texts: Mapping[int, str]
) -> list[tuple[int, tuple[int, int]]]:
    """Return [(chunk_id, span)] where a distinctive identifier appears verbatim.

    Empty when the value is non-distinctive (too short/numeric), absent, or so
    common it exceeds the chunk cap.
    """
    value = (value or "").strip()
    if len(value) < _MIN_TEXT_LEN:
        return []
    if value.isdigit() and len(value) < _MIN_DIGIT_ONLY_LEN:
        return []
    matches: list[tuple[int, tuple[int, int]]] = []
    for chunk_id, chunk_text in chunk_texts.items():
        if not chunk_text:
            continue
        if value.casefold() in chunk_text.casefold():
            span = _first_occurrence(value, chunk_text)
            if span is None:
                continue
            matches.append((chunk_id, span))
            if len(matches) > _MAX_VERBATIM_CHUNKS:
                return []  # non-distinctive; do not re-smear
    return matches


def locate_values(
    values: list[str], chunk_texts: Mapping[int, str]
) -> list[tuple[int, tuple[int, int]]]:
    """Union of verbatim locations across a node's identifier values."""
    seen: set[tuple[int, tuple[int, int]]] = set()
    out: list[tuple[int, tuple[int, int]]] = []
    for value in values:
        for hit in locate_identifier(value, chunk_texts):
            if hit not in seen:
                seen.add(hit)
                out.append(hit)
    return out


def refine_ledger_spans(ledger: ProvenanceLedger, chunk_texts: Mapping[int, str]) -> int:
    """Add verbatim anchors to every non-synthetic ledger entry from its ids.

    Kept for completeness / direct ledger post-processing; the binder is the
    primary caller of :func:`locate_values` because it has the final model ids.
    Returns the number of verbatim anchors added.
    """
    added = 0
    for entry in ledger.nodes.values():
        if entry.synthetic or "scope:document" in entry.notes:
            continue
        existing = {(a.chunk_id, a.span) for a in entry.anchors if a.kind == "verbatim"}
        for chunk_id, span in locate_values(list(entry.ids.values()), chunk_texts):
            if chunk_id not in ledger.chunks or (chunk_id, span) in existing:
                continue
            entry.anchors.append(SourceAnchor(chunk_id=chunk_id, kind="verbatim", span=span))
            existing.add((chunk_id, span))
            added += 1
    if added:
        ledger.resolution = "span"
        logger.info("Provenance anchor scan: %s verbatim anchor(s) resolved", added)
    return added
