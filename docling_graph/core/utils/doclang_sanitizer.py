"""Sanitize a DoclingDocument so ``export_to_doclang()`` cannot crash on it.

``DoclingDocument.export_to_doclang()`` (docling-core) raises
``ValueError: XML pretty-print failed: not well-formed (invalid token)`` when any
text item contains a C0 control character — NUL (0x00) shows up in real PDF text
extraction, and ``export_to_markdown()`` tolerates it while DocLang does not.

This strips XML-unencodable control characters (everything below 0x20 except tab,
newline and carriage return) from a *copy* of the document. The pipeline's own
document is never mutated. Captions are reference items into ``document.texts`` and
so are covered by scrubbing texts; table text lives in ``table.data.table_cells``.
Upstream bug tracked in the spec at .claude/specs/doclang/implementation-plan.md.
"""

from __future__ import annotations

import re

from docling_core.types.doc import DoclingDocument

# XML 1.0 forbids C0 control chars except \t (0x09), \n (0x0a), \r (0x0d).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _dirty(value: object) -> bool:
    """True if value is a string carrying an XML-forbidden control char."""
    return isinstance(value, str) and _CONTROL_CHARS.search(value) is not None


def _has_control_chars(document: DoclingDocument) -> bool:
    """Cheap scan across every serialized text field."""
    for item in getattr(document, "texts", None) or []:
        if _dirty(getattr(item, "text", None)) or _dirty(getattr(item, "orig", None)):
            return True
    for table in getattr(document, "tables", None) or []:
        data = getattr(table, "data", None)
        for cell in getattr(data, "table_cells", None) or []:
            if _dirty(getattr(cell, "text", None)):
                return True
    return False


def _scrub(value: object) -> object:
    """Strip control chars from a string; pass everything else through."""
    if isinstance(value, str):
        return _CONTROL_CHARS.sub("", value)
    return value


def sanitize_for_doclang(document: DoclingDocument) -> DoclingDocument:
    """Return a document safe for ``export_to_doclang()``.

    Fast path: if no forbidden control characters are present anywhere, the input
    is returned unchanged (no copy). Otherwise a deep copy is scrubbed and
    returned, leaving the caller's document untouched.
    """
    if not _has_control_chars(document):
        return document

    clean = document.model_copy(deep=True)
    for item in getattr(clean, "texts", None) or []:
        if getattr(item, "text", None) is not None:
            item.text = _scrub(item.text)
        if getattr(item, "orig", None) is not None:
            item.orig = _scrub(item.orig)
    for table in getattr(clean, "tables", None) or []:
        data = getattr(table, "data", None)
        for cell in getattr(data, "table_cells", None) or []:
            if getattr(cell, "text", None) is not None:
                cell.text = _scrub(cell.text)
    return clean
