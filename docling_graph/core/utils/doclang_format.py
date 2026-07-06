"""DocLang input-format handling for LLM-facing document text.

Everything the extraction path needs to treat the document text as DocLang
instead of markdown: format detection (:func:`is_doclang_format`,
:func:`wants_location`), system-prompt framing (:func:`prompt_framing`), and the
serializers themselves — a chunking ``serializer_provider`` (so HybridChunker
emits DocLang chunks) and a full/per-page serializer. Both serializers sanitize
control characters first (see :mod:`doclang_sanitizer`) so serialization can't
crash on OCR NUL bytes.

Modes:
- ``doclang``      — DocLang XML, structure only (no ``<location>`` geometry).
- ``doclang-geo``  — DocLang XML with page-coordinate geometry.
"""

from __future__ import annotations

import re

from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider
from docling_core.transforms.serializer.base import BaseDocSerializer
from docling_core.transforms.serializer.doclang import DocLangDocSerializer, DocLangParams
from docling_core.types.doc import DoclingDocument

from .doclang_sanitizer import sanitize_for_doclang

DOCLANG_FORMATS = ("doclang", "doclang-geo")

# One-line orientation prepended to system prompts when the document text is
# DocLang, so a markdown-trained model reads the XML tags as structure rather
# than content to extract.
DOCLANG_PROMPT_FRAMING = (
    "The document text is provided in DocLang, an XML markup format. Tags such as "
    "<heading>, <text>, <table>, <list>, and <picture> mark document structure; "
    "<location> values are page coordinates on a 512x512 grid; <page_break/> "
    "separates pages. Read the text inside the tags and treat the tags as "
    "structural hints — never extract tag names or coordinates as data. When "
    "copying a value from the document, copy only its plain text content, never "
    "any surrounding markup."
)


def is_doclang_format(llm_input_format: str) -> bool:
    """True when the format selects a DocLang serialization."""
    return llm_input_format in DOCLANG_FORMATS


def prompt_framing(llm_input_format: str) -> str:
    """Framing line for DocLang system prompts; empty string for markdown."""
    return DOCLANG_PROMPT_FRAMING if is_doclang_format(llm_input_format) else ""


def wants_location(llm_input_format: str) -> bool:
    """True when geometry ``<location>`` elements should be emitted."""
    return llm_input_format == "doclang-geo"


_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>")


def strip_doclang_markup(text: str) -> str:
    """Return the plain-text content of a DocLang serialization.

    CDATA payloads are unwrapped, then every remaining XML tag (structure,
    ``<location>`` geometry, page breaks) is removed. Used for markup-blind
    document sizing and for comparing extracted values against the text the
    model was actually served.
    """
    unwrapped = _CDATA_RE.sub(lambda m: m.group(1), text)
    return _TAG_RE.sub("", unwrapped)


def content_chars(text: str, llm_input_format: str = "markdown") -> int:
    """Character count of the *content* of an LLM-facing serialization.

    DocLang markup (tags, CDATA wrappers, ``<location>`` geometry) inflates the
    raw length substantially (+13% to several hundred percent, measured) without
    adding extractable information. Sizing decisions — direct-vs-dense contract
    resolution in particular — must not flip when the user changes the
    serialization of the same document, so they measure content, not markup.
    Markdown text is returned as-is (its markup overhead is negligible).
    """
    if not is_doclang_format(llm_input_format):
        return len(text)
    return len(strip_doclang_markup(text))


class DocLangSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider that makes HybridChunker emit DocLang chunk text.

    The document is sanitized on each ``get_serializer`` call (fast path returns
    the original when already clean), and the serializer stays bound to a
    document with identical ``self_ref`` values, so chunk item references still
    resolve.
    """

    def __init__(self, add_location: bool) -> None:
        self.add_location = add_location

    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        return DocLangDocSerializer(
            doc=sanitize_for_doclang(doc),
            params=DocLangParams(add_location=self.add_location),
        )


def serialize_doclang(
    document: DoclingDocument,
    *,
    add_location: bool,
    page_no: int | None = None,
) -> str:
    """Serialize a whole document (or a single page) to DocLang text."""
    params = DocLangParams(
        add_location=add_location,
        pages={page_no} if page_no is not None else None,
    )
    return DocLangDocSerializer(doc=sanitize_for_doclang(document), params=params).serialize().text
