"""Tests for DocLang input-format helpers and serializers."""

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from docling_graph.core.utils.doclang_format import (
    DocLangSerializerProvider,
    is_doclang_format,
    prompt_framing,
    serialize_doclang,
    wants_location,
)


def _doc() -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    doc.add_page(page_no=1, size={"width": 612, "height": 792})
    prov = ProvenanceItem(
        page_no=1,
        bbox=BoundingBox(l=10, t=10, r=100, b=30, coord_origin=CoordOrigin.TOPLEFT),
        charspan=(0, 14),
    )
    doc.add_heading(text="Invoice #12345", prov=prov)
    return doc


class TestFormatPredicates:
    def test_is_doclang_format(self):
        assert is_doclang_format("doclang")
        assert is_doclang_format("doclang-geo")
        assert not is_doclang_format("markdown")

    def test_wants_location(self):
        assert wants_location("doclang-geo")
        assert not wants_location("doclang")
        assert not wants_location("markdown")

    def test_prompt_framing(self):
        assert prompt_framing("markdown") == ""
        assert "DocLang" in prompt_framing("doclang")
        assert "DocLang" in prompt_framing("doclang-geo")


class TestSerializeDoclang:
    def test_structure_without_location(self):
        out = serialize_doclang(_doc(), add_location=False)
        assert "<heading" in out
        assert "<location" not in out
        assert "Invoice #12345" in out

    def test_structure_with_location(self):
        out = serialize_doclang(_doc(), add_location=True)
        assert "<location" in out

    def test_sanitizes_control_chars(self):
        doc = DoclingDocument(name="s")
        doc.add_page(page_no=1, size={"width": 612, "height": 792})
        prov = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=1, t=1, r=2, b=2, coord_origin=CoordOrigin.TOPLEFT),
            charspan=(0, 5),
        )
        doc.add_text(label=DocItemLabel.TEXT, text="a\x00b", prov=prov)
        out = serialize_doclang(doc, add_location=False)
        assert "\x00" not in out
        assert "ab" in out


class TestDocLangSerializerProvider:
    def test_get_serializer_returns_doclang_serializer(self):
        provider = DocLangSerializerProvider(add_location=False)
        serializer = provider.get_serializer(_doc())
        out = serializer.serialize().text
        assert "<heading" in out
        assert "<location" not in out

    def test_geo_provider_includes_location(self):
        provider = DocLangSerializerProvider(add_location=True)
        out = provider.get_serializer(_doc()).serialize().text
        assert "<location" in out


class TestContentChars:
    """content_chars: markup-blind sizing for the auto-contract decision."""

    def test_markdown_is_passthrough(self):
        from docling_graph.core.utils.doclang_format import content_chars

        text = "# Title\n\nSome content."
        assert content_chars(text, "markdown") == len(text)
        assert content_chars(text, "auto") == len(text)

    def test_doclang_strips_tags_and_unwraps_cdata(self):
        from docling_graph.core.utils.doclang_format import content_chars

        doclang = (
            '<doctag><location value="29"/><location value="106"/>\n'
            "<![CDATA[Les murs d'enceinte]]>\n"
            "<ldiv/><page_break/></doctag>"
        )
        stripped = content_chars(doclang, "doclang")
        assert stripped < len(doclang)
        # The CDATA payload plus whitespace between elements survives.
        assert stripped >= len("Les murs d'enceinte")
        assert content_chars(doclang, "doclang-geo") == stripped

    def test_same_content_sizes_equal_across_formats(self):
        """The decision input must not grow when geometry markup is added."""
        from docling_graph.core.utils.doclang_format import content_chars

        payload = "x" * 500
        markdown = payload
        geo = f'<text><location value="1"/><location value="2"/><![CDATA[{payload}]]></text>'
        assert content_chars(geo, "doclang-geo") == content_chars(markdown, "markdown")
