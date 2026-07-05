"""Tests for the deterministic verbatim anchor scan (provenance mode "detailed")."""

from docling_graph.core.provenance import (
    ChunkRecord,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
)
from docling_graph.core.provenance.anchor_scan import (
    locate_identifier,
    refine_ledger_spans,
)


def _ledger(
    ids: dict[str, str], chunk_ids: list[int], synthetic: bool = False
) -> tuple[ProvenanceLedger, NodeProvenance]:
    entry = NodeProvenance(
        identity_key="items[]|x",
        catalog_path="items[]",
        ids=ids,
        synthetic=synthetic,
        anchors=[SourceAnchor(chunk_id=c) for c in chunk_ids],
    )
    chunks = {c: ChunkRecord(chunk_id=c, batch_index=0) for c in chunk_ids}
    return ProvenanceLedger(chunks=chunks, nodes={entry.identity_key: entry}), entry


class TestRefineLedgerSpans:
    def test_unique_hit_yields_span_and_upgrades_resolution(self):
        ledger, entry = _ledger({"name": "Widget Pro"}, [0, 1])
        texts = {0: "intro text without the id", 1: "The widget pro spec sheet."}
        added = refine_ledger_spans(ledger, texts)
        assert added == 1
        verbatim = [a for a in entry.anchors if a.kind == "verbatim"]
        assert len(verbatim) == 1
        anchor = verbatim[0]
        assert anchor.chunk_id == 1
        start, end = anchor.span
        assert texts[1][start:end].lower() == "widget pro"
        assert ledger.resolution == "span"

    def test_multiple_chunks_each_get_a_verbatim_anchor(self):
        # An identifier genuinely appearing in two chunks yields two exact
        # anchors (both are true locations), one per chunk.
        ledger, entry = _ledger({"name": "Widget Pro"}, [0, 1])
        texts = {0: "widget pro here", 1: "widget pro there"}
        assert refine_ledger_spans(ledger, texts) == 2
        verbatim = sorted(a.chunk_id for a in entry.anchors if a.kind == "verbatim")
        assert verbatim == [0, 1]
        assert ledger.resolution == "span"

    def test_multiple_occurrences_in_one_chunk_take_first(self):
        ledger, entry = _ledger({"name": "Widget Pro"}, [0])
        texts = {0: "widget pro and again widget pro"}
        assert refine_ledger_spans(ledger, texts) == 1
        anchor = next(a for a in entry.anchors if a.kind == "verbatim")
        assert anchor.span == (0, len("Widget Pro"))

    def test_short_and_short_numeric_ids_skipped(self):
        ledger, _ = _ledger({"a": "X1", "b": "123"}, [0])
        texts = {0: "X1 appears and 123 appears"}
        assert refine_ledger_spans(ledger, texts) == 0

    def test_long_numeric_id_allowed(self):
        ledger, _ = _ledger({"ref": "20240117"}, [0])
        texts = {0: "invoice 20240117 issued"}
        assert refine_ledger_spans(ledger, texts) == 1

    def test_synthetic_entries_never_scanned(self):
        ledger, _ = _ledger({"name": "Widget Pro"}, [0], synthetic=True)
        texts = {0: "the widget pro sheet"}
        assert refine_ledger_spans(ledger, texts) == 0

    def test_non_distinctive_id_over_chunk_cap_is_skipped(self):
        # An identifier appearing in more chunks than the cap is treated as a
        # common term and skipped rather than re-smearing the node.
        ledger, entry = _ledger({"name": "Widget Pro"}, list(range(9)))
        texts = dict.fromkeys(range(9), "widget pro line")
        assert refine_ledger_spans(ledger, texts) == 0
        assert not any(a.kind == "verbatim" for a in entry.anchors)

    def test_only_ledger_chunks_are_scanned(self):
        ledger, _ = _ledger({"name": "Widget Pro"}, [0])
        # chunk 7 is not part of the ledger's chunk index -> never scanned
        texts = {0: "nothing here", 7: "widget pro lives here"}
        assert refine_ledger_spans(ledger, texts) == 0


class TestEscapeAwareLocate:
    """DocLang chunk text entity-escapes & < >; the scan retries the escaped form."""

    def test_finds_ampersand_value_in_escaped_text(self):
        # DocLang serializes "R&D Corp" as "R&amp;D Corp".
        hits = locate_identifier("R&D Corp", {0: "<text>R&amp;D Corp report</text>"})
        assert len(hits) == 1
        chunk_id, (start, end) = hits[0]
        assert chunk_id == 0
        assert "R&amp;D Corp" in "<text>R&amp;D Corp report</text>"[start:end]

    def test_finds_angle_bracket_value_in_escaped_text(self):
        hits = locate_identifier("A<B threshold", {0: "note A&lt;B threshold here"})
        assert len(hits) == 1

    def test_plain_value_still_found_without_escaping(self):
        hits = locate_identifier("Widget Pro", {0: "the Widget Pro sheet"})
        assert len(hits) == 1

    def test_absent_value_returns_empty(self):
        assert locate_identifier("R&D Corp", {0: "unrelated text"}) == []
