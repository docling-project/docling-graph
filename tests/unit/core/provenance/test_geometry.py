"""Tests for geometry provenance: ItemGeometry, ledger mapping, entry helper."""

from docling_graph.core.provenance import (
    ChunkRecord,
    ItemGeometry,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
    chunk_index_ledger,
    geometry_from_meta,
)
from docling_graph.core.provenance.models import dclg_location_from_bbox


class TestGeometryFromMeta:
    def test_builds_item_geometry_from_metadata(self):
        meta = {
            "item_geometry": [
                {
                    "ref": "#/texts/7",
                    "page_no": 3,
                    "bbox": [10, 20, 100, 40],
                    "page_width": 200,
                    "page_height": 300,
                    "dclg_location": [26, 34, 256, 68],
                }
            ]
        }
        geoms = geometry_from_meta(meta)
        assert len(geoms) == 1
        assert geoms[0] == ItemGeometry(
            ref="#/texts/7",
            page_no=3,
            bbox=(10, 20, 100, 40),
            page_width=200,
            page_height=300,
            dclg_location=(26, 34, 256, 68),
        )

    def test_empty_when_no_geometry_key(self):
        assert geometry_from_meta({}) == ()

    def test_skips_malformed_entries(self):
        meta = {
            "item_geometry": [
                {"ref": "#/texts/1", "page_no": 1, "bbox": [1, 2, 3]},  # short bbox
                {"ref": "#/texts/2"},  # missing page/bbox
                {"ref": "#/texts/3", "page_no": 2, "bbox": [0, 0, 5, 5]},
            ]
        }
        geoms = geometry_from_meta(meta)
        assert len(geoms) == 1
        assert geoms[0].ref == "#/texts/3"

    def test_bbox_coerced_to_int(self):
        # Older/float-shaped input is coerced to whole pixels.
        meta = {"item_geometry": [{"ref": "#/x", "page_no": 1, "bbox": [10.0, 20.0, 30.0, 40.0]}]}
        assert geometry_from_meta(meta)[0].bbox == (10, 20, 30, 40)

    def test_carries_page_dimensions_and_dclg_location(self):
        meta = {
            "item_geometry": [
                {
                    "ref": "#/texts/0",
                    "page_no": 1,
                    "bbox": [118, 106, 238, 189],
                    "page_width": 1021,
                    "page_height": 1423,
                    "dclg_location": [59, 38, 119, 68],
                }
            ]
        }
        g = geometry_from_meta(meta)[0]
        assert g.page_width == 1021
        assert g.page_height == 1423
        assert g.dclg_location == (59, 38, 119, 68)

    def test_page_dimensions_optional(self):
        meta = {"item_geometry": [{"ref": "#/x", "page_no": 1, "bbox": [0, 0, 1, 1]}]}
        g = geometry_from_meta(meta)[0]
        assert g.page_width is None
        assert g.page_height is None
        assert g.dclg_location is None


class TestDclgLocationFromBbox:
    """dclg_location_from_bbox reproduces document.dclg's <location> values literally."""

    def test_matches_document_dclg_values(self):
        # Real invoice numbers: texts/0 on a 1021x1423 page -> (59, 38, 119, 68).
        # Computed from the ORIGINAL float coordinates, not the rounded bbox.
        assert dclg_location_from_bbox(117.6986, 106.1512, 238.2333, 188.6336, 1021.0, 1423.0) == (
            59,
            38,
            119,
            68,
        )

    def test_clamped_to_grid(self):
        # A box touching the page edge maps to 511 (resolution - 1), never 512.
        assert dclg_location_from_bbox(0.0, 0.0, 1000.0, 500.0, 1000.0, 500.0) == (0, 0, 511, 511)

    def test_corner_normalization(self):
        # Inverted corners are normalized to (min, min, max, max).
        assert dclg_location_from_bbox(100.0, 50.0, 10.0, 5.0, 512.0, 512.0) == (10, 5, 100, 50)

    def test_serialized_into_json(self):
        ledger = chunk_index_ledger(
            ["text"],
            [
                {
                    "chunk_id": 0,
                    "item_geometry": [
                        {
                            "ref": "#/texts/0",
                            "page_no": 1,
                            "bbox": [118, 106, 238, 189],
                            "page_width": 1021,
                            "page_height": 1423,
                            "dclg_location": [59, 38, 119, 68],
                        }
                    ],
                }
            ],
        )
        import json

        dumped = json.loads(ledger.model_dump_json())
        entry = dumped["chunks"]["0"]["item_geometry"][0]
        assert entry["dclg_location"] == [59, 38, 119, 68]
        assert "coord_origin" not in entry


class TestChunkIndexLedgerGeometry:
    def test_ledger_carries_geometry_and_is_version_2(self):
        chunks = ["chunk zero text"]
        metadata = [
            {
                "chunk_id": 0,
                "page_numbers": [1],
                "doc_item_refs": ["#/texts/0"],
                "item_geometry": [{"ref": "#/texts/0", "page_no": 1, "bbox": [1.0, 2.0, 3.0, 4.0]}],
                "token_count": 3,
            }
        ]
        ledger = chunk_index_ledger(chunks, metadata)
        assert ledger.version == 2
        record = ledger.chunks[0]
        assert len(record.item_geometry) == 1
        assert record.item_geometry[0].ref == "#/texts/0"
        assert record.item_geometry[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_ledger_roundtrips_through_json(self):
        ledger = chunk_index_ledger(
            ["text"],
            [
                {
                    "chunk_id": 0,
                    "page_numbers": [2],
                    "item_geometry": [
                        {"ref": "#/tables/1", "page_no": 2, "bbox": [5.0, 6.0, 7.0, 8.0]}
                    ],
                }
            ],
        )
        reloaded = ProvenanceLedger.model_validate_json(ledger.model_dump_json())
        assert reloaded.version == 2
        assert reloaded.chunks[0].item_geometry[0].page_no == 2

    def test_backward_compat_no_geometry_key(self):
        """A metadata dict without item_geometry yields an empty tuple (no error)."""
        ledger = chunk_index_ledger(["text"], [{"chunk_id": 0, "page_numbers": [1]}])
        assert ledger.chunks[0].item_geometry == ()


class TestGeometryForEntry:
    def test_collects_and_dedupes_geometry_across_anchors(self):
        g1 = ItemGeometry(ref="#/texts/1", page_no=1, bbox=(0, 0, 1, 1))
        g2 = ItemGeometry(ref="#/texts/2", page_no=1, bbox=(0, 1, 1, 2))
        ledger = ProvenanceLedger(
            chunks={
                0: ChunkRecord(chunk_id=0, batch_index=0, item_geometry=(g1, g2)),
                1: ChunkRecord(chunk_id=1, batch_index=0, item_geometry=(g1,)),  # dup g1
            },
        )
        entry = NodeProvenance(
            identity_key="k",
            anchors=[SourceAnchor(chunk_id=0), SourceAnchor(chunk_id=1)],
        )
        geoms = ledger.geometry_for_entry(entry)
        assert [g.ref for g in geoms] == ["#/texts/1", "#/texts/2"]

    def test_empty_when_no_geometry(self):
        ledger = ProvenanceLedger(
            chunks={0: ChunkRecord(chunk_id=0, batch_index=0)},
        )
        entry = NodeProvenance(identity_key="k", anchors=[SourceAnchor(chunk_id=0)])
        assert ledger.geometry_for_entry(entry) == ()
