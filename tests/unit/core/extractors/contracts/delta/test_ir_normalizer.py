from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.helpers import build_dedup_policy
from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
    DeltaIrNormalizerConfig,
    normalize_delta_ir_batch_results,
)


class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class RootDoc(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    people: list[Person] = Field(default_factory=list)


def test_normalizer_drops_unknown_paths_and_strips_nested_properties() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "unknown[]",
                        "ids": {"id": "x"},
                        "properties": {"name": "bad"},
                    },
                    {
                        "path": "people[]",
                        "ids": {"name": " Alice  "},
                        "parent": None,
                        "properties": {"name": "Alice", "nested": {"a": 1}},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    assert len(normalized) == 1
    assert len(normalized[0]["nodes"]) == 1
    node = normalized[0]["nodes"][0]
    assert node["ids"] == {"name": "Alice"}
    assert node["parent"]["path"] == ""
    assert "nested" not in node["properties"]
    assert node["provenance"]["page_numbers"] == [1]
    assert stats["unknown_path_dropped"] == 1
    assert stats["nested_property_dropped"] == 1


def test_normalizer_repairs_document_prefixed_paths() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "document",
                        "ids": {"document_number": "INV-1"},
                        "properties": {"document_number": "INV-1"},
                    },
                    {
                        "path": "document.people.1",
                        "ids": {"name": "Alice"},
                        "parent": {"path": "document", "ids": {"document_number": "INV-1"}},
                        "properties": {"name": "Alice"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    assert len(normalized) == 1
    assert len(normalized[0]["nodes"]) == 2
    paths = {n["path"] for n in normalized[0]["nodes"]}
    assert "" in paths
    assert "people[]" in paths
    assert stats["unknown_path_dropped"] == 0
    assert stats["path_alias_repaired"] >= 2


def test_normalizer_infers_parent_ids_from_indexed_child_path() -> None:
    class LineItem(BaseModel):
        model_config = ConfigDict(graph_id_fields=["line_number"])
        line_number: str
        item: Person | None = None

    class Invoice(BaseModel):
        model_config = ConfigDict(graph_id_fields=["document_number"])
        document_number: str
        line_items: list[LineItem] = Field(default_factory=list)

    catalog = build_delta_node_catalog(Invoice)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "document",
                        "ids": {"document_number": "INV-9"},
                        "properties": {"document_number": "INV-9"},
                    },
                    {
                        "path": "document.line_items.3",
                        "ids": {},
                        "parent": {"path": "document", "ids": {"document_number": "INV-9"}},
                        "properties": {"line_number": "3"},
                    },
                    {
                        "path": "document.line_items.3.item",
                        "ids": {"name": "Widget A"},
                        "parent": {"path": "document.line_items", "ids": {}},
                        "properties": {"name": "Widget A"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    item_node = next(n for n in normalized[0]["nodes"] if n["path"] == "line_items[].item")
    assert item_node["parent"]["path"] == "line_items[]"
    assert item_node["parent"]["ids"]["line_number"] == "3"
    assert stats["parent_id_inferred"] >= 1


def test_normalizer_repairs_alias_path_segments() -> None:
    class AliasDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["document_number"])
        document_number: str
        line_items: list[Person] = Field(
            default_factory=list,
            validation_alias=AliasChoices("line_items", "lineItems"),
        )

    catalog = build_delta_node_catalog(AliasDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "document",
                        "ids": {"document_number": "INV-2"},
                        "properties": {"document_number": "INV-2"},
                    },
                    {
                        "path": "document.lineItems.1",
                        "ids": {"name": "Alice"},
                        "parent": {"path": "document", "ids": {"document_number": "INV-2"}},
                        "properties": {"name": "Alice"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    paths = {n["path"] for n in normalized[0]["nodes"]}
    assert "" in paths
    assert "line_items[]" in paths
    assert stats["unknown_path_dropped"] == 0
    assert stats["path_alias_repaired"] >= 2


def test_normalizer_repairs_slash_paths_and_class_prefixes() -> None:
    catalog = build_delta_node_catalog(RootDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "RootDoc",
                        "ids": {"document_number": "INV-200"},
                        "properties": {"document_number": "INV-200"},
                    },
                    {
                        "path": "RootDoc/people/1",
                        "ids": {"name": "Alice"},
                        "parent": {"path": "RootDoc", "ids": {"document_number": "INV-200"}},
                        "properties": {"name": "Alice"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    assert len(normalized[0]["nodes"]) == 2
    paths = {n["path"] for n in normalized[0]["nodes"]}
    assert "" in paths
    assert "people[]" in paths
    assert stats["unknown_path_dropped"] == 0


def test_normalizer_coerces_common_scalar_formats() -> None:
    class ScalarDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["document_number"])
        document_number: str
        amount: float | None = None
        ratio: float | None = None
        event_date: str | None = None

    catalog = build_delta_node_catalog(ScalarDoc)
    policy = build_dedup_policy(catalog)
    normalized, _stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {
                        "path": "",
                        "ids": {"document_number": "INV-300"},
                        "properties": {
                            "document_number": "INV-300",
                            "amount": "CHF 3360.00",
                            "ratio": "7.7%",
                            "event_date": "18 May 2024",
                        },
                    }
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    node = normalized[0]["nodes"][0]
    assert node["properties"]["amount"] == 3360.0
    assert node["properties"]["ratio"] == 7.7
    assert node["properties"]["event_date"] == "2024-05-18"


def test_normalizer_repairs_missing_list_markers_on_intermediate_segments() -> None:
    class Experiment(BaseModel):
        model_config = ConfigDict(graph_id_fields=["experiment_id"])
        experiment_id: str

    class Study(BaseModel):
        model_config = ConfigDict(graph_id_fields=["study_id"])
        study_id: str
        experiments: list[Experiment] = Field(default_factory=list)

    class Paper(BaseModel):
        model_config = ConfigDict(graph_id_fields=["title"])
        title: str
        studies: list[Study] = Field(default_factory=list)

    catalog = build_delta_node_catalog(Paper)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {"path": "", "ids": {"title": "T1"}, "properties": {"title": "T1"}},
                    {
                        "path": "studies",
                        "ids": {"study_id": "S1"},
                        "parent": {"path": "", "ids": {"title": "T1"}},
                        "properties": {"study_id": "S1"},
                    },
                    {
                        "path": "studies.experiments",
                        "ids": {"experiment_id": "E1"},
                        "parent": {"path": "studies", "ids": {"study_id": "S1"}},
                        "properties": {"experiment_id": "E1"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    paths = {n["path"] for n in normalized[0]["nodes"]}
    assert "studies[]" in paths
    assert "studies[].experiments[]" in paths
    assert stats["unknown_path_dropped"] == 0


def test_normalizer_salvages_unknown_paths_and_repairs_properties_from_catalog() -> None:
    class MetaBlock(BaseModel):
        ref_code: str | None = Field(
            default=None, validation_alias=AliasChoices("ref_code", "refCode")
        )

    class GenericDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["document_number"])
        document_number: str
        metric_score: float | None = Field(
            default=None, validation_alias=AliasChoices("metric_score", "metricScore")
        )
        notes: str | None = None
        meta: MetaBlock | None = None

    catalog = build_delta_node_catalog(GenericDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {"path": "", "ids": {"document_number": "DOC-1"}, "properties": {}},
                    {"path": "meta", "ids": {}, "properties": {}},
                    {"path": "metricScore", "ids": {"value": "42.5"}, "properties": {}},
                    {"path": "notes", "ids": {"value": "Observed drift"}, "properties": {}},
                    {
                        "path": "meta.details",
                        "ids": {"refCode": "R-77"},
                        "properties": {"nested": {"refCode": "R-88"}},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )

    root_node = next(n for n in normalized[0]["nodes"] if n["path"] == "")
    meta_node = next(n for n in normalized[0]["nodes"] if n["path"] == "meta")
    assert root_node["properties"]["metric_score"] == 42.5
    assert root_node["properties"]["notes"] == "Observed drift"
    assert meta_node["properties"]["ref_code"] in {"R-77", "R-88"}
    assert stats["unknown_path_salvaged"] >= 2
    assert stats["salvaged_properties"] >= 2


def test_normalizer_backfills_single_identity_id_from_properties() -> None:
    class Offer(BaseModel):
        model_config = ConfigDict(graph_id_fields=["nom"])
        nom: str
        label: str | None = None

    class OfferDoc(BaseModel):
        model_config = ConfigDict(graph_id_fields=["reference_document"])
        reference_document: str
        offres: list[Offer] = Field(default_factory=list)

    catalog = build_delta_node_catalog(OfferDoc)
    policy = build_dedup_policy(catalog)
    normalized, stats = normalize_delta_ir_batch_results(
        batch_results=[
            {
                "nodes": [
                    {"path": "", "ids": {"reference_document": "DOC-9"}, "properties": {}},
                    {
                        "path": "offres[]",
                        "ids": {},
                        "parent": {"path": "", "ids": {"reference_document": "DOC-9"}},
                        "properties": {"nom": "PNO", "label": "Owner non-occupant"},
                    },
                ],
                "relationships": [],
            }
        ],
        batch_plan=[[(0, "chunk", 10)]],
        chunk_metadata=[{"page_numbers": [1], "token_count": 10}],
        catalog=catalog,
        dedup_policy=policy,
        config=DeltaIrNormalizerConfig(),
    )
    offer_node = next(n for n in normalized[0]["nodes"] if n["path"] == "offres[]")
    assert offer_node["ids"]["nom"] == "PNO"
    assert stats["id_backfilled_from_properties"] >= 1
