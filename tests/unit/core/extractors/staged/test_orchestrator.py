import json
from unittest.mock import MagicMock

from pydantic import BaseModel, ConfigDict, Field

from docling_graph.core.extractors.contracts.staged.orchestrator import StagedOrchestrator, StagedPassConfig


class Party(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str


class InvoiceTemplate(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_number"])
    document_number: str
    seller: Party = Field(..., json_schema_extra={"edge_label": "ISSUED_BY"})
    line_items: list[dict] = Field(default_factory=list)


def test_orchestrator_runs_skeleton_groups_repair():
    schema_json = json.dumps(InvoiceTemplate.model_json_schema(), indent=2)
    llm_call = MagicMock()
    llm_call.side_effect = [
        {"document_number": "3139"},  # skeleton
        {"seller": {"name": "Robert Schneider AG"}},  # group 0
        {"line_items": [{"description": "Garden work"}]},  # group 1
        {"line_items": [{"description": "Garden work"}]},  # repair round
    ]

    orchestrator = StagedOrchestrator(
        llm_call_fn=llm_call,
        schema_json=schema_json,
        template=InvoiceTemplate,
        config=StagedPassConfig(max_fields_per_group=1, max_skeleton_fields=1, max_repair_rounds=1),
    )
    merged = orchestrator.extract(markdown="invoice text", context="doc")

    assert merged is not None
    assert merged["document_number"] == "3139"
    assert merged["seller"]["name"] == "Robert Schneider AG"
    assert isinstance(merged["line_items"], list)
    assert llm_call.call_count >= 3


def test_orchestrator_retries_failed_pass_once():
    schema_json = json.dumps(InvoiceTemplate.model_json_schema(), indent=2)
    llm_call = MagicMock()
    llm_call.side_effect = [
        None,  # skeleton attempt 1
        {"document_number": "3139"},  # skeleton retry
        {"seller": {"name": "Robert Schneider AG"}},
        {"line_items": [{"description": "Garden work"}]},
    ]

    orchestrator = StagedOrchestrator(
        llm_call_fn=llm_call,
        schema_json=schema_json,
        template=InvoiceTemplate,
        config=StagedPassConfig(max_pass_retries=1, max_repair_rounds=0),
    )
    merged = orchestrator.extract(markdown="invoice text", context="doc")
    assert merged is not None
    assert merged["document_number"] == "3139"

