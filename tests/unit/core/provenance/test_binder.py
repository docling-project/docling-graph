"""Tests for ledger-to-graph binding (spec hook H9) and cleaner provenance union (H10)."""

import networkx as nx
from pydantic import BaseModel, ConfigDict

from docling_graph.core.converters.graph_converter import GraphConverter
from docling_graph.core.converters.node_id_registry import NodeIDRegistry
from docling_graph.core.provenance import (
    PROVENANCE_NODE_ATTR,
    ChunkRecord,
    DocumentOrigin,
    NodeProvenance,
    ProvenanceLedger,
    SourceAnchor,
)
from docling_graph.core.provenance.binder import bind_provenance
from docling_graph.core.utils.graph_cleaner import GraphCleaner
from tests.fixtures.sample_templates.test_template import SampleCompany, SamplePerson


class TwoFieldEntity(BaseModel):
    """Root entity with a two-field identity, used only to exercise a verbatim
    match that unions hits across id fields into more chunks than the compact
    view's per-node anchor cap."""

    field_a: str
    field_b: str

    model_config = ConfigDict(graph_id_fields=["field_a", "field_b"])


def _company_ledger(
    chunk_text: dict[int, str] | None = None, **extra_nodes: NodeProvenance
) -> ProvenanceLedger:
    """A dense-style ledger (node_level=True) with skeleton observed anchors.

    ``chunk_text`` optionally supplies chunk texts so the binder's verbatim
    locator can run; by default chunks carry no text (observed-only path).
    """
    chunk_text = chunk_text or {}
    root_entry = NodeProvenance(
        identity_key="|company_name=acme",
        catalog_path="",
        node_type="SampleCompany",
        ids={"company_name": "Acme"},
        anchors=[SourceAnchor(chunk_id=0), SourceAnchor(chunk_id=1)],
        notes=["scope:document"],
    )
    emp_entry = NodeProvenance(
        identity_key="employees[]|email=janeacmecom",
        catalog_path="employees[]",
        node_type="SamplePerson",
        ids={"email": "jane@acme.com"},
        anchors=[SourceAnchor(chunk_id=1)],
    )
    nodes = {root_entry.identity_key: root_entry, emp_entry.identity_key: emp_entry}
    nodes.update({e.identity_key: e for e in extra_nodes.values()})
    return ProvenanceLedger(
        node_level=True,
        document=DocumentOrigin(document_id="doc-123", source="x.pdf"),
        chunks={
            0: ChunkRecord(
                chunk_id=0, batch_index=0, page_numbers=(1,), text=chunk_text.get(0, "")
            ),
            1: ChunkRecord(
                chunk_id=1, batch_index=1, page_numbers=(2,), text=chunk_text.get(1, "")
            ),
        },
        nodes=nodes,
    )


def _company(email: str = "jane@acme.com") -> SampleCompany:
    return SampleCompany(
        company_name="Acme",
        industry="Robotics",
        founded_year=1999,
        employees=[SamplePerson(first_name="Jane", last_name="Doe", email=email)],
    )


def _convert_with_binding(
    company: SampleCompany,
    ledger: ProvenanceLedger,
    *,
    include_spans: bool = False,
    template: type = SampleCompany,
) -> tuple[nx.DiGraph, NodeIDRegistry, dict[str, int]]:
    registry = NodeIDRegistry()
    converter = GraphConverter(registry=registry)
    stats: dict[str, int] = {}

    def binder(graph: nx.DiGraph, models: list) -> None:
        stats.update(
            bind_provenance(
                graph=graph,
                models=models,
                ledger=ledger,
                registry=registry,
                template=template,
                include_spans=include_spans,
            )
        )

    graph, _ = converter.pydantic_list_to_graph([company], provenance_binder=binder)
    return graph, registry, stats


class TestBindProvenance:
    def test_observed_binding_without_chunk_text(self):
        # No chunk text -> the verbatim locator finds nothing, so nodes fall
        # back to their skeleton (observed) anchors.
        ledger = _company_ledger()
        company = _company()
        graph, registry, stats = _convert_with_binding(company, ledger)

        root_id = registry.get_node_id(company)
        emp_id = registry.get_node_id(company.employees[0])
        assert stats["bound_observed"] == 2
        assert stats["unresolved"] == 0
        # Root is document-scoped
        assert graph.nodes[root_id][PROVENANCE_NODE_ATTR] == {
            "document_id": "doc-123",
            "scope": "document",
        }
        # Nested entity carries approximate chunk/page grounding
        emp_view = graph.nodes[emp_id][PROVENANCE_NODE_ATTR]
        assert emp_view["chunks"] == [1]
        assert emp_view["pages"] == [2]
        assert emp_view["match"] == "observed"
        assert emp_view["approximate"] is True

    def test_verbatim_binding_pinpoints_exact_chunk(self):
        # The employee's email appears verbatim in chunk 1 -> exact location,
        # even though the skeleton observed anchor is on chunk 1 too.
        ledger = _company_ledger(chunk_text={0: "intro about Acme", 1: "reach jane@acme.com now"})
        company = _company()
        graph, registry, stats = _convert_with_binding(company, ledger)

        emp_id = registry.get_node_id(company.employees[0])
        view = graph.nodes[emp_id][PROVENANCE_NODE_ATTR]
        assert view["match"] == "verbatim"
        assert view["chunks"] == [1]
        assert view["pages"] == [2]
        assert "approximate" not in view
        assert stats["bound_verbatim"] == 1
        # The verbatim anchor is recorded back into the ledger entry.
        emp_entry = ledger.nodes["employees[]|email=janeacmecom"]
        assert any(a.kind == "verbatim" for a in emp_entry.anchors)
        assert ledger.resolution == "span"

    def test_unresolved_marks_dense_node_without_guessing(self):
        # Dense ledger (node_level=True): an unmatched, unlocatable node is
        # unresolved, never a wrong attribution.
        ledger = _company_ledger()
        company = _company(email="nobody@else.org")  # never observed, not in any chunk text
        graph, registry, stats = _convert_with_binding(company, ledger)

        emp_id = registry.get_node_id(company.employees[0])
        assert graph.nodes[emp_id][PROVENANCE_NODE_ATTR] == {"status": "unresolved"}
        assert stats["unresolved"] == 1
        assert stats["bound_observed"] == 1  # root still binds

    def test_fuzzy_containment_fallback_within_same_path(self):
        # Ledger recorded a longer identifier than the validated model carries;
        # no chunk text, so binding falls to the fuzzy-matched observed entry.
        ledger = _company_ledger()
        long_entry = NodeProvenance(
            identity_key="employees[]|email=janedoeacmecorpcom",
            catalog_path="employees[]",
            node_type="SamplePerson",
            ids={"email": "jane.doe@acmecorp.com"},
            anchors=[SourceAnchor(chunk_id=0)],
        )
        ledger.nodes.pop("employees[]|email=janeacmecom")
        ledger.nodes[long_entry.identity_key] = long_entry

        company = _company(email="doe@acmecorp.com")  # canonical text contained in entry's
        graph, registry, stats = _convert_with_binding(company, ledger)

        emp_id = registry.get_node_id(company.employees[0])
        view = graph.nodes[emp_id][PROVENANCE_NODE_ATTR]
        assert view["chunks"] == [0]
        assert stats["bound_observed"] == 2

    def test_bind_stats_stored_on_ledger(self):
        ledger = _company_ledger()
        _convert_with_binding(_company(), ledger)
        assert ledger.bind_stats["bound_observed"] == 2

    def test_document_scope_ledger_grounds_every_entity(self):
        from docling_graph.core.provenance import document_level_ledger

        ledger = document_level_ledger("full document text")
        ledger.document = DocumentOrigin(document_id="doc-9", source="x.pdf")
        company = _company()
        graph, registry, stats = _convert_with_binding(company, ledger)

        # Both root and nested entity get a document-scope view (never unresolved)
        for node_id in (registry.get_node_id(company), registry.get_node_id(company.employees[0])):
            assert graph.nodes[node_id][PROVENANCE_NODE_ATTR] == {
                "document_id": "doc-9",
                "scope": "document",
            }
        assert stats["bound_document"] == 2
        assert stats["unresolved"] == 0

    def test_root_grounded_via_distinctive_nonidentity_field(self):
        # V2: the root is document-scoped by identity, but a distinctive
        # non-identity attribute (industry) appears verbatim -> the root is
        # pinned to that exact chunk instead of staying whole-document.
        ledger = _company_ledger(
            chunk_text={0: "we operate in the Aerospace-Robotics sector", 1: "team page"}
        )
        company = SampleCompany(
            company_name="Acme",
            industry="Aerospace-Robotics",
            founded_year=1999,
            employees=[SamplePerson(first_name="Jane", last_name="Doe", email="jane@acme.com")],
        )
        graph, registry, _ = _convert_with_binding(company, ledger)

        root_view = graph.nodes[registry.get_node_id(company)][PROVENANCE_NODE_ATTR]
        assert root_view["match"] == "verbatim"
        assert root_view["chunks"] == [0]
        assert root_view["pages"] == [1]

    def test_distinctive_field_grounding_includes_spans_in_detailed_mode(self):
        # V1 + detailed mode: a located non-identity field surfaces char spans.
        ledger = _company_ledger(
            chunk_text={0: "led by Jane Zylkowski, principal engineer", 1: "team page"}
        )
        company = SampleCompany(
            company_name="Acme",
            industry="Robotics",
            founded_year=1999,
            employees=[
                SamplePerson(first_name="Jane", last_name="Zylkowski", email="ghost@nowhere.test")
            ],
        )
        graph, registry, _ = _convert_with_binding(company, ledger, include_spans=True)

        emp_view = graph.nodes[registry.get_node_id(company.employees[0])][PROVENANCE_NODE_ATTR]
        assert emp_view["match"] == "verbatim"
        assert emp_view["spans"]
        assert emp_view["spans"][0]["chunk"] == 0

    def test_verbatim_match_across_two_id_fields_caps_and_omits_chunks(self):
        # A two-field identity's verbatim hits union across BOTH fields; when the
        # union exceeds the compact view's per-node anchor cap (8), the excess is
        # reported via chunks_omitted rather than silently truncated.
        chunk_text = {i: f"AAACODE appears here (chunk {i})" for i in range(6)}
        chunk_text.update({6 + i: f"BBBCODE appears here (chunk {6 + i})" for i in range(3)})
        chunks = {
            i: ChunkRecord(chunk_id=i, batch_index=0, page_numbers=(i + 1,), text=text)
            for i, text in chunk_text.items()
        }
        ledger = ProvenanceLedger(
            node_level=True,
            document=DocumentOrigin(document_id="doc-2f", source="x.pdf"),
            chunks=chunks,
        )
        entity = TwoFieldEntity(field_a="AAACODE", field_b="BBBCODE")
        graph, registry, stats = _convert_with_binding(entity, ledger, template=TwoFieldEntity)

        view = graph.nodes[registry.get_node_id(entity)][PROVENANCE_NODE_ATTR]
        assert view["match"] == "verbatim"
        assert len(view["chunks"]) == 8
        assert view["chunks_omitted"] == 1
        assert stats["bound_verbatim"] == 1


class TestDirectContractBinding:
    """Direct contract (issue #1): a chunk-index ledger with no per-node entries;
    the binder verbatim-locates each node, falling back to document scope."""

    def _direct_ledger(self, chunk_text: dict[int, str]) -> ProvenanceLedger:
        from docling_graph.core.provenance import chunk_index_ledger

        chunks = [chunk_text[i] for i in sorted(chunk_text)]
        metadata = [
            {"chunk_id": i, "page_numbers": [i + 1], "token_count": 5} for i in sorted(chunk_text)
        ]
        ledger = chunk_index_ledger(chunks, metadata)
        ledger.document = DocumentOrigin(document_id="doc-d", source="x.pdf")
        return ledger

    def test_node_located_verbatim_and_missing_node_document_scope(self):
        # Employee email appears in chunk 1; company name "Acme" is document-wide.
        ledger = self._direct_ledger({0: "intro paragraph", 1: "email jane@acme.com here"})
        company = _company()
        graph, registry, stats = _convert_with_binding(company, ledger)

        emp_id = registry.get_node_id(company.employees[0])
        emp_view = graph.nodes[emp_id][PROVENANCE_NODE_ATTR]
        assert emp_view["match"] == "verbatim"
        assert emp_view["chunks"] == [1]
        assert emp_view["pages"] == [2]

        # The company name isn't findable as a distinctive line here -> document scope.
        root_id = registry.get_node_id(company)
        root_view = graph.nodes[root_id][PROVENANCE_NODE_ATTR]
        assert root_view.get("scope") == "document" or root_view.get("match") == "verbatim"
        assert "unresolved" not in {
            graph.nodes[n].get(PROVENANCE_NODE_ATTR, {}).get("status") for n in graph.nodes
        }
        assert stats["bound_verbatim"] >= 1

    def test_distinctive_field_grounds_node_when_id_absent(self):
        # V1: the employee's identifier (email) is not in the text, but a
        # distinctive non-identity field (last_name) is -> verbatim grounding
        # instead of a document-scope fallback.
        ledger = self._direct_ledger(
            {0: "intro paragraph", 1: "signed by Zylkowski, chief engineer"}
        )
        company = SampleCompany(
            company_name="Acme",
            industry="Robotics",
            founded_year=1999,
            employees=[
                SamplePerson(first_name="Jane", last_name="Zylkowski", email="ghost@nowhere.test")
            ],
        )
        graph, registry, stats = _convert_with_binding(company, ledger)

        emp_view = graph.nodes[registry.get_node_id(company.employees[0])][PROVENANCE_NODE_ATTR]
        assert emp_view["match"] == "verbatim"
        assert emp_view["chunks"] == [1]
        assert stats["bound_verbatim"] >= 1


class TestCleanerProvenanceUnion:
    def _node_attrs(self, name: str, prov: dict) -> dict:
        return {
            "label": "Thing",
            "type": "entity",
            "__class__": "Thing",
            "name": name,
            PROVENANCE_NODE_ATTR: prov,
        }

    def test_duplicate_merge_unions_provenance(self):
        graph = nx.DiGraph()
        graph.add_node(
            "Thing_1",
            id="Thing_1",
            **self._node_attrs(
                "X", {"document_id": "d", "match": "observed", "chunks": [1], "pages": [1]}
            ),
        )
        graph.add_node(
            "Thing_2",
            id="Thing_2",
            **self._node_attrs(
                "X", {"document_id": "d", "match": "verbatim", "chunks": [2], "pages": [2]}
            ),
        )
        cleaner = GraphCleaner(verbose=False)
        merged = cleaner._deduplicate_nodes(graph)
        assert merged == 1
        survivor = next(iter(graph.nodes))
        view = graph.nodes[survivor][PROVENANCE_NODE_ATTR]
        assert view["chunks"] == [1, 2]
        assert view["pages"] == [1, 2]
        assert view["match"] == "verbatim"

    def test_content_hash_ignores_provenance(self):
        cleaner = GraphCleaner(verbose=False)
        base = {"label": "T", "type": "entity", "__class__": "T", "name": "X"}
        h1 = cleaner._compute_content_hash(
            {**base, PROVENANCE_NODE_ATTR: {"chunks": [1]}}, node_id="a"
        )
        h2 = cleaner._compute_content_hash(
            {**base, PROVENANCE_NODE_ATTR: {"chunks": [2]}}, node_id="b"
        )
        assert h1 == h2
