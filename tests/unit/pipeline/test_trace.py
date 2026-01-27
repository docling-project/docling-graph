"""
Unit tests for legacy trace data classes.

NOTE: These classes (TraceData, PageData, ChunkData, etc.) are part of the old trace system
and are no longer actively used in the pipeline. They remain in the codebase for backward
compatibility but may be removed in a future version.

The new debug system uses OutputDirectoryManager instead.
"""

import networkx as nx
import pytest
from pydantic import BaseModel

from docling_graph.pipeline.trace import (
    ChunkData,
    ExtractionData,
    GraphData,
    PageData,
    TraceData,
)


class TestPageData:
    """Tests for PageData class."""

    def test_page_data_creation(self):
        """Test basic PageData creation."""
        page = PageData(page_number=1, text_content="Test content", metadata={"key": "value"})
        assert page.page_number == 1
        assert page.text_content == "Test content"
        assert page.metadata == {"key": "value"}

    def test_page_data_with_empty_metadata(self):
        """Test PageData with empty metadata."""
        page = PageData(page_number=2, text_content="Content", metadata={})
        assert page.page_number == 2
        assert page.metadata == {}

    def test_page_data_with_complex_metadata(self):
        """Test PageData with complex metadata."""
        page = PageData(
            page_number=1,
            text_content="Test",
            metadata={"page_size": 1024, "has_tables": True, "has_images": False, "language": "en"},
        )
        assert page.metadata["page_size"] == 1024
        assert page.metadata["has_tables"] is True


class TestChunkData:
    """Tests for ChunkData class."""

    def test_chunk_data_creation(self):
        """Test basic ChunkData creation."""
        chunk = ChunkData(
            chunk_id=0,
            text_content="Chunk text",
            page_numbers=[1, 2],
            token_count=150,
            metadata={"source": "test"},
        )
        assert chunk.chunk_id == 0
        assert chunk.text_content == "Chunk text"
        assert chunk.page_numbers == [1, 2]
        assert chunk.token_count == 150

    def test_chunk_data_single_page(self):
        """Test ChunkData spanning single page."""
        chunk = ChunkData(
            chunk_id=1,
            text_content="Single page chunk",
            page_numbers=[3],
            token_count=100,
            metadata={},
        )
        assert len(chunk.page_numbers) == 1
        assert chunk.page_numbers[0] == 3

    def test_chunk_data_multiple_pages(self):
        """Test ChunkData spanning multiple pages."""
        chunk = ChunkData(
            chunk_id=2,
            text_content="Multi-page chunk",
            page_numbers=[1, 2, 3, 4],
            token_count=500,
            metadata={},
        )
        assert len(chunk.page_numbers) == 4
        assert chunk.page_numbers == [1, 2, 3, 4]


class TestExtractionData:
    """Tests for ExtractionData class."""

    def test_extraction_data_success(self):
        """Test ExtractionData for successful extraction."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)

        extraction = ExtractionData(
            extraction_id=0,
            source_type="chunk",
            source_id=1,
            parsed_model=model,
            extraction_time=1.5,
            error=None,
        )

        assert extraction.extraction_id == 0
        assert extraction.source_type == "chunk"
        assert extraction.source_id == 1
        assert extraction.parsed_model == model
        assert extraction.extraction_time == 1.5
        assert extraction.error is None
        assert extraction.metadata == {}

    def test_extraction_data_with_error(self):
        """Test ExtractionData with error."""
        extraction = ExtractionData(
            extraction_id=1,
            source_type="page",
            source_id=2,
            parsed_model=None,
            extraction_time=0.5,
            error="Extraction failed: timeout",
        )

        assert extraction.extraction_id == 1
        assert extraction.parsed_model is None
        assert extraction.error == "Extraction failed: timeout"

    def test_extraction_data_page_source(self):
        """Test ExtractionData with page source type."""
        extraction = ExtractionData(
            extraction_id=2,
            source_type="page",
            source_id=0,
            parsed_model=None,
            extraction_time=2.0,
            error=None,
        )

        assert extraction.source_type == "page"
        assert extraction.source_id == 0


class TestGraphData:
    """Tests for GraphData class."""

    def test_graph_data_creation(self):
        """Test basic GraphData creation."""

        class TestModel(BaseModel):
            name: str

        model = TestModel(name="test")
        graph = nx.DiGraph()
        graph.add_node("node1", label="Test")
        graph.add_edge("node1", "node2", label="RELATES_TO")

        graph_data = GraphData(
            graph_id=0,
            source_type="chunk",
            source_id=1,
            graph=graph,
            pydantic_model=model,
            node_count=2,
            edge_count=1,
        )

        assert graph_data.graph_id == 0
        assert graph_data.source_type == "chunk"
        assert graph_data.source_id == 1
        assert graph_data.node_count == 2
        assert graph_data.edge_count == 1
        assert isinstance(graph_data.graph, nx.DiGraph)

    def test_graph_data_empty_graph(self):
        """Test GraphData with empty graph."""
        graph = nx.DiGraph()

        graph_data = GraphData(
            graph_id=1,
            source_type="page",
            source_id=0,
            graph=graph,
            pydantic_model=None,
            node_count=0,
            edge_count=0,
        )

        assert graph_data.node_count == 0
        assert graph_data.edge_count == 0
        assert graph_data.graph.number_of_nodes() == 0


class TestTraceData:
    """Tests for TraceData class."""

    def test_trace_data_initialization(self):
        """Test TraceData initialization."""
        trace = TraceData()

        assert trace.pages == []
        assert trace.chunks is None
        assert trace.extractions == []
        assert trace.intermediate_graphs == []

    def test_trace_data_add_pages(self):
        """Test adding pages to TraceData."""
        trace = TraceData()

        page1 = PageData(page_number=1, text_content="Page 1", metadata={})
        page2 = PageData(page_number=2, text_content="Page 2", metadata={})

        trace.pages.append(page1)
        trace.pages.append(page2)

        assert len(trace.pages) == 2
        assert trace.pages[0].page_number == 1
        assert trace.pages[1].page_number == 2

    def test_trace_data_add_chunks(self):
        """Test adding chunks to TraceData."""
        trace = TraceData()
        trace.chunks = []

        chunk1 = ChunkData(
            chunk_id=0, text_content="Chunk 1", page_numbers=[1], token_count=100, metadata={}
        )
        chunk2 = ChunkData(
            chunk_id=1, text_content="Chunk 2", page_numbers=[2], token_count=150, metadata={}
        )

        trace.chunks.append(chunk1)
        trace.chunks.append(chunk2)

        assert len(trace.chunks) == 2
        assert trace.chunks[0].chunk_id == 0
        assert trace.chunks[1].chunk_id == 1

    def test_trace_data_add_extractions(self):
        """Test adding extractions to TraceData."""
        trace = TraceData()

        extraction1 = ExtractionData(
            extraction_id=0,
            source_type="chunk",
            source_id=0,
            parsed_model=None,
            extraction_time=1.0,
            error=None,
        )
        extraction2 = ExtractionData(
            extraction_id=1,
            source_type="chunk",
            source_id=1,
            parsed_model=None,
            extraction_time=1.5,
            error=None,
        )

        trace.extractions.append(extraction1)
        trace.extractions.append(extraction2)

        assert len(trace.extractions) == 2
        assert trace.extractions[0].extraction_id == 0
        assert trace.extractions[1].extraction_id == 1

    def test_trace_data_add_intermediate_graphs(self):
        """Test adding intermediate graphs to TraceData."""
        trace = TraceData()

        graph1 = nx.DiGraph()
        graph1.add_node("n1")

        graph_data1 = GraphData(
            graph_id=0,
            source_type="chunk",
            source_id=0,
            graph=graph1,
            pydantic_model=None,
            node_count=1,
            edge_count=0,
        )

        trace.intermediate_graphs.append(graph_data1)

        assert len(trace.intermediate_graphs) == 1
        assert trace.intermediate_graphs[0].graph_id == 0

    def test_trace_data_complete_workflow(self):
        """Test complete TraceData workflow with all data types."""
        trace = TraceData()

        # Add pages
        trace.pages.append(PageData(page_number=1, text_content="P1", metadata={}))
        trace.pages.append(PageData(page_number=2, text_content="P2", metadata={}))

        # Add chunks
        trace.chunks = []
        trace.chunks.append(
            ChunkData(chunk_id=0, text_content="C1", page_numbers=[1], token_count=100, metadata={})
        )

        # Add extractions
        trace.extractions.append(
            ExtractionData(
                extraction_id=0,
                source_type="chunk",
                source_id=0,
                parsed_model=None,
                extraction_time=1.0,
                error=None,
            )
        )

        # Add intermediate graphs
        graph = nx.DiGraph()
        trace.intermediate_graphs.append(
            GraphData(
                graph_id=0,
                source_type="chunk",
                source_id=0,
                graph=graph,
                pydantic_model=None,
                node_count=0,
                edge_count=0,
            )
        )

        # Verify all data is present
        assert len(trace.pages) == 2
        assert len(trace.chunks) == 1
        assert len(trace.extractions) == 1
        assert len(trace.intermediate_graphs) == 1
