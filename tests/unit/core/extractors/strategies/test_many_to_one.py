"""
Unit tests for ManyToOneStrategy.

Tests the many-to-one strategy with direct full-document extraction:
- Direct extraction (single LLM call)
- VLM backend support
"""

from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from docling_graph.core.extractors.strategies.many_to_one import ManyToOneStrategy
from docling_graph.protocols import ExtractionBackendProtocol, TextExtractionBackendProtocol


class MockTemplate(BaseModel):
    """Simple test template."""

    name: str
    value: int = 0


@pytest.fixture
def mock_llm_backend():
    """Create a mock LLM backend."""
    backend = MagicMock(spec=TextExtractionBackendProtocol)
    backend.client = MagicMock()
    backend.__class__.__name__ = "MockLlmBackend"

    def mock_extract(markdown, template, context, is_partial) -> MockTemplate | None:
        if "fail" in markdown:
            return None
        return template(name=context, value=len(markdown))

    backend.extract_from_markdown.side_effect = mock_extract

    return backend


@pytest.fixture
def mock_vlm_backend():
    """Create a mock VLM backend."""
    backend = MagicMock(spec=ExtractionBackendProtocol)
    backend.__class__.__name__ = "MockVlmBackend"

    def mock_extract(source, template) -> List[MockTemplate]:
        if "single" in source:
            return [template(name="Page 1", value=10)]
        if "multi" in source:
            return [
                template(name="Page 1", value=10),
                template(name="Page 2", value=20),
            ]
        return []

    backend.extract_from_document.side_effect = mock_extract

    return backend


@pytest.fixture(autouse=True)
def patch_deps():
    """Patch common dependencies."""
    with (
        patch("docling_graph.core.extractors.strategies.many_to_one.DocumentProcessor") as mock_dp,
        patch(
            "docling_graph.core.extractors.strategies.many_to_one.merge_pydantic_models"
        ) as mock_merge,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_llm_backend") as mock_is_llm,
        patch("docling_graph.core.extractors.strategies.many_to_one.is_vlm_backend") as mock_is_vlm,
    ):
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.convert_to_docling_doc.return_value = "MockDoc"
        mock_doc_processor.extract_full_markdown.return_value = "full_doc_md"

        mock_merge.return_value = MockTemplate(name="Merged", value=123)

        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        yield mock_dp, mock_merge, mock_is_llm, mock_is_vlm


class TestInitialization:
    """Test strategy initialization."""

    def test_init_with_llm_backend(self, mock_llm_backend, patch_deps):
        """Test initialization with LLM backend."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)

        assert strategy.backend == mock_llm_backend

    def test_init_with_docling_config(self, mock_llm_backend, patch_deps):
        """Test initialization with custom docling config."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(
            backend=mock_llm_backend,
            docling_config="vision",
        )

        assert strategy.doc_processor is not None


class TestVLMExtraction:
    """Test VLM backend extraction."""

    def test_extract_single_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction for single-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("single_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Page 1"
        mock_merge.assert_not_called()

    def test_extract_multi_page(self, mock_vlm_backend, patch_deps):
        """Test VLM extraction and merge for multi-page document."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _document = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 1
        assert results[0].name == "Merged"
        mock_merge.assert_called_once()

    def test_merge_failure_returns_all_pages(self, mock_vlm_backend, patch_deps):
        """Test that VLM merge failure returns all page models (zero data loss)."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True

        mock_merge.return_value = None

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _ = strategy.extract("multi_page_doc.pdf", MockTemplate)

        assert len(results) == 2
        assert results[0].name == "Page 1"
        assert results[1].name == "Page 2"


class TestDirectExtraction:
    """Test direct extraction (single LLM call)."""

    def test_direct_full_document_extraction(self, mock_llm_backend, patch_deps):
        """Test direct full-document extraction."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "test content"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert mock_llm_backend.extract_from_markdown.called
        assert len(results) >= 0

    def test_attach_direct_provenance_upgrades_to_chunk_index(self, mock_llm_backend, patch_deps):
        """Direct extraction upgrades the document-level ledger to a chunk index
        (with text + pages) so the binder can locate nodes precisely (issue #1)."""
        from docling_graph.core.provenance import ProvenanceLedger, document_level_ledger

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        # Backend already produced a document-level ledger (provenance enabled).
        mock_llm_backend.last_provenance = document_level_ledger("whole doc text")
        strategy.doc_processor.chunker = MagicMock()  # a chunker is available
        strategy.doc_processor.extract_chunks_with_metadata = MagicMock(
            return_value=(
                ["chunk zero text", "chunk one text"],
                [
                    {"chunk_id": 0, "page_numbers": [1], "token_count": 3},
                    {"chunk_id": 1, "page_numbers": [2], "token_count": 3},
                ],
            )
        )

        strategy._attach_direct_provenance(mock_llm_backend, MagicMock())

        led = mock_llm_backend.last_provenance
        assert isinstance(led, ProvenanceLedger)
        assert led.node_level is False
        assert set(led.chunks) == {0, 1}
        assert led.chunks[1].text == "chunk one text"
        assert led.chunks[1].page_numbers == (2,)

    def test_attach_direct_provenance_noop_when_disabled(self, mock_llm_backend, patch_deps):
        """No backend ledger (provenance off) -> nothing attached."""
        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        mock_llm_backend.last_provenance = None
        strategy._attach_direct_provenance(mock_llm_backend, MagicMock())
        assert mock_llm_backend.last_provenance is None

    def test_direct_failure_returns_empty(self, mock_llm_backend, patch_deps):
        """Test that direct extraction returns empty list on failure."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "fail"

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, _ = strategy.extract("test.pdf", MockTemplate)

        assert len(results) == 0

    def test_extract_unknown_backend_returns_empty_and_none(self, mock_llm_backend, patch_deps):
        """Backend that is neither LLM nor VLM: TypeError is caught, returns [], None (106-117)."""
        _, _, mock_is_llm, mock_is_vlm = patch_deps
        mock_is_llm.return_value = False
        mock_is_vlm.return_value = False

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is None

    def test_vlm_no_models_returns_empty_and_none(self, mock_vlm_backend, patch_deps):
        """VLM extract_from_document returns empty list -> [], None (149-151)."""
        _, _, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True
        mock_vlm_backend.extract_from_document.side_effect = None
        mock_vlm_backend.extract_from_document.return_value = []

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, doc = strategy.extract("empty_doc.pdf", MockTemplate)

        assert results == []
        assert doc is None

    def test_vlm_exception_returns_empty_and_none(self, mock_vlm_backend, patch_deps):
        """VLM extract_from_document raises -> returns [], None and logger.error (148-153)."""
        _, _, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True
        mock_vlm_backend.extract_from_document.side_effect = RuntimeError("VLM failed")

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is None

    def test_vlm_merge_returns_none_returns_all_page_models(self, mock_vlm_backend, patch_deps):
        """VLM merge_pydantic_models returns None -> return all page models (144-146)."""
        _, mock_merge, _, mock_is_vlm = patch_deps
        mock_is_vlm.return_value = True
        page1 = MockTemplate(name="P1", value=1)
        page2 = MockTemplate(name="P2", value=2)
        mock_vlm_backend.extract_from_document.side_effect = None
        mock_vlm_backend.extract_from_document.return_value = [page1, page2]
        mock_merge.return_value = None

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        results, _ = strategy.extract("multi.pdf", MockTemplate)

        assert len(results) == 2
        assert results[0].name == "P1" and results[1].name == "P2"

    def test_extract_direct_mode_no_model_returns_empty_list_and_document(
        self, mock_llm_backend, patch_deps
    ):
        """Direct path: extract_from_markdown returns None -> [], document (462-464)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc = MagicMock()
        mock_dp.return_value.convert_to_docling_doc.return_value = mock_doc
        mock_dp.return_value.extract_full_markdown.return_value = "full md"
        mock_llm_backend.extract_from_markdown.side_effect = None
        mock_llm_backend.extract_from_markdown.return_value = None

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is mock_doc


class TestExtractWithLlm:
    """Test the _extract_with_llm conversion + delegation wrapper."""

    def test_conversion_exception_returns_empty_and_none(self, mock_llm_backend, patch_deps):
        """convert_to_docling_doc raising is caught by _extract_with_llm's own try/except (193-198)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.convert_to_docling_doc.side_effect = RuntimeError("conversion boom")

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, doc = strategy.extract("test.pdf", MockTemplate)

        assert results == []
        assert doc is None


class TestExtractFromDocument:
    """Test extraction from a pre-converted DoclingDocument."""

    def test_extract_from_document_uses_contract_path(self, mock_llm_backend, patch_deps):
        """extract_from_document skips conversion and runs the direct contract."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        results, document = strategy.extract_from_document("PreloadedDoc", MockTemplate)

        mock_dp.return_value.convert_to_docling_doc.assert_not_called()
        mock_dp.return_value.extract_full_markdown.assert_called_once_with("PreloadedDoc")
        assert len(results) == 1
        assert document == "PreloadedDoc"

    def test_extract_from_document_rejects_vlm_backend(self, mock_vlm_backend, patch_deps):
        """VLM backends need a source file, so DoclingDocument input must raise."""
        from docling_graph.exceptions import ExtractionError

        _, _, mock_is_llm, mock_is_vlm = patch_deps
        mock_is_llm.return_value = False
        mock_is_vlm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_vlm_backend)
        with pytest.raises(ExtractionError):
            strategy.extract_from_document("PreloadedDoc", MockTemplate)


class TestDenseFallback:
    """Dense returning no model must actually fall back to direct extraction."""

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_document")
    def test_dense_none_falls_back_to_direct(self, mock_dense, patch_deps):
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dense.return_value = (None, 0.1)

        backend = MagicMock()
        backend.__class__.__name__ = "MockLlmBackend"
        backend.extract_from_chunk_batches = MagicMock()
        backend.extract_from_markdown = MagicMock(return_value=MockTemplate(name="direct", value=1))

        strategy = ManyToOneStrategy(backend=backend, extraction_contract="dense")
        models, _document = strategy.extract("doc.pdf", MockTemplate)

        mock_dense.assert_called_once()
        backend.extract_from_markdown.assert_called_once()
        assert len(models) == 1
        assert models[0].name == "direct"

    @patch("docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_document")
    def test_dense_success_skips_direct(self, mock_dense, patch_deps):
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dense.return_value = (MockTemplate(name="dense", value=2), 0.1)

        backend = MagicMock()
        backend.__class__.__name__ = "MockLlmBackend"
        backend.extract_from_chunk_batches = MagicMock()
        backend.extract_from_markdown = MagicMock()

        strategy = ManyToOneStrategy(backend=backend, extraction_contract="dense")
        models, _document = strategy.extract("doc.pdf", MockTemplate)

        backend.extract_from_markdown.assert_not_called()
        assert models[0].name == "dense"


class TestValidation:
    """Test constructor validation and config branches."""

    def test_dense_without_chunking_raises(self, mock_llm_backend, patch_deps):
        """Dense contract requires chunking; disabling it must raise (line 67)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        with pytest.raises(ValueError, match="Dense extraction requires use_chunking=True"):
            ManyToOneStrategy(
                backend=mock_llm_backend,
                extraction_contract="dense",
                use_chunking=False,
            )

    def test_direct_without_chunking_uses_none_chunker_config(self, mock_llm_backend, patch_deps):
        """use_chunking=False with direct contract is fine; chunker_config stays None."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        ManyToOneStrategy(
            backend=mock_llm_backend,
            extraction_contract="direct",
            use_chunking=False,
        )

        _, kwargs = mock_dp.call_args
        assert kwargs["chunker_config"] is None


class TestDirectModeTraceData:
    """Cover trace_data emission and dense/direct branches in _extract_direct_mode."""

    def test_direct_mode_emits_trace_events_multi_page(self, mock_llm_backend, patch_deps):
        """Non-empty page_markdowns path emits per-page events + completion (354-371, 434-451)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "full md"
        mock_doc_processor.extract_page_markdowns.return_value = ["page one", "page two"]

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        strategy.trace_data = MagicMock()
        mock_llm_backend.last_call_diagnostics = {"tokens": 42}

        document = MagicMock()
        models, doc = strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        assert doc is document
        assert len(models) == 1

        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("page_markdown_extracted") == 2
        assert "docling_conversion_completed" in emitted_events
        assert "extraction_completed" in emitted_events

    def test_direct_mode_propagates_trace_data_to_backend(self, patch_deps):
        """When the backend declares a trace_data attribute, it is set to the strategy's (386-387)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.extract_full_markdown.return_value = "full md"
        mock_dp.return_value.extract_page_markdowns.return_value = []

        backend = MagicMock()  # no spec -> hasattr(backend, "trace_data") is True
        backend.__class__.__name__ = "MockLlmBackend"
        backend.extract_from_markdown.return_value = MockTemplate(name="ok", value=1)
        backend.last_call_diagnostics = None

        strategy = ManyToOneStrategy(backend=backend)
        strategy.trace_data = MagicMock()

        strategy._extract_direct_mode(backend, MagicMock(), MockTemplate)

        assert backend.trace_data is strategy.trace_data

    def test_direct_mode_emits_trace_events_no_pages(self, mock_llm_backend, patch_deps):
        """Empty page_markdowns falls back to a single synthetic page event (357-363)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_doc_processor = mock_dp.return_value
        mock_doc_processor.extract_full_markdown.return_value = "full md"
        mock_doc_processor.extract_page_markdowns.return_value = []

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        strategy.trace_data = MagicMock()

        document = MagicMock()
        strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("page_markdown_extracted") == 1
        first_page_call = strategy.trace_data.emit.call_args_list[0]
        assert first_page_call.args[2]["page_number"] == 1

    def test_direct_mode_dense_success_emits_completed_and_skips_direct_call(
        self, mock_llm_backend, patch_deps
    ):
        """Dense contract success path emits extraction_completed and returns early (389-419)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_llm_backend.extract_from_chunk_batches = MagicMock()
        mock_llm_backend.last_call_diagnostics = {"model": "dense-model"}

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_document"
        ) as mock_dense:
            mock_dense.return_value = (MockTemplate(name="dense", value=9), 0.2)

            strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="dense")
            strategy.trace_data = MagicMock()

            document = MagicMock()
            models, doc = strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        assert models[0].name == "dense"
        assert doc is document
        mock_llm_backend.extract_from_markdown.assert_not_called()

        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("extraction_completed") == 1
        completed_call = next(
            call
            for call in strategy.trace_data.emit.call_args_list
            if call.args[0] == "extraction_completed"
        )
        assert completed_call.args[2]["source_type"] == "chunk_batch"
        assert completed_call.args[2]["metadata"] == {"model": "dense-model"}

    def test_direct_mode_dense_none_falls_back_and_emits_both_completed_events(
        self, mock_llm_backend, patch_deps
    ):
        """Dense returns None -> falls through to direct call, emitting two completed events."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        mock_llm_backend.extract_from_chunk_batches = MagicMock()

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_document"
        ) as mock_dense:
            mock_dense.return_value = (None, 0.1)

            strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="dense")
            strategy.trace_data = MagicMock()

            document = MagicMock()
            models, _doc = strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        assert len(models) == 1
        mock_llm_backend.extract_from_markdown.assert_called_once()

        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("extraction_completed") == 2

    def test_direct_mode_exception_emits_extraction_failed(self, mock_llm_backend, patch_deps):
        """An exception inside the try block emits extraction_failed and returns [], document (461-477)."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.extract_full_markdown.side_effect = RuntimeError("boom")

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        strategy.trace_data = MagicMock()

        document = MagicMock()
        models, doc = strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        assert models == []
        assert doc is document
        strategy.trace_data.emit.assert_called_once_with(
            "extraction_failed",
            "extraction",
            {
                "extraction_id": 0,
                "source_type": "chunk",
                "source_id": 0,
                "parsed_model": None,
                "extraction_time": 0.0,
                "error": "boom",
                "metadata": {},
            },
        )

    def test_direct_mode_exception_without_trace_data_still_returns_empty(
        self, mock_llm_backend, patch_deps
    ):
        """Exception path with trace_data left as None must not attempt to emit."""
        mock_dp, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_dp.return_value.extract_full_markdown.side_effect = RuntimeError("boom")

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        assert strategy.trace_data is None

        document = MagicMock()
        models, doc = strategy._extract_direct_mode(mock_llm_backend, document, MockTemplate)

        assert models == []
        assert doc is document


class TestDirectModeFromTextTraceData:
    """Cover trace_data emission and dense/direct branches in _extract_direct_mode_from_text."""

    def test_from_text_emits_trace_events_direct_path(self, mock_llm_backend, patch_deps):
        """Direct (non-dense) contract emits page + conversion + completed events (230-245, 296-313)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.last_call_diagnostics = {"tokens": 7}

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        strategy.trace_data = MagicMock()

        models, doc = strategy._extract_direct_mode_from_text(
            mock_llm_backend, "some text", MockTemplate
        )

        assert doc is None
        assert len(models) == 1
        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert "page_markdown_extracted" in emitted_events
        assert "docling_conversion_completed" in emitted_events
        assert emitted_events.count("extraction_completed") == 1

    def test_from_text_propagates_trace_data_to_backend(self, patch_deps):
        """When the backend declares a trace_data attribute, it is set to the strategy's (250-251)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        backend = MagicMock()  # no spec -> hasattr(backend, "trace_data") is True
        backend.__class__.__name__ = "MockLlmBackend"
        backend.extract_from_markdown.return_value = MockTemplate(name="ok", value=1)
        backend.last_call_diagnostics = None

        strategy = ManyToOneStrategy(backend=backend)
        strategy.trace_data = MagicMock()

        strategy._extract_direct_mode_from_text(backend, "some text", MockTemplate)

        assert backend.trace_data is strategy.trace_data

    def test_from_text_dense_success_emits_completed_and_skips_direct_call(
        self, mock_llm_backend, patch_deps
    ):
        """Dense contract success emits extraction_completed with source_type chunk and returns early."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_chunk_batches = MagicMock()

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_text"
        ) as mock_dense:
            mock_dense.return_value = (MockTemplate(name="dense-text", value=3), 0.05)

            strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="dense")
            strategy.trace_data = MagicMock()

            models, doc = strategy._extract_direct_mode_from_text(
                mock_llm_backend, "some text", MockTemplate
            )

        assert doc is None
        assert models[0].name == "dense-text"
        mock_llm_backend.extract_from_markdown.assert_not_called()

        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("extraction_completed") == 1

    def test_from_text_dense_success_includes_backend_diagnostics(
        self, mock_llm_backend, patch_deps
    ):
        """Non-empty last_call_diagnostics on the backend is merged into emitted metadata (266)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_chunk_batches = MagicMock()
        mock_llm_backend.last_call_diagnostics = {"tokens": 99}

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_text"
        ) as mock_dense:
            mock_dense.return_value = (MockTemplate(name="dense-text", value=3), 0.05)

            strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="dense")
            strategy.trace_data = MagicMock()

            strategy._extract_direct_mode_from_text(mock_llm_backend, "some text", MockTemplate)

        completed_call = next(
            call
            for call in strategy.trace_data.emit.call_args_list
            if call.args[0] == "extraction_completed"
        )
        assert completed_call.args[2]["metadata"] == {"tokens": 99}

    def test_from_text_dense_none_falls_back_to_direct(self, mock_llm_backend, patch_deps):
        """Dense returns None -> falls through to backend.extract_from_markdown (283-294)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_chunk_batches = MagicMock()

        with patch(
            "docling_graph.core.extractors.strategies.many_to_one.extract_dense_from_text"
        ) as mock_dense:
            mock_dense.return_value = (None, 0.05)

            strategy = ManyToOneStrategy(backend=mock_llm_backend, extraction_contract="dense")
            strategy.trace_data = MagicMock()

            models, doc = strategy._extract_direct_mode_from_text(
                mock_llm_backend, "some text", MockTemplate
            )

        assert doc is None
        assert len(models) == 1
        mock_llm_backend.extract_from_markdown.assert_called_once()
        emitted_events = [call.args[0] for call in strategy.trace_data.emit.call_args_list]
        assert emitted_events.count("extraction_completed") == 2

    def test_from_text_direct_returns_no_model(self, mock_llm_backend, patch_deps):
        """Direct call returning None -> [], None (318-320)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_markdown.side_effect = None
        mock_llm_backend.extract_from_markdown.return_value = None

        strategy = ManyToOneStrategy(backend=mock_llm_backend)

        models, doc = strategy._extract_direct_mode_from_text(
            mock_llm_backend, "some text", MockTemplate
        )

        assert models == []
        assert doc is None

    def test_from_text_exception_emits_extraction_failed(self, mock_llm_backend, patch_deps):
        """Exception inside the try block emits extraction_failed and returns [], None (322-338)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_markdown.side_effect = RuntimeError("text boom")

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        strategy.trace_data = MagicMock()

        models, doc = strategy._extract_direct_mode_from_text(
            mock_llm_backend, "some text", MockTemplate
        )

        assert models == []
        assert doc is None
        strategy.trace_data.emit.assert_called_with(
            "extraction_failed",
            "extraction",
            {
                "extraction_id": 0,
                "source_type": "chunk",
                "source_id": 0,
                "parsed_model": None,
                "extraction_time": 0.0,
                "error": "text boom",
                "metadata": {},
            },
        )

    def test_from_text_exception_without_trace_data_still_returns_empty(
        self, mock_llm_backend, patch_deps
    ):
        """Exception path with trace_data left as None must not attempt to emit."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True
        mock_llm_backend.extract_from_markdown.side_effect = RuntimeError("text boom")

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        assert strategy.trace_data is None

        models, doc = strategy._extract_direct_mode_from_text(
            mock_llm_backend, "some text", MockTemplate
        )

        assert models == []
        assert doc is None

    def test_extract_with_llm_from_text_delegates(self, mock_llm_backend, patch_deps):
        """Public wrapper _extract_with_llm_from_text delegates to the contract-driven method."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        models, doc = strategy._extract_with_llm_from_text(
            mock_llm_backend, "hello text", MockTemplate
        )

        assert doc is None
        assert len(models) == 1

    def test_extract_with_llm_from_text_exception_returns_empty(self, mock_llm_backend, patch_deps):
        """A raise inside _extract_direct_mode_from_text bubbling past its own try is still caught
        by the outer wrapper's except block (211-218)."""
        _, _, mock_is_llm, _ = patch_deps
        mock_is_llm.return_value = True

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        with patch.object(
            strategy, "_extract_direct_mode_from_text", side_effect=RuntimeError("outer boom")
        ):
            models, doc = strategy._extract_with_llm_from_text(
                mock_llm_backend, "hello text", MockTemplate
            )

        assert models == []
        assert doc is None


class TestAttachDirectProvenanceEdgeCases:
    """Cover the early-return and error branches of _attach_direct_provenance."""

    def test_no_chunker_keeps_document_level_fallback(self, mock_llm_backend, patch_deps):
        """chunker is None -> early return, ledger left untouched (492-493)."""
        from docling_graph.core.provenance import document_level_ledger

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        original_ledger = document_level_ledger("whole doc text")
        mock_llm_backend.last_provenance = original_ledger
        strategy.doc_processor.chunker = None

        strategy._attach_direct_provenance(mock_llm_backend, MagicMock())

        assert mock_llm_backend.last_provenance is original_ledger

    def test_chunking_exception_logs_and_returns(self, mock_llm_backend, patch_deps):
        """extract_chunks_with_metadata raising is logged and swallowed (494-498)."""
        from docling_graph.core.provenance import document_level_ledger

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        original_ledger = document_level_ledger("whole doc text")
        mock_llm_backend.last_provenance = original_ledger
        strategy.doc_processor.chunker = MagicMock()
        strategy.doc_processor.extract_chunks_with_metadata = MagicMock(
            side_effect=RuntimeError("chunk boom")
        )

        strategy._attach_direct_provenance(mock_llm_backend, MagicMock())

        assert mock_llm_backend.last_provenance is original_ledger

    def test_empty_chunks_keeps_document_level_fallback(self, mock_llm_backend, patch_deps):
        """Empty chunk list -> early return (499-500)."""
        from docling_graph.core.provenance import document_level_ledger

        strategy = ManyToOneStrategy(backend=mock_llm_backend)
        original_ledger = document_level_ledger("whole doc text")
        mock_llm_backend.last_provenance = original_ledger
        strategy.doc_processor.chunker = MagicMock()
        strategy.doc_processor.extract_chunks_with_metadata = MagicMock(return_value=([], []))

        strategy._attach_direct_provenance(mock_llm_backend, MagicMock())

        assert mock_llm_backend.last_provenance is original_ledger
