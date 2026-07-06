"""Tests for pipeline stages."""

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from docling_graph.core import PipelineConfig
from docling_graph.exceptions import ConfigurationError, ExtractionError, PipelineError
from docling_graph.pipeline.context import PipelineContext
from docling_graph.pipeline.stages import (
    DoclingExportStage,
    ExportStage,
    ExtractionStage,
    GraphConversionStage,
    TemplateLoadingStage,
    VisualizationStage,
)


class TestTemplateLoadingStage:
    """Test suite for TemplateLoadingStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = TemplateLoadingStage()
        assert stage.name() == "Template Loading"

    def test_load_template_from_string(self):
        """Test loading template from string path."""
        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        stage = TemplateLoadingStage()
        result = stage.execute(context)

        assert result.template is not None
        assert result.template.__name__ == "BaseModel"

    def test_load_template_from_class(self):
        """Test loading template from class directly."""
        from pydantic import BaseModel

        config = PipelineConfig(
            source="test.pdf", template=BaseModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        stage = TemplateLoadingStage()
        result = stage.execute(context)

        assert result.template is BaseModel

    def test_invalid_template_path_raises_error(self):
        """Test that invalid template path raises ConfigurationError."""
        config = PipelineConfig(
            source="test.pdf", template="invalid.module.Template", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        stage = TemplateLoadingStage()

        with pytest.raises(ConfigurationError):
            stage.execute(context)

    def test_invalid_template_type_raises_error(self):
        """Test that a template value which is neither a string nor a class raises."""
        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)
        # Bypass validation to force an unsupported template value type.
        context.config.template = 123

        stage = TemplateLoadingStage()

        with pytest.raises(ConfigurationError):
            stage.execute(context)

    def test_template_path_without_dot_raises_error(self):
        """A template string with no module separator is rejected immediately."""
        config = PipelineConfig(
            source="test.pdf", template="NoDotHere", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        stage = TemplateLoadingStage()

        with pytest.raises(ConfigurationError, match="at least one dot"):
            stage.execute(context)

    def test_template_load_falls_back_to_cwd_sys_path(self):
        """When the module isn't importable as-is, retry with cwd inserted into sys.path."""
        import sys

        call_count = {"n": 0}
        real_import_module = importlib.import_module

        def fake_import_module(name: str, *args: object, **kwargs: object) -> object:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ModuleNotFoundError(name)
            return real_import_module(name, *args, **kwargs)

        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        cwd = str(Path.cwd())
        # sys.path may contain cwd multiple times (pytest rootdir insertion,
        # editable-install path entries, etc.); purge every occurrence so the
        # "insert" branch executes, and remember how many to restore after.
        original_count = sys.path.count(cwd)
        while cwd in sys.path:
            sys.path.remove(cwd)
        try:
            with patch(
                "docling_graph.pipeline.stages.importlib.import_module",
                side_effect=fake_import_module,
            ):
                stage = TemplateLoadingStage()
                result = stage.execute(context)
            # The retry-with-cwd-on-sys.path branch (insert + finally-remove)
            # executed and still returned the resolved template.
            assert result.template.__name__ == "BaseModel"
            assert call_count["n"] == 2
            assert cwd not in sys.path
        finally:
            for _ in range(original_count):
                sys.path.insert(0, cwd)

    def test_template_load_cwd_already_on_path(self):
        """When cwd is already on sys.path, the retry re-imports without mutating sys.path."""
        import sys

        call_count = {"n": 0}
        real_import_module = importlib.import_module

        def fake_import_module(name: str, *args: object, **kwargs: object) -> object:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ModuleNotFoundError(name)
            return real_import_module(name, *args, **kwargs)

        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        cwd = str(Path.cwd())
        was_present = cwd in sys.path
        if not was_present:
            sys.path.insert(0, cwd)
        try:
            with patch(
                "docling_graph.pipeline.stages.importlib.import_module",
                side_effect=fake_import_module,
            ):
                stage = TemplateLoadingStage()
                result = stage.execute(context)
            assert result.template.__name__ == "BaseModel"
        finally:
            if not was_present and cwd in sys.path:
                sys.path.remove(cwd)

    def test_template_not_a_basemodel_subclass_raises_error(self):
        """Resolving to a non-BaseModel object raises ConfigurationError."""
        config = PipelineConfig(
            source="test.pdf", template="os.path.join", backend="llm", inference="local"
        )
        context = PipelineContext(config=config)

        stage = TemplateLoadingStage()

        with pytest.raises(ConfigurationError, match="BaseModel subclass"):
            stage.execute(context)


class TestExtractionStage:
    """Test suite for ExtractionStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = ExtractionStage()
        assert stage.name() == "Extraction"

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_success(self, mock_init_client, mock_factory):
        """Test successful extraction."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        # Mock LLM client initialization
        mock_client = Mock()
        mock_init_client.return_value = mock_client

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = (
            [TestModel(name="Test", value=100)],
            Mock(),  # docling_document
        )
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)

        stage = ExtractionStage()
        result = stage.execute(context)

        assert result.extracted_models is not None
        assert len(result.extracted_models) == 1
        assert result.extracted_models[0].name == "Test"
        assert result.docling_document is not None
        assert result.extractor is mock_extractor

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_passes_structured_output_default_true(self, mock_init_client, mock_factory):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor
        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["structured_output"] is True
        assert kwargs["structured_sparse_check"] is True

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_passes_structured_output_false_when_disabled(
        self, mock_init_client, mock_factory
    ):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor
        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            structured_output=False,
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["structured_output"] is False

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_passes_structured_sparse_check_false_when_disabled(
        self, mock_init_client, mock_factory
    ):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor
        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            structured_sparse_check=False,
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["structured_sparse_check"] is False

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_passes_docling_serve_config(self, mock_init_client, mock_factory):
        """A configured docling-serve URL reaches the extractor factory."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor
        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            docling_serve_url="http://serve:5001",
            docling_serve_api_key="secret",
            docling_serve_timeout=120,
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["docling_serve_config"] == {
            "base_url": "http://serve:5001",
            "api_key": "secret",
            "timeout": 120,
        }

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_no_docling_serve_by_default(
        self, mock_init_client, mock_factory, monkeypatch
    ):
        """Without a serve URL, the factory gets no docling-serve config."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        monkeypatch.delenv("DOCLING_SERVE_URL", raising=False)
        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor
        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        assert mock_factory.call_args.kwargs["docling_serve_config"] is None

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_uses_custom_llm_client(self, mock_init_client, mock_factory):
        """When llm_client is set, the pipeline uses it and does not initialize provider/model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        custom_client = Mock()

        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="Test")], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            llm_client=custom_client,
        )
        context = PipelineContext(config=config, template=TestModel)

        stage = ExtractionStage()
        stage.execute(context)

        mock_init_client.assert_not_called()
        mock_factory.assert_called_once()
        # Extractor must be created with the custom client, not a provider-built one
        call_kwargs = mock_factory.call_args[1]
        assert call_kwargs.get("llm_client") is custom_client

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_no_models_raises_error(self, mock_init_client, mock_factory):
        """Test that no models extracted raises ExtractionError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        # Mock LLM client initialization
        mock_client = Mock()
        mock_init_client.return_value = mock_client

        # Mock extractor that returns empty list
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)

        stage = ExtractionStage()

        with pytest.raises(ExtractionError):
            stage.execute(context)

    def test_extraction_no_template_raises_error(self):
        """Test that a missing template raises ExtractionError before any extraction work."""
        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=None)

        stage = ExtractionStage()

        with pytest.raises(ExtractionError, match="Template is required"):
            stage.execute(context)

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_propagates_trace_data_to_extractor(self, mock_init_client, mock_factory):
        """When the extractor exposes a trace_data attribute, the stage wires it up."""
        from pydantic import BaseModel

        from docling_graph.pipeline.trace import EventTrace

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.trace_data = None
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        trace = EventTrace()
        context = PipelineContext(config=config, template=TestModel, trace_data=trace)

        ExtractionStage().execute(context)

        assert mock_extractor.trace_data is trace

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    def test_extraction_vlm_backend_creates_vlm_extractor(self, mock_factory):
        """The vlm backend path builds the extractor without an LLM client."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="vlm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["backend_name"] == "vlm"
        assert "llm_client" not in kwargs

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_debug_uses_output_manager_debug_dir(self, mock_init_client, mock_factory):
        """debug=True with an output_manager routes debug_dir through get_debug_dir()."""
        from pydantic import BaseModel

        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor

        def _run(tmp_path) -> None:
            config = PipelineConfig(
                source="test.pdf",
                template=TestModel,
                backend="llm",
                inference="local",
                debug=True,
                output_dir=str(tmp_path),
            )
            output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
            context = PipelineContext(
                config=config, template=TestModel, output_manager=output_manager
            )
            ExtractionStage().execute(context)
            kwargs = mock_factory.call_args.kwargs
            assert kwargs["dense_config"]["debug_dir"] == str(output_manager.get_debug_dir())

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _run(Path(tmp))

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_extraction_debug_falls_back_to_output_dir(self, mock_init_client, mock_factory):
        """debug=True with no output_manager falls back to config.output_dir / 'debug'."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract.return_value = ([TestModel(name="ok")], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            debug=True,
            output_dir="/tmp/some-output",
        )
        context = PipelineContext(config=config, template=TestModel)

        ExtractionStage().execute(context)

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["dense_config"]["debug_dir"] == str(Path("/tmp/some-output") / "debug")

    def test_get_model_config_missing_backend_raises_error(self):
        """No config for the backend/inference combination raises ConfigurationError."""
        from docling_graph.pipeline.stages import ExtractionStage

        with pytest.raises(ConfigurationError, match="No configuration found"):
            ExtractionStage._get_model_config({}, "llm", "local")

    def test_get_model_config_empty_model_raises_error(self):
        """A resolved empty model string raises ConfigurationError."""
        from docling_graph.pipeline.stages import ExtractionStage

        models_config = {"llm": {"local": {"provider": "ollama", "model": ""}}}

        with pytest.raises(ConfigurationError, match="Resolved model is empty"):
            ExtractionStage._get_model_config(models_config, "llm", "local")

    @patch("docling_graph.llm_clients.config.resolve_effective_model_config")
    @patch("docling_graph.pipeline.stages.get_client")
    def test_initialize_llm_client_builds_client_from_provider(
        self, mock_get_client, mock_resolve_config
    ):
        """_initialize_llm_client resolves the effective config and constructs the client."""
        from docling_graph.pipeline.stages import ExtractionStage

        mock_client_class = Mock()
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance
        mock_get_client.return_value = mock_client_class
        mock_resolve_config.return_value = {"model": "gpt-x", "provider": "openai"}

        result = ExtractionStage._initialize_llm_client("openai", "gpt-x", overrides=None)

        assert result is mock_instance
        mock_get_client.assert_called_once_with("openai")
        mock_client_class.assert_called_once_with(
            model_config={"model": "gpt-x", "provider": "openai"}
        )


class TestExtractionStageProvenanceCapture:
    """Test suite for ExtractionStage._capture_provenance (spec hook H7)."""

    @staticmethod
    def _build_context(
        tmp_path: Path,
        source: str = "test.pdf",
        provenance: str = "standard",
        **context_kwargs: object,
    ) -> tuple[PipelineContext, type]:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source=source,
            template=TestModel,
            backend="llm",
            inference="local",
            provenance=provenance,
        )
        context = PipelineContext(config=config, template=TestModel, **context_kwargs)
        return context, TestModel

    def test_provenance_off_skips_capture(self, tmp_path):
        """provenance='off' short-circuits before touching the extractor/ledger."""
        context, _ = self._build_context(tmp_path, provenance="off")
        context.extractor = Mock()

        ExtractionStage()._capture_provenance(context)

        assert context.provenance is None

    def test_provenance_no_ledger_on_backend_skips_capture(self, tmp_path):
        """When the backend has no last_provenance ledger, nothing is captured."""
        context, _ = self._build_context(tmp_path)
        context.extractor = Mock()
        context.extractor.backend = Mock(last_provenance=None)

        ExtractionStage()._capture_provenance(context)

        assert context.provenance is None

    def test_provenance_rejects_non_ledger_value(self, tmp_path):
        """A non-ProvenanceLedger value on backend.last_provenance is ignored (mock safety)."""
        context, _ = self._build_context(tmp_path)
        context.extractor = Mock()
        context.extractor.backend = Mock(last_provenance="not-a-ledger")

        ExtractionStage()._capture_provenance(context)

        assert context.provenance is None

    def test_provenance_captures_document_id_from_real_file(self, tmp_path):
        """When normalized_source points at a real file, document_id hashes file bytes."""
        from docling_graph.core.provenance import ProvenanceLedger, content_hash

        real_file = tmp_path / "doc.pdf"
        real_file.write_bytes(b"hello world")

        context, _template = self._build_context(tmp_path, source=str(real_file))
        context.normalized_source = real_file
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert context.provenance is ledger
        assert ledger.document is not None
        assert ledger.document.document_id == content_hash(b"hello world")

    def test_provenance_captures_document_id_from_config_source_file(self, tmp_path):
        """candidate resolution also checks config.source directly when it is a real file."""
        from docling_graph.core.provenance import ProvenanceLedger, content_hash

        real_file = tmp_path / "doc2.pdf"
        real_file.write_bytes(b"abc123")

        context, _template = self._build_context(tmp_path, source=str(real_file))
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.document_id == content_hash(b"abc123")

    def test_provenance_falls_back_to_hashing_source_string(self, tmp_path):
        """When no real file is found, document_id hashes the source string instead."""
        from docling_graph.core.provenance import ProvenanceLedger, content_hash

        context, _template = self._build_context(tmp_path, source="not-a-real-file.pdf")
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.document_id == content_hash(b"not-a-real-file.pdf")

    def test_provenance_document_id_falls_back_on_read_error(self, tmp_path):
        """A candidate file that raises on read_bytes() falls back to hashing the source string."""
        from docling_graph.core.provenance import ProvenanceLedger, content_hash

        real_file = tmp_path / "unreadable.pdf"
        real_file.write_bytes(b"content")

        context, _template = self._build_context(tmp_path, source=str(real_file))
        context.normalized_source = real_file
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        with patch.object(Path, "read_bytes", side_effect=OSError("permission denied")):
            ExtractionStage()._capture_provenance(context)

        assert ledger.document.document_id == content_hash(str(real_file).encode("utf-8"))

    def test_provenance_uses_input_metadata_source_and_type(self, tmp_path):
        """input_metadata's original_source/input_type override the raw config source."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path, source="ignored.pdf")
        context.input_metadata = {
            "original_source": "https://example.com/doc.pdf",
            "input_type": "url",
        }
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.source == "https://example.com/doc.pdf"
        assert ledger.document.input_type == "url"

    def test_provenance_reads_page_count_from_docling_document(self, tmp_path):
        """page_count is populated from docling_document.num_pages() when callable."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.docling_document = Mock()
        context.docling_document.num_pages.return_value = 7
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.page_count == 7

    def test_provenance_page_count_none_when_num_pages_missing(self, tmp_path):
        """No num_pages attribute at all leaves page_count as None."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.docling_document = object()  # no num_pages attribute
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.page_count is None

    def test_provenance_page_count_none_when_num_pages_raises(self, tmp_path):
        """An exception from num_pages() is swallowed and page_count stays None."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.docling_document = Mock()
        context.docling_document.num_pages.side_effect = RuntimeError("boom")
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.page_count is None

    def test_provenance_schema_hash_success_with_template(self, tmp_path):
        """template_name/template_schema_hash are populated when template is set."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, template = self._build_context(tmp_path)
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.template_name == template.__name__
        assert ledger.document.template_schema_hash != ""

    def test_provenance_schema_hash_empty_when_template_none(self, tmp_path):
        """No template on the context leaves template_name/hash empty."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.template = None
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.template_name == ""
        assert ledger.document.template_schema_hash == ""

    def test_provenance_schema_hash_falls_back_on_exception(self, tmp_path):
        """A broken model_json_schema() is swallowed, leaving schema_hash empty."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)

        broken_template = Mock()
        broken_template.__name__ = "Broken"
        broken_template.model_json_schema.side_effect = RuntimeError("schema boom")
        context.template = broken_template

        ExtractionStage()._capture_provenance(context)

        assert ledger.document.template_name == "Broken"
        assert ledger.document.template_schema_hash == ""

    def test_provenance_emits_trace_event_when_trace_data_present(self, tmp_path):
        """trace_data.emit('provenance_captured', ...) fires with ledger stats."""
        from docling_graph.core.provenance import ProvenanceLedger
        from docling_graph.pipeline.trace import EventTrace

        context, _template = self._build_context(tmp_path)
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)
        trace = EventTrace()
        context.trace_data = trace

        ExtractionStage()._capture_provenance(context)

        events = trace.find_events("provenance_captured")
        assert len(events) == 1
        assert events[0].payload["resolution"] == ledger.resolution

    def test_provenance_no_trace_event_when_trace_data_absent(self, tmp_path):
        """No trace_data means no emit call, but capture still succeeds."""
        from docling_graph.core.provenance import ProvenanceLedger

        context, _template = self._build_context(tmp_path)
        context.extractor = Mock()
        ledger = ProvenanceLedger()
        context.extractor.backend = Mock(last_provenance=ledger)
        context.trace_data = None

        ExtractionStage()._capture_provenance(context)

        assert context.provenance is ledger


class TestDoclingExportStage:
    """Test suite for DoclingExportStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = DoclingExportStage()
        assert stage.name() == "Docling Export"

    def test_skip_if_not_configured(self, tmp_path):
        """Test stage skips if export not configured."""
        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_docling=False,
            output_dir=str(tmp_path),
        )

        # Mock the docling_document's export_to_markdown to return a string
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test Document"
        mock_doc.export_to_dict.return_value = {"test": "data"}

        context = PipelineContext(config=config, docling_document=mock_doc, output_dir=tmp_path)

        stage = DoclingExportStage()
        result = stage.execute(context)

        # Should return context (may have exported even if export_docling=False)
        assert result.config == config

    def test_writes_chunks_json_when_provenance_present(self, tmp_path):
        """Chunk records captured in the provenance ledger get dumped to chunks.json."""
        from docling_graph.core.provenance.models import ChunkRecord, ProvenanceLedger
        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )

        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test Document"
        mock_doc.export_to_dict.return_value = {"test": "data"}

        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        ledger = ProvenanceLedger(
            chunks={
                1: ChunkRecord(chunk_id=1, batch_index=0, text="second chunk"),
                0: ChunkRecord(chunk_id=0, batch_index=0, text="first chunk"),
            }
        )
        context = PipelineContext(
            config=config,
            docling_document=mock_doc,
            output_manager=output_manager,
            provenance=ledger,
        )

        stage = DoclingExportStage()
        stage.execute(context)

        chunks_path = output_manager.get_docling_dir() / "chunks.json"
        assert chunks_path.exists()
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        assert [c["chunk_id"] for c in data] == [0, 1]
        assert data[0]["text"] == "first chunk"

    def test_skips_chunks_json_when_no_provenance(self, tmp_path):
        """No provenance ledger means no chunks.json (nothing to dump)."""
        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )

        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test Document"
        mock_doc.export_to_dict.return_value = {"test": "data"}

        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        context = PipelineContext(
            config=config,
            docling_document=mock_doc,
            output_manager=output_manager,
        )

        stage = DoclingExportStage()
        stage.execute(context)

        assert not (output_manager.get_docling_dir() / "chunks.json").exists()

    def test_skips_export_when_all_formats_disabled(self, tmp_path):
        """When export_docling/json/markdown are all disabled, the stage returns early."""
        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_docling=False,
            export_docling_json=False,
            export_markdown=False,
            output_dir=str(tmp_path),
        )
        mock_doc = Mock()
        context = PipelineContext(config=config, docling_document=mock_doc, output_dir=tmp_path)

        stage = DoclingExportStage()
        result = stage.execute(context)

        assert result.config == config
        mock_doc.export_to_markdown.assert_not_called()

    def test_warns_when_no_docling_document(self, tmp_path):
        """No docling_document on the context short-circuits with a warning."""
        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        context = PipelineContext(config=config, docling_document=None, output_dir=tmp_path)

        stage = DoclingExportStage()
        result = stage.execute(context)

        assert result.docling_document is None

    def test_warns_when_no_output_manager(self, tmp_path):
        """A docling_document with no output_manager also short-circuits."""
        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        mock_doc = Mock()
        context = PipelineContext(
            config=config, docling_document=mock_doc, output_manager=None, output_dir=tmp_path
        )

        stage = DoclingExportStage()
        result = stage.execute(context)

        mock_doc.export_to_markdown.assert_not_called()
        assert result.output_manager is None

    def test_emits_trace_event_after_export(self, tmp_path):
        """A successful export emits an 'export_written' trace event."""
        from docling_graph.core.utils.output_manager import OutputDirectoryManager
        from docling_graph.pipeline.trace import EventTrace

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test Document"
        mock_doc.export_to_dict.return_value = {"test": "data"}

        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            docling_document=mock_doc,
            output_manager=output_manager,
            trace_data=trace,
        )

        stage = DoclingExportStage()
        stage.execute(context)

        events = trace.find_events("export_written")
        assert any(e.stage == "docling_export" for e in events)

    def test_emits_trace_event_for_chunks_export(self, tmp_path):
        """Writing chunks.json also emits its own 'export_written' trace event."""
        from docling_graph.core.provenance.models import ChunkRecord, ProvenanceLedger
        from docling_graph.core.utils.output_manager import OutputDirectoryManager
        from docling_graph.pipeline.trace import EventTrace

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test Document"
        mock_doc.export_to_dict.return_value = {"test": "data"}

        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        ledger = ProvenanceLedger(
            chunks={0: ChunkRecord(chunk_id=0, batch_index=0, text="chunk text")}
        )
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            docling_document=mock_doc,
            output_manager=output_manager,
            provenance=ledger,
            trace_data=trace,
        )

        stage = DoclingExportStage()
        stage.execute(context)

        events = [e for e in trace.find_events("export_written") if "chunk_count" in e.payload]
        assert len(events) == 1
        assert events[0].payload["chunk_count"] == 1


class TestGraphConversionStage:
    """Test suite for GraphConversionStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = GraphConversionStage()
        assert stage.name() == "Graph Conversion"

    @patch("docling_graph.pipeline.stages.GraphConverter")
    def test_graph_conversion_success(self, mock_converter_class):
        """Test successful graph conversion."""
        import networkx as nx
        from pydantic import BaseModel

        from docling_graph.core.converters.models import GraphMetadata

        class TestModel(BaseModel):
            name: str

        # Mock converter
        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1", label="Test")

        # Mock metadata with required source_models field
        mock_metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)

        mock_converter = Mock()
        mock_converter.pydantic_list_to_graph.return_value = (mock_graph, mock_metadata)
        mock_converter_class.return_value = mock_converter

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(
            config=config, template=TestModel, extracted_models=[TestModel(name="Test")]
        )

        stage = GraphConversionStage()
        result = stage.execute(context)

        assert result.knowledge_graph is not None
        assert result.graph_metadata is not None
        assert result.node_registry is not None

    def test_no_extracted_models_raises_error(self):
        """Missing extracted_models raises PipelineError before touching the converter."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel, extracted_models=None)

        stage = GraphConversionStage()
        with pytest.raises(PipelineError, match="No extracted models available"):
            stage.execute(context)

    @patch("docling_graph.pipeline.stages.GraphConverter")
    def test_emits_graph_created_trace_event(self, mock_converter_class):
        """A successful conversion emits a 'graph_created' trace event with counts."""
        import networkx as nx
        from pydantic import BaseModel

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.pipeline.trace import EventTrace

        class TestModel(BaseModel):
            name: str

        mock_graph = nx.DiGraph()
        mock_metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        mock_converter = Mock()
        mock_converter.pydantic_list_to_graph.return_value = (mock_graph, mock_metadata)
        mock_converter_class.return_value = mock_converter

        config = PipelineConfig(
            source="test.pdf", template=TestModel, backend="llm", inference="local"
        )
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            template=TestModel,
            extracted_models=[TestModel(name="Test")],
            trace_data=trace,
        )

        GraphConversionStage().execute(context)

        events = trace.find_events("graph_created")
        assert len(events) == 1
        assert events[0].payload["node_count"] == 1

    @patch("docling_graph.core.provenance.binder.bind_provenance")
    @patch("docling_graph.pipeline.stages.GraphConverter")
    def test_emits_provenance_bound_trace_event_when_bind_stats_present(
        self, mock_converter_class, mock_bind_provenance
    ):
        """When a provenance binder produces bind_stats, a 'provenance_bound' event fires."""
        import networkx as nx
        from pydantic import BaseModel

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.core.provenance import ProvenanceLedger
        from docling_graph.pipeline.trace import EventTrace

        class TestModel(BaseModel):
            name: str

        mock_graph = nx.DiGraph()
        mock_metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        mock_converter = Mock()

        def fake_pydantic_list_to_graph(
            models: object, provenance_binder: object = None
        ) -> tuple[object, object]:
            if provenance_binder is not None:
                provenance_binder(mock_graph, models)
            return mock_graph, mock_metadata

        mock_converter.pydantic_list_to_graph.side_effect = fake_pydantic_list_to_graph
        mock_converter_class.return_value = mock_converter
        mock_bind_provenance.return_value = {"bound": 1}

        config = PipelineConfig(
            source="test.pdf",
            template=TestModel,
            backend="llm",
            inference="local",
            provenance="standard",
        )
        trace = EventTrace()
        ledger = ProvenanceLedger()
        context = PipelineContext(
            config=config,
            template=TestModel,
            extracted_models=[TestModel(name="Test")],
            trace_data=trace,
            provenance=ledger,
        )

        GraphConversionStage().execute(context)

        mock_bind_provenance.assert_called_once()
        events = trace.find_events("provenance_bound")
        assert len(events) == 1
        assert events[0].payload == {"bound": 1}


class TestExportStage:
    """Test suite for ExportStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = ExportStage()
        assert stage.name() == "Export"

    @patch("docling_graph.pipeline.stages.CSVExporter")
    @patch("docling_graph.pipeline.stages.JSONExporter")
    def test_export_success(self, mock_json_exporter, mock_csv_exporter, tmp_path):
        """Test successful export."""
        import networkx as nx

        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        # Mock exporters
        mock_csv_instance = Mock()
        mock_json_instance = Mock()
        mock_csv_exporter.return_value = mock_csv_instance
        mock_json_exporter.return_value = mock_json_instance

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_format="csv",
            output_dir=str(tmp_path),
        )

        # Create output manager
        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            output_dir=tmp_path,
            output_manager=output_manager,
        )

        stage = ExportStage()
        stage.execute(context)

        # Should have called exporters
        mock_csv_instance.export.assert_called_once()
        mock_json_instance.export.assert_called_once()

    def test_warns_when_no_output_manager(self):
        """No output_manager on the context short-circuits export with a warning."""
        import networkx as nx

        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        mock_graph = nx.DiGraph()
        context = PipelineContext(config=config, knowledge_graph=mock_graph, output_manager=None)

        stage = ExportStage()
        result = stage.execute(context)

        assert result.output_manager is None

    @patch("docling_graph.pipeline.stages.CypherExporter")
    @patch("docling_graph.pipeline.stages.JSONExporter")
    def test_export_cypher_format(self, mock_json_exporter, mock_cypher_exporter, tmp_path):
        """export_format='cypher' routes through CypherExporter instead of CSVExporter."""
        import networkx as nx

        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        mock_cypher_instance = Mock()
        mock_json_instance = Mock()
        mock_cypher_exporter.return_value = mock_cypher_instance
        mock_json_exporter.return_value = mock_json_instance

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_format="cypher",
            output_dir=str(tmp_path),
        )
        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            output_dir=tmp_path,
            output_manager=output_manager,
        )

        stage = ExportStage()
        stage.execute(context)

        mock_cypher_instance.export.assert_called_once()
        mock_json_instance.export.assert_called_once()

    @patch("docling_graph.pipeline.stages.CSVExporter")
    @patch("docling_graph.pipeline.stages.JSONExporter")
    def test_export_emits_trace_event(self, mock_json_exporter, mock_csv_exporter, tmp_path):
        """A successful export emits an 'export_written' trace event."""
        import networkx as nx

        from docling_graph.core.utils.output_manager import OutputDirectoryManager
        from docling_graph.pipeline.trace import EventTrace

        mock_csv_exporter.return_value = Mock()
        mock_json_exporter.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_format="csv",
            output_dir=str(tmp_path),
        )
        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            output_dir=tmp_path,
            output_manager=output_manager,
            trace_data=trace,
        )

        stage = ExportStage()
        stage.execute(context)

        events = trace.find_events("export_written")
        assert len(events) == 1
        assert events[0].payload["format"] == "csv"

    @patch("docling_graph.pipeline.stages.CSVExporter")
    @patch("docling_graph.pipeline.stages.JSONExporter")
    def test_export_writes_provenance_ledger_when_present(
        self, mock_json_exporter, mock_csv_exporter, tmp_path
    ):
        """A non-None provenance ledger (provenance != 'off') is persisted as provenance.json."""
        import networkx as nx

        from docling_graph.core.provenance import ProvenanceLedger
        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        mock_csv_exporter.return_value = Mock()
        mock_json_exporter.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            export_format="csv",
            output_dir=str(tmp_path),
            provenance="standard",
        )
        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            output_dir=tmp_path,
            output_manager=output_manager,
            provenance=ProvenanceLedger(),
        )

        stage = ExportStage()
        stage.execute(context)

        provenance_path = output_manager.get_docling_graph_dir() / "provenance.json"
        assert provenance_path.exists()


class TestVisualizationStage:
    """Test suite for VisualizationStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = VisualizationStage()
        assert stage.name() == "Visualization"

    @patch("docling_graph.pipeline.stages.InteractiveVisualizer")
    @patch("docling_graph.pipeline.stages.ReportGenerator")
    def test_visualization_success(self, mock_report_class, mock_viz_class, tmp_path):
        """Test successful visualization."""
        import networkx as nx

        # Mock visualizers
        mock_viz = Mock()
        mock_report = Mock()
        mock_viz_class.return_value = mock_viz
        mock_report_class.return_value = mock_report

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )

        from docling_graph.core.converters.models import GraphMetadata

        metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)

        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            graph_metadata=metadata,
            output_dir=tmp_path,
            extracted_models=[Mock()],
        )

        stage = VisualizationStage()
        stage.execute(context)

        # Should have called visualizers with correct methods
        mock_report.visualize.assert_called_once()
        mock_viz.save_cytoscape_graph.assert_called_once()

    @patch("docling_graph.pipeline.stages.InteractiveVisualizer")
    @patch("docling_graph.pipeline.stages.ReportGenerator")
    def test_uses_output_manager_dir_when_available(
        self, mock_report_class, mock_viz_class, tmp_path
    ):
        """When output_manager is set, its docling_graph dir wins over output_dir."""
        import networkx as nx

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.core.utils.output_manager import OutputDirectoryManager

        mock_viz_class.return_value = Mock()
        mock_report_class.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        output_manager = OutputDirectoryManager(tmp_path, "test.pdf")
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            graph_metadata=metadata,
            output_manager=output_manager,
            extracted_models=[Mock()],
        )

        VisualizationStage().execute(context)

        expected_dir = output_manager.get_docling_graph_dir()
        report_call = mock_report_class.return_value.visualize.call_args
        assert str(report_call.args[1]).startswith(str(expected_dir))

    def test_no_output_dir_raises_error(self):
        """Missing both output_manager and output_dir raises PipelineError."""
        config = PipelineConfig(
            source="test.pdf", template="pydantic.BaseModel", backend="llm", inference="local"
        )
        context = PipelineContext(
            config=config, output_manager=None, output_dir=None, extracted_models=[Mock()]
        )

        stage = VisualizationStage()
        with pytest.raises(PipelineError, match="Output directory is required"):
            stage.execute(context)

    def test_no_extracted_models_raises_error(self, tmp_path):
        """Missing extracted_models raises PipelineError."""
        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        context = PipelineContext(config=config, output_dir=tmp_path, extracted_models=None)

        stage = VisualizationStage()
        with pytest.raises(PipelineError, match="No extracted models available"):
            stage.execute(context)

    @patch("docling_graph.pipeline.stages.InteractiveVisualizer")
    @patch("docling_graph.pipeline.stages.ReportGenerator")
    def test_extracts_llm_diagnostics_from_trace_data(
        self, mock_report_class, mock_viz_class, tmp_path
    ):
        """extraction_completed trace events populate llm_diagnostics for the report."""
        import networkx as nx

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.pipeline.trace import EventTrace

        mock_report = Mock()
        mock_report_class.return_value = mock_report
        mock_viz_class.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        trace = EventTrace()
        trace.emit(
            "extraction_completed",
            "extraction",
            {
                "metadata": {
                    "structured_attempted": True,
                    "structured_failed": False,
                    "fallback_used": False,
                    "fallback_error_class": None,
                }
            },
        )
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            graph_metadata=metadata,
            output_dir=tmp_path,
            extracted_models=[Mock()],
            trace_data=trace,
        )

        VisualizationStage().execute(context)

        call_kwargs = mock_report.visualize.call_args.kwargs
        assert call_kwargs["llm_diagnostics"]["structured_attempted"] is True
        assert call_kwargs["llm_diagnostics"]["fallback_used"] is False

    @patch("docling_graph.pipeline.stages.InteractiveVisualizer")
    @patch("docling_graph.pipeline.stages.ReportGenerator")
    def test_no_llm_diagnostics_when_no_extraction_completed_event(
        self, mock_report_class, mock_viz_class, tmp_path
    ):
        """No matching trace events means llm_diagnostics stays empty."""
        import networkx as nx

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.pipeline.trace import EventTrace

        mock_report = Mock()
        mock_report_class.return_value = mock_report
        mock_viz_class.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            graph_metadata=metadata,
            output_dir=tmp_path,
            extracted_models=[Mock()],
            trace_data=trace,
        )

        VisualizationStage().execute(context)

        call_kwargs = mock_report.visualize.call_args.kwargs
        assert call_kwargs["llm_diagnostics"] == {}

    @patch("docling_graph.pipeline.stages.InteractiveVisualizer")
    @patch("docling_graph.pipeline.stages.ReportGenerator")
    def test_emits_trace_event_after_visualization(
        self, mock_report_class, mock_viz_class, tmp_path
    ):
        """A successful visualization pass emits an 'export_written' trace event."""
        import networkx as nx

        from docling_graph.core.converters.models import GraphMetadata
        from docling_graph.pipeline.trace import EventTrace

        mock_report_class.return_value = Mock()
        mock_viz_class.return_value = Mock()

        mock_graph = nx.DiGraph()
        mock_graph.add_node("node1")

        config = PipelineConfig(
            source="test.pdf",
            template="pydantic.BaseModel",
            backend="llm",
            inference="local",
            output_dir=str(tmp_path),
        )
        metadata = GraphMetadata(node_count=1, edge_count=0, source_models=1)
        trace = EventTrace()
        context = PipelineContext(
            config=config,
            knowledge_graph=mock_graph,
            graph_metadata=metadata,
            output_dir=tmp_path,
            extracted_models=[Mock()],
            trace_data=trace,
        )

        VisualizationStage().execute(context)

        events = trace.find_events("export_written")
        assert len(events) == 1
        assert events[0].stage == "visualization"


class TestStageInterface:
    """Test suite for stage interface compliance."""

    def test_all_stages_have_name_method(self):
        """Test that all stages implement name() method."""
        stages = [
            TemplateLoadingStage(),
            ExtractionStage(),
            DoclingExportStage(),
            GraphConversionStage(),
            ExportStage(),
            VisualizationStage(),
        ]

        for stage in stages:
            assert hasattr(stage, "name")
            assert callable(stage.name)
            name = stage.name()
            assert isinstance(name, str)
            assert len(name) > 0

    def test_all_stages_have_execute_method(self):
        """Test that all stages implement execute() method."""
        stages = [
            TemplateLoadingStage(),
            ExtractionStage(),
            DoclingExportStage(),
            GraphConversionStage(),
            ExportStage(),
            VisualizationStage(),
        ]

        for stage in stages:
            assert hasattr(stage, "execute")
            assert callable(stage.execute)


class TestExtractionStageDoclingDocumentInput:
    """DoclingDocument inputs must go through the strategy contract path."""

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_docling_document_uses_extract_from_document(self, mock_init_client, mock_factory):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract_from_document.return_value = ([TestModel(name="X")], Mock())
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        mock_document = Mock()
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = mock_document
        context.input_metadata = {"input_type": "docling_document"}

        stage = ExtractionStage()
        result = stage.execute(context)

        mock_extractor.extract_from_document.assert_called_once_with(mock_document, TestModel)
        mock_extractor.extract.assert_not_called()
        assert len(result.extracted_models) == 1

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    def test_docling_document_no_models_raises(self, mock_init_client, mock_factory):
        from pydantic import BaseModel

        from docling_graph.exceptions import ExtractionError

        class TestModel(BaseModel):
            name: str

        mock_init_client.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.extract_from_document.return_value = ([], None)
        mock_factory.return_value = mock_extractor

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = Mock()
        context.input_metadata = {"input_type": "docling_document"}

        stage = ExtractionStage()
        with pytest.raises(ExtractionError):
            stage.execute(context)

    def test_no_docling_document_raises_error(self):
        """Test that a missing docling_document raises ExtractionError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = None

        stage = ExtractionStage()
        with pytest.raises(ExtractionError, match="No DoclingDocument available"):
            stage._extract_from_docling_document(context)

    def test_extractor_without_extract_from_document_raises_error(self):
        """An extractor lacking extract_from_document() raises ExtractionError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = Mock()
        context.extractor = Mock(spec=[])  # no extract_from_document attribute

        stage = ExtractionStage()
        with pytest.raises(ExtractionError, match="does not support pre-converted"):
            stage._extract_from_docling_document(context)

    def test_extract_from_document_wraps_unexpected_exception(self):
        """A generic exception from extract_from_document is wrapped in ExtractionError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = Mock()
        context.extractor = Mock()
        context.extractor.extract_from_document.side_effect = RuntimeError("bad json")

        stage = ExtractionStage()
        with pytest.raises(ExtractionError, match="Failed to extract from DoclingDocument"):
            stage._extract_from_docling_document(context)

    def test_extract_from_document_reraises_extraction_error(self):
        """An ExtractionError from extract_from_document propagates unchanged."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = Mock()
        context.extractor = Mock()
        original_error = ExtractionError("original failure")
        context.extractor.extract_from_document.side_effect = original_error

        stage = ExtractionStage()
        with pytest.raises(ExtractionError) as exc_info:
            stage._extract_from_docling_document(context)
        assert exc_info.value is original_error

    def test_propagates_trace_data_to_existing_extractor(self):
        """When context.extractor is already set, trace_data is still wired onto it."""
        from pydantic import BaseModel

        from docling_graph.pipeline.trace import EventTrace

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        trace = EventTrace()
        context = PipelineContext(config=config, template=TestModel, trace_data=trace)
        context.docling_document = Mock()
        context.extractor = Mock()
        context.extractor.trace_data = None
        context.extractor.extract_from_document.return_value = ([TestModel(name="x")], Mock())

        ExtractionStage()._extract_from_docling_document(context)

        assert context.extractor.trace_data is trace

    def test_extract_from_document_creates_extractor_when_missing(self):
        """When context.extractor is unset, it is created via _create_extractor."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        config = PipelineConfig(
            source="doc.json", template=TestModel, backend="llm", inference="local"
        )
        context = PipelineContext(config=config, template=TestModel)
        context.docling_document = Mock()
        context.extractor = None

        created_extractor = Mock()
        created_extractor.extract_from_document.return_value = ([TestModel(name="x")], Mock())

        stage = ExtractionStage()
        with patch.object(
            stage, "_create_extractor", return_value=created_extractor
        ) as mock_create:
            result = stage._extract_from_docling_document(context)

        mock_create.assert_called_once_with(context)
        assert context.extractor is created_extractor
        assert len(result) == 1
