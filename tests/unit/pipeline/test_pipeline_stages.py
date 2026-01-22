"""Tests for pipeline stages."""

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


class TestExtractionStage:
    """Test suite for ExtractionStage."""

    def test_stage_name(self):
        """Test stage name."""
        stage = ExtractionStage()
        assert stage.name() == "Extraction"

    @patch("docling_graph.pipeline.stages.ExtractorFactory.create_extractor")
    @patch("docling_graph.pipeline.stages.ExtractionStage._initialize_llm_client")
    @patch("docling_graph.llm_clients.config.get_model_config")
    def test_extraction_success(self, mock_get_model_config, mock_init_client, mock_factory):
        """Test successful extraction."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        # Mock model config to avoid vLLM connection
        mock_config = MagicMock()
        mock_config.context_limit = 4096
        mock_get_model_config.return_value = mock_config

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
    @patch("docling_graph.llm_clients.config.get_model_config")
    def test_extraction_no_models_raises_error(
        self, mock_get_model_config, mock_init_client, mock_factory
    ):
        """Test that no models extracted raises ExtractionError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        # Mock model config to avoid vLLM connection
        mock_config = MagicMock()
        mock_config.context_limit = 4096
        mock_get_model_config.return_value = mock_config

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
        context = PipelineContext(config=config, knowledge_graph=mock_graph, output_dir=tmp_path)

        stage = ExportStage()
        stage.execute(context)

        # Should have called exporters
        mock_csv_instance.export.assert_called_once()
        mock_json_instance.export.assert_called_once()


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
