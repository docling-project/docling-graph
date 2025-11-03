from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docling_graph.config import PipelineConfig
from docling_graph.pipeline import run_pipeline


@pytest.mark.integration
class TestPipelineEndToEnd:
    @pytest.fixture
    def temp_output_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_pipeline_handles_missing_template(self, temp_output_dir):
        config = PipelineConfig(
            source="nonexistent.pdf",
            template="nonexistent.module.Template",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            reverse_edges=False,
            output_dir=str(temp_output_dir),
            export_format="csv",
            export_docling=False,
            export_markdown=False,
        )
        with pytest.raises(ModuleNotFoundError):
            run_pipeline(config)

    def test_pipeline_with_mock_extractor(self, temp_output_dir):
        config = PipelineConfig(
            source="sample.pdf",
            template="tests.fixtures.test_template.SamplePydanticModel",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_output_dir),
            export_format="csv",
            export_docling=False,
        )
        with patch("docling_graph.pipeline._load_template_class") as mock_load:
            mock_load.return_value = MagicMock()
            with patch("docling_graph.pipeline._get_model_config") as mock_get_model:
                mock_get_model.return_value = {"provider": "ollama", "model": "llama3.1:8b"}
                with patch("docling_graph.pipeline._initialize_llm_client") as mock_init_llm:
                    mock_init_llm.return_value = MagicMock()
                    result = run_pipeline(config)
                    assert result is None

    def test_pipeline_error_handling_missing_source(self, temp_output_dir):
        config = PipelineConfig(
            source="missing.pdf",
            template="docling_graph.templates.standard.TemplateModel",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir=str(temp_output_dir),
            export_format="csv",
        )
        with pytest.raises(ModuleNotFoundError):
            run_pipeline(config)


@pytest.mark.integration
class TestPipelineResourceCleanup:
    def test_pipeline_cleanup_called_on_error(self):
        with patch("docling_graph.pipeline._get_model_config") as mock_get_model:
            mock_get_model.return_value = {"provider": "ollama", "model": "llama3.1:8b"}
            with patch("docling_graph.pipeline._initialize_llm_client") as mock_init_llm:
                mock_init_llm.return_value = MagicMock()
                # Add more patches and logic as needed for cleanup test


@pytest.mark.integration
class TestPipelineConfigValidation:
    def test_config_with_minimal_required_fields(self):
        config = PipelineConfig(
            source="some.pdf",
            template="tests.fixtures.test_template.SamplePydanticModel",
            processing_mode="one-to-one",
            backend="llm",
            inference="local",
            docling_config="ocr",
            output_dir="outputs",
            export_format="csv",
        )
        assert config.backend == "llm"
