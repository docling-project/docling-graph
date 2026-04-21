from docling_graph.cli.constants import (
    API_PROVIDERS,
    BACKENDS,
    EXPORT_FORMATS,
    EXTRACTION_CONTRACTS,
    INFERENCE_LOCATIONS,
    LOCAL_PROVIDER_DEFAULTS,
    LOCAL_PROVIDERS,
    PROCESSING_MODES,
)
from docling_graph.config import PipelineConfig


class TestConstants:
    def test_export_formats_contains_valid_values(self):
        # json removed since EXPORT_FORMATS doesn't have it
        assert "csv" in EXPORT_FORMATS
        assert "cypher" in EXPORT_FORMATS

    def test_pipeline_config_remote_providers_match_api_providers(self):
        cfg = PipelineConfig()
        assert cfg.models.llm.remote.provider in API_PROVIDERS
        assert cfg.models.llm.local.provider in LOCAL_PROVIDERS

    def test_lmstudio_in_local_providers_and_defaults(self):
        """LM Studio is a supported local provider with a default model placeholder."""
        assert "lmstudio" in LOCAL_PROVIDERS
        assert "lmstudio" in LOCAL_PROVIDER_DEFAULTS
        assert LOCAL_PROVIDER_DEFAULTS["lmstudio"] == "local-model"

    def test_processing_modes_contains_valid_values(self):
        assert "one-to-one" in PROCESSING_MODES
        assert "many-to-one" in PROCESSING_MODES

    def test_extraction_contracts_contains_valid_values(self):
        assert "direct" in EXTRACTION_CONTRACTS
        assert "dense" in EXTRACTION_CONTRACTS

    def test_backend_types_contains_valid_values(self):
        assert "llm" in BACKENDS
        assert "vlm" in BACKENDS

    def test_inference_locations_contains_valid_values(self):
        assert "local" in INFERENCE_LOCATIONS
        assert "remote" in INFERENCE_LOCATIONS
