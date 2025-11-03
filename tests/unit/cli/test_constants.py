import pytest

from docling_graph.cli.constants import (
    API_PROVIDERS,
    BACKENDS,
    EXPORT_FORMATS,
    INFERENCE_LOCATIONS,
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
        models = getattr(cfg, "models", {})
        if hasattr(models, "model_dump"):
            models = models.model_dump()
        providers = sorted(API_PROVIDERS)

        remote_providers = set()  # fix, initialize as set
        for val in models.values():
            if "remote" in val:
                remote_providers.update(val["remote"].get("providers", []))
        assert set(remote_providers).issubset(set(providers))

    def test_processing_modes_contains_valid_values(self):
        assert "one-to-one" in PROCESSING_MODES
        assert "many-to-one" in PROCESSING_MODES

    def test_backend_types_contains_valid_values(self):
        assert "llm" in BACKENDS
        assert "vlm" in BACKENDS

    def test_inference_locations_contains_valid_values(self):
        assert "local" in INFERENCE_LOCATIONS
        assert "remote" in INFERENCE_LOCATIONS
