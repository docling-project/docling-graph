import pytest

from docling_graph.cli.constants import (
    API_PROVIDERS,
    BACKENDS,
    EXPORT_FORMATS,
    EXTRACTION_CONTRACTS,
    INFERENCE_LOCATIONS,
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

    def test_processing_modes_contains_valid_values(self):
        assert "one-to-one" in PROCESSING_MODES
        assert "many-to-one" in PROCESSING_MODES

    def test_extraction_contracts_contains_valid_values(self):
        assert "direct" in EXTRACTION_CONTRACTS
        assert "staged" in EXTRACTION_CONTRACTS
        assert "delta" in EXTRACTION_CONTRACTS

    def test_backend_types_contains_valid_values(self):
        assert "llm" in BACKENDS
        assert "vlm" in BACKENDS

    def test_inference_locations_contains_valid_values(self):
        assert "local" in INFERENCE_LOCATIONS
        assert "remote" in INFERENCE_LOCATIONS

    def test_staged_config_knobs_in_to_dict(self):
        """PipelineConfig.to_dict() includes staged_id_shard_size from preset."""
        cfg = PipelineConfig(staged_tuning_preset="advanced")
        d = cfg.to_dict()
        assert "staged_id_shard_size" in d
        assert d["staged_id_shard_size"] == 0  # advanced preset default: no sharding
        cfg_std = PipelineConfig(staged_tuning_preset="standard")
        d_std = cfg_std.to_dict()
        assert d_std["staged_id_shard_size"] == 0  # standard preset default: no sharding
