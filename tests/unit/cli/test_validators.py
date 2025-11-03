"""
Tests for CLI validators.
"""

from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.validators import (
    check_provider_installed,
    validate_and_warn_dependencies,
    validate_backend_type,
    validate_config_dependencies,
    validate_docling_config,
    validate_export_format,
    validate_inference,
    validate_processing_mode,
    validate_vlm_constraints,
)


class TestValidateProcessingMode:
    """Test processing mode validation."""

    def test_validate_processing_mode_one_to_one(self):
        """Should accept 'one-to-one' mode."""
        result = validate_processing_mode("one-to-one")
        assert result == "one-to-one"

    def test_validate_processing_mode_many_to_one(self):
        """Should accept 'many-to-one' mode."""
        result = validate_processing_mode("many-to-one")
        assert result == "many-to-one"

    def test_validate_processing_mode_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_processing_mode("ONE-TO-ONE")
        assert result == "one-to-one"

    def test_validate_processing_mode_invalid_raises_exit(self):
        """Should exit on invalid processing mode."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_processing_mode("invalid-mode")
        assert exc_info.value.exit_code == 1


class TestValidateBackendType:
    """Test backend type validation."""

    def test_validate_backend_type_llm(self):
        """Should accept 'llm' backend."""
        result = validate_backend_type("llm")
        assert result == "llm"

    def test_validate_backend_type_vlm(self):
        """Should accept 'vlm' backend."""
        result = validate_backend_type("vlm")
        assert result == "vlm"

    def test_validate_backend_type_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_backend_type("LLM")
        assert result == "llm"

    def test_validate_backend_type_invalid_raises_exit(self):
        """Should exit on invalid backend type."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_backend_type("invalid-backend")
        assert exc_info.value.exit_code == 1


class TestValidateInference:
    """Test inference location validation."""

    def test_validate_inference_local(self):
        """Should accept 'local' inference."""
        result = validate_inference("local")
        assert result == "local"

    def test_validate_inference_remote(self):
        """Should accept 'remote' inference."""
        result = validate_inference("remote")
        assert result == "remote"

    def test_validate_inference_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_inference("REMOTE")
        assert result == "remote"

    def test_validate_inference_invalid_raises_exit(self):
        """Should exit on invalid inference location."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_inference("cloud")
        assert exc_info.value.exit_code == 1


class TestValidateDoclingConfig:
    """Test Docling configuration validation."""

    def test_validate_docling_config_ocr(self):
        """Should accept 'ocr' pipeline."""
        result = validate_docling_config("ocr")
        assert result == "ocr"

    def test_validate_docling_config_vision(self):
        """Should accept 'vision' pipeline."""
        result = validate_docling_config("vision")
        assert result == "vision"

    def test_validate_docling_config_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_docling_config("OCR")
        assert result == "ocr"

    def test_validate_docling_config_invalid_raises_exit(self):
        """Should exit on invalid pipeline."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_docling_config("invalid-pipeline")
        assert exc_info.value.exit_code == 1


class TestValidateExportFormat:
    """Test export format validation."""

    def test_validate_export_format_csv(self):
        """Should accept 'csv' format."""
        result = validate_export_format("csv")
        assert result == "csv"

    def test_validate_export_format_cypher(self):
        """Should accept 'cypher' format."""
        result = validate_export_format("cypher")
        assert result == "cypher"

    def test_validate_export_format_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_export_format("CSV")
        assert result == "csv"

    def test_validate_export_format_invalid_raises_exit(self):
        """Should exit on invalid export format."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_export_format("invalid-format")
        assert exc_info.value.exit_code == 1


class TestValidateVLMConstraints:
    """Test VLM constraint validation."""

    def test_vlm_local_inference_valid(self):
        """Should allow VLM with local inference."""
        # Should not raise
        validate_vlm_constraints("vlm", "local")

    def test_vlm_remote_inference_invalid(self):
        """Should reject VLM with remote inference."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_vlm_constraints("vlm", "remote")
        assert exc_info.value.exit_code == 1

    def test_llm_remote_inference_valid(self):
        """Should allow LLM with remote inference."""
        validate_vlm_constraints("llm", "remote")

    def test_llm_local_inference_valid(self):
        """Should allow LLM with local inference."""
        validate_vlm_constraints("llm", "local")


class TestCheckProviderInstalled:
    """Test provider installation checking."""

    def test_check_provider_installed_returns_bool(self):
        """Should return boolean for provider check."""
        result = check_provider_installed("ollama")
        assert isinstance(result, bool)

    def test_check_unknown_provider_returns_true(self):
        """Should return True for unknown provider."""
        result = check_provider_installed("unknown-provider")
        assert result is True


class TestValidateConfigDependencies:
    """Test configuration dependency validation."""

    def test_validate_config_dependencies_local_llm(self):
        """Should validate local LLM configuration."""
        config = {
            "defaults": {"inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama", "default_model": "llama"}}},
        }
        is_valid, inference_type = validate_config_dependencies(config)
        assert isinstance(is_valid, bool)
        assert inference_type == "local"

    def test_validate_config_dependencies_remote(self):
        """Should validate remote configuration."""
        config = {
            "defaults": {"inference": "remote"},
            "models": {"llm": {"remote": {"provider": "mistral", "default_model": "test"}}},
        }
        is_valid, inference_type = validate_config_dependencies(config)
        assert isinstance(is_valid, bool)
        assert inference_type == "remote"

    def test_validate_config_dependencies_missing_config(self):
        """Should handle missing config sections."""
        config = {}
        is_valid, inference_type = validate_config_dependencies(config)
        assert inference_type == "remote"  # Default value


class TestValidateAndWarnDependencies:
    """Test dependency validation and warnings."""

    def test_validate_and_warn_dependencies_returns_bool(self):
        """Should return boolean result."""
        config = {
            "defaults": {"inference": "remote"},
            "models": {"llm": {"remote": {"provider": "mistral"}}},
        }
        result = validate_and_warn_dependencies(config, interactive=False)
        assert isinstance(result, bool)
