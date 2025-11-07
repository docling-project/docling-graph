"""
Tests for CLI validators.
"""

from unittest.mock import MagicMock, patch

import pytest
import typer

from docling_graph.cli.validators import (
    check_provider_installed,
    get_provider_from_config,
    print_dependency_setup_guide,
    print_next_steps_with_deps,
    validate_and_warn_dependencies,
    validate_backend_type,
    validate_config_dependencies,
    validate_docling_config,
    validate_export_format,
    validate_inference,
    validate_option,
    validate_processing_mode,
    validate_provider,
    validate_vlm_constraints,
)


class TestValidateOption:
    """Test generic validate_option function."""

    def test_validate_option_valid_value(self):
        """Should return valid option."""
        result = validate_option("test", {"test", "other"}, "test option")
        assert result == "test"

    def test_validate_option_case_insensitive(self):
        """Should handle case insensitively."""
        result = validate_option("TEST", {"test", "other"}, "test option")
        assert result == "test"

    @patch("rich.print")
    def test_validate_option_invalid_raises_exit(self, mock_print):
        """Should raise Exit for invalid option."""
        with pytest.raises(typer.Exit):
            validate_option("invalid", {"valid1", "valid2"}, "test option")


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
        """Should raise Exit for invalid mode."""
        with pytest.raises(typer.Exit):
            validate_processing_mode("invalid-mode")


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
        """Should raise Exit for invalid backend."""
        with pytest.raises(typer.Exit):
            validate_backend_type("invalid")


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
        result = validate_inference("LOCAL")
        assert result == "local"

    def test_validate_inference_invalid_raises_exit(self):
        """Should raise Exit for invalid inference location."""
        with pytest.raises(typer.Exit):
            validate_inference("cloud")


class TestValidateDoclingConfig:
    """Test docling config validation."""

    def test_validate_docling_config_ocr(self):
        """Should accept 'ocr' config."""
        result = validate_docling_config("ocr")
        assert result == "ocr"

    def test_validate_docling_config_vision(self):
        """Should accept 'vision' config."""
        result = validate_docling_config("vision")
        assert result == "vision"

    def test_validate_docling_config_case_insensitive(self):
        """Should handle case-insensitive input."""
        result = validate_docling_config("OCR")
        assert result == "ocr"

    def test_validate_docling_config_invalid_raises_exit(self):
        """Should raise Exit for invalid config."""
        with pytest.raises(typer.Exit):
            validate_docling_config("invalid")


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
        """Should raise Exit for invalid format."""
        with pytest.raises(typer.Exit):
            validate_export_format("json")


class TestValidateVLMConstraints:
    """Test VLM constraint validation."""

    def test_vlm_local_inference_valid(self):
        """VLM with local inference should be valid."""
        # Should not raise
        validate_vlm_constraints("vlm", "local")

    @patch("rich.print")
    def test_vlm_remote_inference_invalid(self, mock_print):
        """VLM with remote inference should raise Exit."""
        with pytest.raises(typer.Exit):
            validate_vlm_constraints("vlm", "remote")

    def test_llm_remote_inference_valid(self):
        """LLM with remote inference should be valid."""
        # Should not raise
        validate_vlm_constraints("llm", "remote")

    def test_llm_local_inference_valid(self):
        """LLM with local inference should be valid."""
        # Should not raise
        validate_vlm_constraints("llm", "local")


class TestValidateProvider:
    """Test provider validation."""

    def test_validate_provider_local_valid(self):
        """Should accept valid local provider."""
        result = validate_provider("ollama", "local")
        assert result == "ollama"

    def test_validate_provider_remote_valid(self):
        """Should accept valid remote provider."""
        result = validate_provider("openai", "remote")
        assert result == "openai"

    def test_validate_provider_invalid_raises_error(self):
        """Should raise ValueError for invalid provider."""
        with pytest.raises(ValueError):
            validate_provider("invalid", "local")


class TestCheckProviderInstalled:
    """Test provider installation check."""

    def test_check_provider_installed_returns_bool(self):
        """Should return boolean value."""
        result = check_provider_installed("ollama")
        assert isinstance(result, bool)

    def test_check_unknown_provider_returns_true(self):
        """Should return True for unknown providers (safe default)."""
        result = check_provider_installed("nonexistent_provider_xyz")
        assert result is True


class TestGetProviderFromConfig:
    """Test get_provider_from_config function."""

    def test_get_provider_from_config_llm_local(self):
        """Should extract local LLM provider and inference type."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        provider, inference_type = get_provider_from_config(config)
        assert provider == "ollama"
        assert inference_type == "local"

    def test_get_provider_from_config_llm_remote(self):
        """Should extract remote LLM provider and inference type."""
        config = {
            "defaults": {"backend": "llm", "inference": "remote"},
            "models": {"llm": {"remote": {"provider": "openai"}}},
        }
        provider, inference_type = get_provider_from_config(config)
        assert provider == "openai"
        assert inference_type == "remote"

    def test_get_provider_from_config_returns_tuple(self):
        """Should return tuple of (provider, inference_type)."""
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        result = get_provider_from_config(config)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestValidateConfigDependencies:
    """Test config dependency validation."""

    @patch("docling_graph.cli.validators.check_provider_installed")
    def test_validate_config_dependencies_returns_tuple(self, mock_check):
        """Should return tuple of (bool, inference_type)."""
        mock_check.return_value = True
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        result = validate_config_dependencies(config)

        assert isinstance(result, tuple)
        assert len(result) == 2
        is_valid, inference_type = result
        assert isinstance(is_valid, bool)
        assert isinstance(inference_type, str)

    @patch("docling_graph.cli.validators.check_provider_installed")
    def test_validate_config_dependencies_installed(self, mock_check):
        """Should return (True, inference_type) when installed."""
        mock_check.return_value = True
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        is_valid, inference_type = validate_config_dependencies(config)

        assert is_valid is True
        assert inference_type == "local"

    @patch("docling_graph.cli.validators.check_provider_installed")
    def test_validate_config_dependencies_missing(self, mock_check):
        """Should return (False, inference_type) when missing."""
        mock_check.return_value = False
        config = {
            "defaults": {"backend": "llm", "inference": "remote"},
            "models": {"llm": {"remote": {"provider": "openai"}}},
        }

        is_valid, inference_type = validate_config_dependencies(config)

        assert is_valid is False
        assert inference_type == "remote"


class TestValidateAndWarnDependencies:
    """Test validate_and_warn_dependencies function."""

    @patch("docling_graph.cli.validators.check_provider_installed")
    @patch("rich.print")
    def test_validate_and_warn_dependencies_installed(self, mock_print, mock_check):
        """Should return True when dependencies are installed."""
        mock_check.return_value = True
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        result = validate_and_warn_dependencies(config)

        assert result is True

    @patch("docling_graph.cli.validators.print_dependency_setup_guide")
    @patch("docling_graph.cli.validators.check_provider_installed")
    def test_validate_and_warn_dependencies_missing(self, mock_check, mock_guide):
        """Should return False and show guide when dependencies missing."""
        mock_check.return_value = False
        config = {
            "defaults": {"backend": "llm", "inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }

        result = validate_and_warn_dependencies(config)

        assert result is False
        mock_guide.assert_called_once()


class TestPrintDependencySetupGuide:
    """Test print_dependency_setup_guide function."""

    @patch("docling_graph.cli.validators.rich_print")
    def test_print_dependency_setup_guide_takes_inference_type(self, mock_rich_print):
        """Should accept inference_type as first parameter."""
        print_dependency_setup_guide("local", "ollama")
        assert mock_rich_print.called

    @patch("docling_graph.cli.validators.rich_print")
    def test_print_dependency_setup_guide_without_provider(self, mock_rich_print):
        """Should work without specific provider."""
        print_dependency_setup_guide("remote")
        assert mock_rich_print.called


class TestPrintNextStepsWithDeps:
    """Test print_next_steps_with_deps function."""

    @patch("docling_graph.cli.validators.rich_print")
    def test_print_next_steps_with_deps_requires_existing_steps(self, mock_rich_print):
        """Should require existing_steps parameter."""
        config = {
            "defaults": {"inference": "local"},
            "models": {"llm": {"local": {"provider": "ollama"}}},
        }
        existing_steps = "Next steps:\n1. Do something"

        print_next_steps_with_deps(config, existing_steps)

        assert mock_rich_print.called
