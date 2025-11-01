"""
Unit tests for CLI validators.
"""

import pytest
import typer

from docling_graph.cli.validators import (
    validate_backend_type,
    validate_docling_config,
    validate_export_format,
    validate_inference,
    validate_processing_mode,
    validate_vlm_constraints,
)


class TestValidateProcessingMode:
    """Tests for validate_processing_mode function."""

    def test_valid_one_to_one(self):
        """Test validation of 'one-to-one' mode."""
        assert validate_processing_mode("one-to-one") == "one-to-one"
        assert validate_processing_mode("ONE-TO-ONE") == "one-to-one"
        assert validate_processing_mode("One-To-One") == "one-to-one"

    def test_valid_many_to_one(self):
        """Test validation of 'many-to-one' mode."""
        assert validate_processing_mode("many-to-one") == "many-to-one"
        assert validate_processing_mode("MANY-TO-ONE") == "many-to-one"

    def test_invalid_mode(self):
        """Test validation fails for invalid modes."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_processing_mode("invalid-mode")
        assert exc_info.value.exit_code == 1

    def test_empty_mode(self):
        """Test validation fails for empty mode."""
        with pytest.raises(typer.Exit):
            validate_processing_mode("")

    @pytest.mark.parametrize("invalid_input", ["one-two-one", "many", "one", "all", "batch"])
    def test_various_invalid_modes(self, invalid_input):
        """Test various invalid mode inputs."""
        with pytest.raises(typer.Exit):
            validate_processing_mode(invalid_input)


class TestValidateBackendType:
    """Tests for validate_backend_type function."""

    def test_valid_llm(self):
        """Test validation of 'llm' backend."""
        assert validate_backend_type("llm") == "llm"
        assert validate_backend_type("LLM") == "llm"
        assert validate_backend_type("Llm") == "llm"

    def test_valid_vlm(self):
        """Test validation of 'vlm' backend."""
        assert validate_backend_type("vlm") == "vlm"
        assert validate_backend_type("VLM") == "vlm"

    def test_invalid_backend(self):
        """Test validation fails for invalid backends."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_backend_type("gpt")
        assert exc_info.value.exit_code == 1

    @pytest.mark.parametrize("invalid_input", ["transformer", "bert", "gpt", "api", ""])
    def test_various_invalid_backends(self, invalid_input):
        """Test various invalid backend inputs."""
        with pytest.raises(typer.Exit):
            validate_backend_type(invalid_input)


class TestValidateInference:
    """Tests for validate_inference function."""

    def test_valid_local(self):
        """Test validation of 'local' inference."""
        assert validate_inference("local") == "local"
        assert validate_inference("LOCAL") == "local"

    def test_valid_remote(self):
        """Test validation of 'remote' inference."""
        assert validate_inference("remote") == "remote"
        assert validate_inference("REMOTE") == "remote"

    def test_invalid_inference(self):
        """Test validation fails for invalid inference."""
        with pytest.raises(typer.Exit):
            validate_inference("cloud")

    @pytest.mark.parametrize("invalid_input", ["api", "cloud", "hybrid", "distributed", ""])
    def test_various_invalid_inference(self, invalid_input):
        """Test various invalid inference inputs."""
        with pytest.raises(typer.Exit):
            validate_inference(invalid_input)


class TestValidateDoclingConfig:
    """Tests for validate_docling_config function."""

    def test_valid_ocr(self):
        """Test validation of 'ocr' config."""
        assert validate_docling_config("ocr") == "ocr"
        assert validate_docling_config("OCR") == "ocr"

    def test_valid_vision(self):
        """Test validation of 'vision' config."""
        assert validate_docling_config("vision") == "vision"
        assert validate_docling_config("VISION") == "vision"

    def test_invalid_config(self):
        """Test validation fails for invalid config."""
        with pytest.raises(typer.Exit):
            validate_docling_config("default")

    @pytest.mark.parametrize("invalid_input", ["default", "tesseract", "paddle", "easyocr", ""])
    def test_various_invalid_configs(self, invalid_input):
        """Test various invalid config inputs."""
        with pytest.raises(typer.Exit):
            validate_docling_config(invalid_input)


class TestValidateExportFormat:
    """Tests for validate_export_format function."""

    def test_valid_csv(self):
        """Test validation of 'csv' format."""
        assert validate_export_format("csv") == "csv"
        assert validate_export_format("CSV") == "csv"

    def test_valid_cypher(self):
        """Test validation of 'cypher' format."""
        assert validate_export_format("cypher") == "cypher"
        assert validate_export_format("CYPHER") == "cypher"

    def test_invalid_format(self):
        """Test validation fails for invalid format."""
        with pytest.raises(typer.Exit):
            validate_export_format("xml")


class TestValidateVlmConstraints:
    """Tests for validate_vlm_constraints function."""

    def test_valid_vlm_local(self):
        """Test VLM with local inference is valid."""
        # Should not raise any exception
        validate_vlm_constraints("vlm", "local")

    def test_valid_llm_remote(self):
        """Test LLM with remote inference is valid."""
        # Should not raise any exception
        validate_vlm_constraints("llm", "remote")

    def test_valid_llm_local(self):
        """Test LLM with local inference is valid."""
        # Should not raise any exception
        validate_vlm_constraints("llm", "local")

    def test_invalid_vlm_remote(self):
        """Test VLM with remote inference is invalid."""
        with pytest.raises(typer.Exit) as exc_info:
            validate_vlm_constraints("vlm", "remote")
        assert exc_info.value.exit_code == 1

    @pytest.mark.parametrize(
        "backend,inference,should_pass",
        [
            ("vlm", "local", True),
            ("vlm", "remote", False),
            ("llm", "local", True),
            ("llm", "remote", True),
        ],
    )
    def test_various_constraint_combinations(self, backend, inference, should_pass):
        """Test various backend/inference combinations."""
        if should_pass:
            validate_vlm_constraints(backend, inference)
        else:
            with pytest.raises(typer.Exit):
                validate_vlm_constraints(backend, inference)


class TestValidatorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_none_inputs(self):
        """Test validators with None input."""
        # These should raise AttributeError or TypeError
        with pytest.raises((AttributeError, TypeError)):
            validate_processing_mode(None)

    def test_numeric_inputs(self):
        """Test validators with numeric input."""
        with pytest.raises((AttributeError, TypeError)):
            validate_backend_type(123)

    def test_whitespace_inputs(self):
        """Test validators with whitespace."""
        with pytest.raises(typer.Exit):
            validate_processing_mode("  ")

        with pytest.raises(typer.Exit):
            validate_backend_type("\t")

    def test_special_characters(self):
        """Test validators with special characters."""
        with pytest.raises(typer.Exit):
            validate_inference("local@remote")

        with pytest.raises(typer.Exit):
            validate_export_format("csv;cypher")
