"""Tests for unified exception hierarchy."""

from typing import NoReturn

import pytest

from docling_graph.exceptions import (
    ClientError,
    ConfigurationError,
    DoclingGraphError,
    ExtractionError,
    GraphError,
    PipelineError,
    ValidationError,
)


class TestDoclingGraphError:
    """Test suite for base DoclingGraphError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = DoclingGraphError("Test error")
        assert error.message == "Test error"
        assert error.details == {}
        assert error.cause is None

    def test_error_with_details(self):
        """Test error with details dict."""
        error = DoclingGraphError(
            "Test error",
            details={"key": "value", "number": 42}
        )
        assert error.message == "Test error"
        assert error.details["key"] == "value"
        assert error.details["number"] == 42

    def test_error_with_cause(self):
        """Test error with cause exception."""
        original = ValueError("Original error")
        error = DoclingGraphError(
            "Wrapped error",
            cause=original
        )
        assert error.message == "Wrapped error"
        assert error.cause is original
        assert isinstance(error.cause, ValueError)

    def test_error_string_representation(self):
        """Test error string representation."""
        error = DoclingGraphError(
            "Test error",
            details={"field": "value"}
        )
        error_str = str(error)
        assert "Test error" in error_str
        assert "field" in error_str

    def test_error_repr(self):
        """Test error repr."""
        error = DoclingGraphError("Test error")
        repr_str = repr(error)
        assert "DoclingGraphError" in repr_str
        assert "Test error" in repr_str


class TestConfigurationError:
    """Test suite for ConfigurationError."""

    def test_is_docling_graph_error(self):
        """Test that ConfigurationError extends DoclingGraphError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_with_details(self):
        """Test configuration error with details."""
        error = ConfigurationError(
            "Invalid configuration",
            details={"field": "api_key", "value": None}
        )
        assert error.message == "Invalid configuration"
        assert error.details["field"] == "api_key"

    def test_missing_env_var_error(self):
        """Test error for missing environment variable."""
        error = ConfigurationError(
            "Required environment variable not set",
            details={"variable": "API_KEY"}
        )
        assert "API_KEY" in error.details["variable"]


class TestClientError:
    """Test suite for ClientError."""

    def test_is_docling_graph_error(self):
        """Test that ClientError extends DoclingGraphError."""
        error = ClientError("Client error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, ClientError)

    def test_client_error_with_model_info(self):
        """Test client error with model information."""
        error = ClientError(
            "API call failed",
            details={"model": "gpt-4", "status": 500}
        )
        assert error.details["model"] == "gpt-4"
        assert error.details["status"] == 500

    def test_client_error_with_cause(self):
        """Test client error wrapping another exception."""
        original = ConnectionError("Network error")
        error = ClientError(
            "Failed to connect to API",
            details={"endpoint": "https://api.example.com"},
            cause=original
        )
        assert error.cause is original
        assert isinstance(error.cause, ConnectionError)


class TestExtractionError:
    """Test suite for ExtractionError."""

    def test_is_docling_graph_error(self):
        """Test that ExtractionError extends DoclingGraphError."""
        error = ExtractionError("Extraction error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, ExtractionError)

    def test_extraction_error_with_source(self):
        """Test extraction error with source information."""
        error = ExtractionError(
            "Failed to extract data",
            details={"source": "document.pdf", "page": 5}
        )
        assert error.details["source"] == "document.pdf"
        assert error.details["page"] == 5

    def test_no_models_extracted_error(self):
        """Test error for no models extracted."""
        error = ExtractionError(
            "No models extracted from document",
            details={"source": "document.pdf", "template": "MyTemplate"}
        )
        assert "document.pdf" in error.details["source"]


class TestValidationError:
    """Test suite for ValidationError."""

    def test_is_docling_graph_error(self):
        """Test that ValidationError extends DoclingGraphError."""
        error = ValidationError("Validation error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, ValidationError)

    def test_validation_error_with_field_info(self):
        """Test validation error with field information."""
        error = ValidationError(
            "Invalid field value",
            details={"field": "email", "value": "invalid", "expected": "email format"}
        )
        assert error.details["field"] == "email"
        assert error.details["expected"] == "email format"


class TestGraphError:
    """Test suite for GraphError."""

    def test_is_docling_graph_error(self):
        """Test that GraphError extends DoclingGraphError."""
        error = GraphError("Graph error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, GraphError)

    def test_graph_error_with_stats(self):
        """Test graph error with graph statistics."""
        error = GraphError(
            "Graph validation failed",
            details={"nodes": 0, "edges": 0, "expected_nodes": 10}
        )
        assert error.details["nodes"] == 0
        assert error.details["expected_nodes"] == 10


class TestPipelineError:
    """Test suite for PipelineError."""

    def test_is_docling_graph_error(self):
        """Test that PipelineError extends DoclingGraphError."""
        error = PipelineError("Pipeline error")
        assert isinstance(error, DoclingGraphError)
        assert isinstance(error, PipelineError)

    def test_pipeline_error_with_stage_info(self):
        """Test pipeline error with stage information."""
        error = PipelineError(
            "Pipeline failed at stage",
            details={"stage": "Extraction", "error": "timeout"}
        )
        assert error.details["stage"] == "Extraction"
        assert error.details["error"] == "timeout"

    def test_pipeline_error_with_cause(self):
        """Test pipeline error wrapping stage exception."""
        original = ExtractionError("Extraction failed")
        error = PipelineError(
            "Pipeline execution failed",
            details={"stage": "Extraction"},
            cause=original
        )
        assert error.cause is original
        assert isinstance(error.cause, ExtractionError)


class TestExceptionHierarchy:
    """Test suite for exception hierarchy relationships."""

    def test_all_inherit_from_base(self):
        """Test that all exceptions inherit from DoclingGraphError."""
        exceptions = [
            ConfigurationError("test"),
            ClientError("test"),
            ExtractionError("test"),
            ValidationError("test"),
            GraphError("test"),
            PipelineError("test")
        ]
        for exc in exceptions:
            assert isinstance(exc, DoclingGraphError)
            assert isinstance(exc, Exception)

    def test_exceptions_are_distinct(self):
        """Test that exception types are distinct."""
        config_err = ConfigurationError("test")
        client_err = ClientError("test")

        assert not isinstance(config_err, ClientError)
        assert not isinstance(client_err, ConfigurationError)

    def test_can_catch_specific_exceptions(self):
        """Test catching specific exception types."""
        def raise_client_error() -> NoReturn:
            raise ClientError("API failed")

        with pytest.raises(ClientError):
            raise_client_error()

        # Should also be catchable as base type
        with pytest.raises(DoclingGraphError):
            raise_client_error()

    def test_exception_chaining(self):
        """Test exception chaining with cause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ClientError("Wrapped error", cause=e) from e
        except ClientError as exc:
            assert exc.cause is not None
            assert isinstance(exc.cause, ValueError)
            assert exc.__cause__ is exc.cause

# Made with Bob
