"""Tests for ResponseHandler - centralized JSON parsing."""

import pytest

from docling_graph.exceptions import ClientError
from docling_graph.llm_clients.response_handler import ResponseHandler


class TestResponseHandler:
    """Test suite for ResponseHandler."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        response = ResponseHandler.parse_json_response(
            '{"key": "value", "number": 42}',
            "TestClient"
        )
        assert response == {"key": "value", "number": 42}

    def test_parse_json_with_markdown_code_block(self):
        """Test parsing JSON from markdown code block."""
        response = ResponseHandler.parse_json_response(
            '```json\n{"key": "value"}\n```',
            "TestClient"
        )
        assert response == {"key": "value"}

    def test_parse_json_with_backticks_only(self):
        """Test parsing JSON with backticks but no json marker."""
        response = ResponseHandler.parse_json_response(
            '```\n{"key": "value"}\n```',
            "TestClient"
        )
        assert response == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text (aggressive clean)."""
        response = ResponseHandler.parse_json_response(
            'Here is the JSON: {"key": "value"} and some more text',
            "TestClient",
            aggressive_clean=True
        )
        assert response == {"key": "value"}

    def test_parse_json_aggressive_clean_multiple_objects(self):
        """Test aggressive clean with multiple JSON objects (takes first)."""
        response = ResponseHandler.parse_json_response(
            '{"first": "object"} and {"second": "object"}',
            "TestClient",
            aggressive_clean=True
        )
        assert response == {"first": "object"}

    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        json_str = '{"outer": {"inner": {"deep": "value"}}}'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response == {"outer": {"inner": {"deep": "value"}}}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        json_str = '[{"id": 1}, {"id": 2}]'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response == [{"id": 1}, {"id": 2}]

    def test_parse_empty_json_object(self):
        """Test parsing empty JSON object."""
        response = ResponseHandler.parse_json_response("{}", "TestClient")
        assert response == {}

    def test_parse_json_with_special_characters(self):
        """Test parsing JSON with special characters."""
        json_str = '{"text": "Line 1\\nLine 2\\tTabbed"}'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response == {"text": "Line 1\nLine 2\tTabbed"}

    def test_parse_json_with_unicode(self):
        """Test parsing JSON with Unicode characters."""
        json_str = '{"name": "José", "emoji": "🎉"}'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response == {"name": "José", "emoji": "🎉"}

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises ClientError."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response(
                "not valid json at all",
                "TestClient"
            )
        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.details["client_name"] == "TestClient"

    def test_parse_empty_string_raises_error(self):
        """Test that empty string raises ClientError."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response("", "TestClient")
        assert "empty" in str(exc_info.value).lower()

    def test_parse_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises ClientError."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response("   \n\t  ", "TestClient")
        assert "empty" in str(exc_info.value).lower()

    def test_parse_none_raises_error(self):
        """Test that None raises ClientError."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response(None, "TestClient")
        assert "empty" in str(exc_info.value).lower()

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing comma (should fail)."""
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response(
                '{"key": "value",}',
                "TestClient"
            )

    def test_parse_json_with_comments_fails(self):
        """Test that JSON with comments fails (not valid JSON)."""
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response(
                '{"key": "value"} // comment',
                "TestClient"
            )

    def test_aggressive_clean_extracts_from_text(self):
        """Test aggressive clean extracts JSON from prose."""
        text = """
        The analysis shows the following results:
        {"status": "success", "count": 42}
        This indicates a positive outcome.
        """
        response = ResponseHandler.parse_json_response(
            text,
            "TestClient",
            aggressive_clean=True
        )
        assert response == {"status": "success", "count": 42}

    def test_aggressive_clean_with_no_json_raises_error(self):
        """Test aggressive clean with no JSON raises error."""
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response(
                "This text contains no JSON at all",
                "TestClient",
                aggressive_clean=True
            )

    def test_error_includes_raw_response_snippet(self):
        """Test that error includes snippet of raw response."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response(
                "invalid json content here",
                "TestClient"
            )
        # Should include snippet in details
        assert "raw_response" in exc_info.value.details

    def test_parse_json_with_numbers(self):
        """Test parsing JSON with various number types."""
        json_str = '{"int": 42, "float": 3.14, "negative": -10, "exp": 1e5}'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response["int"] == 42
        assert response["float"] == 3.14
        assert response["negative"] == -10
        assert response["exp"] == 1e5

    def test_parse_json_with_booleans_and_null(self):
        """Test parsing JSON with booleans and null."""
        json_str = '{"true_val": true, "false_val": false, "null_val": null}'
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert response["true_val"] is True
        assert response["false_val"] is False
        assert response["null_val"] is None

    def test_parse_large_json(self):
        """Test parsing large JSON object."""
        large_obj = {f"key_{i}": f"value_{i}" for i in range(1000)}
        import json
        json_str = json.dumps(large_obj)
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert len(response) == 1000
        assert response["key_0"] == "value_0"
        assert response["key_999"] == "value_999"

# Made with Bob
