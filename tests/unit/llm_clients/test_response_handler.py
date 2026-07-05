"""Tests for ResponseHandler - centralized JSON parsing."""

import pytest

from docling_graph.exceptions import ClientError
from docling_graph.llm_clients.response_handler import ResponseHandler


class TestResponseHandler:
    """Test suite for ResponseHandler."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        response = ResponseHandler.parse_json_response(
            '{"key": "value", "number": 42}', "TestClient"
        )
        assert response == {"key": "value", "number": 42}

    def test_parse_json_with_markdown_code_block(self):
        """Test parsing JSON from markdown code block."""
        response = ResponseHandler.parse_json_response(
            '```json\n{"key": "value"}\n```', "TestClient"
        )
        assert response == {"key": "value"}

    def test_parse_json_with_backticks_only(self):
        """Test parsing JSON with backticks but no json marker."""
        response = ResponseHandler.parse_json_response('```\n{"key": "value"}\n```', "TestClient")
        assert response == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text (aggressive clean)."""
        response = ResponseHandler.parse_json_response(
            'Here is the JSON: {"key": "value"} and some more text',
            "TestClient",
            aggressive_clean=True,
        )
        assert response == {"key": "value"}

    def test_parse_json_aggressive_clean_multiple_objects(self):
        """Test aggressive clean with multiple JSON objects (takes first)."""
        response = ResponseHandler.parse_json_response(
            '{"first": "object"} and {"second": "object"}', "TestClient", aggressive_clean=True
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
            ResponseHandler.parse_json_response("not valid json at all", "TestClient")
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
        """Test parsing JSON with trailing comma (best-effort repair)."""
        response = ResponseHandler.parse_json_response('{"key": "value",}', "TestClient")
        assert response == {"key": "value"}

    def test_parse_json_with_comments_fails(self):
        """Test that JSON with comments fails (not valid JSON)."""
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response('{"key": "value"} // comment', "TestClient")

    def test_aggressive_clean_extracts_from_text(self):
        """Test aggressive clean extracts JSON from prose."""
        text = """
        The analysis shows the following results:
        {"status": "success", "count": 42}
        This indicates a positive outcome.
        """
        response = ResponseHandler.parse_json_response(text, "TestClient", aggressive_clean=True)
        assert response == {"status": "success", "count": 42}

    def test_aggressive_clean_with_no_json_raises_error(self):
        """Test aggressive clean with no JSON raises error."""
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response(
                "This text contains no JSON at all", "TestClient", aggressive_clean=True
            )

    def test_error_includes_raw_response_snippet(self):
        """Test that error includes snippet of raw response."""
        with pytest.raises(ClientError) as exc_info:
            ResponseHandler.parse_json_response("invalid json content here", "TestClient")
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

    def test_parse_json_repairs_missing_value_to_null(self):
        """Repair malformed JSON where a key has no value before comma."""
        malformed = '{"nodes":[{"path":"","ids":{"document_number":"3139"},"parent":null},{"path":"buyer","ids":{"name":"x"},"parent":,}]}'
        response = ResponseHandler.parse_json_response(malformed, "TestClient")
        assert isinstance(response, dict)
        assert "nodes" in response
        assert response["nodes"][1]["parent"] is None

    def test_parse_large_json(self):
        """Test parsing large JSON object."""
        large_obj = {f"key_{i}": f"value_{i}" for i in range(1000)}
        import json

        json_str = json.dumps(large_obj)
        response = ResponseHandler.parse_json_response(json_str, "TestClient")
        assert len(response) == 1000
        assert response["key_0"] == "value_0"
        assert response["key_999"] == "value_999"

    def test_parse_json_with_raw_newlines_and_tabs_in_strings(self):
        """LLM may emit raw newlines/tabs in string values; sanitizer escapes them."""
        # Simulates: "nom": "canalisations \n\t\t\t..."
        raw = '{"items": [{"nom": "canalisations \n\t\t\t\t\t"}]}'
        response = ResponseHandler.parse_json_response(raw, "TestClient")
        assert response == {"items": [{"nom": "canalisations \n\t\t\t\t\t"}]}

    def test_parse_json_with_broken_unicode_escape(self):
        """LLM may emit \\u with newline/whitespace before hex digits; sanitizer normalizes."""
        raw = '{"nom": "text\\u\n0009\\u\t0009"}'
        response = ResponseHandler.parse_json_response(raw, "TestClient")
        assert response == {"nom": "text\t\t"}


class TestLongestValidPrefixSalvage:
    """Truncated/stalled generations: recover every complete element emitted."""

    def test_recovers_nodes_before_broken_element_and_whitespace_runaway(self):
        """The exact failure shape from a guided-decoding stall: a bare value
        emitted as an object key ('{"Gardenwork"') followed by a whitespace
        runaway until max_tokens. All five complete nodes must be recovered."""
        raw = (
            '{"nodes": [{"i": 1, "path": "", "ids": {"document_number": "3139"}}, '
            '{"i": 2, "path": "seller", "ids": {"name": "Robert Schneider"}}, '
            '{"i": 3, "path": "buyer", "ids": {"name": "Pia Rutschmann"}}, '
            '{"i": 4, "path": "line_items[]", "ids": {"1": "Gardenwork"}}, '
            '{"i": 5, "path": "line_items[]", "ids": {"2": "Disposal of cuttings"}}, '
            '{"i": 6, "path": "line_items[].item", "ids": {"Gardenwork"' + ("\n    " * 100)
        )
        parsed = ResponseHandler.parse_json_response(
            raw, "TestClient", truncated=True, max_tokens=4092
        )
        assert isinstance(parsed, dict)
        assert len(parsed["nodes"]) == 5
        assert parsed["nodes"][4]["ids"] == {"2": "Disposal of cuttings"}

    def test_recovers_prefix_cut_mid_string_value(self):
        """Truncation mid string-value: complete prior elements survive."""
        raw = '{"items": [{"a": 1}, {"b": 2}, {"c": "unterminat'
        parsed = ResponseHandler.parse_json_response(
            raw, "TestClient", truncated=True, max_tokens=100
        )
        assert isinstance(parsed, dict)
        assert parsed["items"][:2] == [{"a": 1}, {"b": 2}]

    def test_totally_unrecoverable_still_raises(self):
        with pytest.raises(ClientError):
            ResponseHandler.parse_json_response(
                '{"nodes": [{"i"', "TestClient", truncated=True, max_tokens=10
            )

    def test_valid_json_untouched(self):
        parsed = ResponseHandler.parse_json_response(
            '{"nodes": [{"i": 1}]}', "TestClient", truncated=False
        )
        assert parsed == {"nodes": [{"i": 1}]}


class TestFastPath:
    """Well-formed JSON must bypass the character-level cleaning passes."""

    def test_valid_json_skips_cleaning(self):
        from unittest.mock import patch as _patch

        raw = '{"a": 1, "b": ["x", "y"]}'
        with _patch.object(ResponseHandler, "_clean_response") as mock_clean:
            result = ResponseHandler.parse_json_response(raw, "TestClient")
        mock_clean.assert_not_called()
        assert result == {"a": 1, "b": ["x", "y"]}

    def test_markdown_wrapped_json_still_parsed(self):
        raw = 'Some text\n```json\n{"a": 1}\n```'
        result = ResponseHandler.parse_json_response(raw, "TestClient")
        assert result == {"a": 1}


class TestListEnvelopeShapes:
    """Empty list envelopes are valid 'found nothing' answers, not warnings."""

    def test_empty_envelopes_pass_without_warning(self):
        from unittest.mock import patch

        for payload in ('{"nodes": []}', '{"items": []}', '{"merges": []}'):
            with patch("docling_graph.llm_clients.response_handler.rich_print") as mock_print:
                ResponseHandler.parse_json_response(payload, "TestClient")
            assert mock_print.call_count == 0, payload

    def test_all_null_scalar_values_still_warn(self):
        from unittest.mock import patch

        with patch("docling_graph.llm_clients.response_handler.rich_print") as mock_print:
            ResponseHandler.parse_json_response('{"a": null, "b": null}', "TestClient")
        assert mock_print.call_count == 1
