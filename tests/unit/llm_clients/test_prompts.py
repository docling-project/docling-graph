"""
Tests for LLM prompt generation utilities.
"""

import pytest

from docling_graph.llm_clients.prompts import get_legacy_prompt, get_prompt


class TestGetPrompt:
    """Test prompt generation for new dict-based format."""

    def test_get_prompt_complete_document(self):
        """Should generate prompts for complete document extraction."""
        markdown = "# Invoice\nAmount: $100"
        schema = '{"amount": "number"}'

        result = get_prompt(markdown, schema, is_partial=False)

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result
        assert result["system"] is not None
        assert result["user"] is not None

    def test_get_prompt_complete_document_structure(self):
        """Should include proper structure in complete document prompts."""
        markdown = "# Invoice\nAmount: $100"
        schema = '{"amount": "number"}'

        result = get_prompt(markdown, schema, is_partial=False)

        # System prompt should have extraction instructions
        assert "extract" in result["system"].lower()
        assert "structured information" in result["system"].lower()

        # User prompt should contain document content and schema
        assert markdown in result["user"]
        assert schema in result["user"]
        assert "COMPLETE DOCUMENT" in result["user"]

    def test_get_prompt_partial_document(self):
        """Should generate prompts for partial document extraction."""
        markdown = "Page 2\nSome data"
        schema = '{"field": "string"}'

        result = get_prompt(markdown, schema, is_partial=True)

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result

    def test_get_prompt_partial_document_structure(self):
        """Should include proper structure in partial document prompts."""
        markdown = "Page 2\nSome data"
        schema = '{"field": "string"}'

        result = get_prompt(markdown, schema, is_partial=True)

        # System prompt should indicate partial extraction is okay
        assert "page" in result["system"].lower() or "partial" in result["system"].lower()

        # User prompt should contain page content
        assert markdown in result["user"]
        assert "DOCUMENT PAGE" in result["user"]

    def test_get_prompt_returns_valid_json_instructions(self):
        """Should include JSON output instructions."""
        markdown = "Test document"
        schema = '{"test": "string"}'

        result = get_prompt(markdown, schema, is_partial=False)

        # Should instruct to return valid JSON
        assert "json" in result["system"].lower()
        assert "valid" in result["system"].lower()

    def test_get_prompt_includes_empty_field_handling(self):
        """Should instruct how to handle empty fields."""
        markdown = "Incomplete data"
        schema = '{"field1": "string", "field2": "array"}'

        result = get_prompt(markdown, schema, is_partial=False)

        system = result["system"].lower()
        # Should include instructions for empty fields
        assert '""' in result["system"] or "empty string" in system
        assert "[]" in result["system"] or "array" in system

    def test_get_prompt_markdown_content_properly_formatted(self):
        """Should format markdown content clearly in user prompt."""
        markdown = "# Title\n## Subtitle\nContent here"
        schema = '{"test": "string"}'

        result = get_prompt(markdown, schema, is_partial=False)

        # Markdown should be in user prompt with clear delimiters
        assert "===" in result["user"]
        assert markdown in result["user"]

    def test_get_prompt_schema_clearly_marked(self):
        """Should clearly mark schema in user prompt."""
        markdown = "Test"
        schema = '{"field": "type"}'

        result = get_prompt(markdown, schema, is_partial=False)

        # Schema should be clearly marked
        assert "SCHEMA" in result["user"]
        assert schema in result["user"]


class TestGetLegacyPrompt:
    """Test legacy single-string prompt format."""

    def test_get_legacy_prompt_returns_string(self):
        """Should return a single string."""
        markdown = "Test content"
        schema = '{"field": "string"}'

        result = get_legacy_prompt(markdown, schema, is_partial=False)

        assert isinstance(result, str)

    def test_get_legacy_prompt_combines_system_and_user(self):
        """Should combine system and user prompts."""
        markdown = "Test"
        schema = '{"f": "s"}'

        result = get_legacy_prompt(markdown, schema, is_partial=False)

        # Should contain both system and user content
        assert "extract" in result.lower()
        assert "Test" in result

    def test_get_legacy_prompt_complete_document(self):
        """Should handle complete document extraction."""
        markdown = "Invoice\nAmount: $100"
        schema = '{"amount": "number"}'

        result = get_legacy_prompt(markdown, schema, is_partial=False)

        assert markdown in result
        assert schema in result
        assert isinstance(result, str)

    def test_get_legacy_prompt_partial_document(self):
        """Should handle partial document extraction."""
        markdown = "Page 2\nData"
        schema = '{"field": "string"}'

        result = get_legacy_prompt(markdown, schema, is_partial=True)

        assert markdown in result
        assert schema in result
        assert isinstance(result, str)

    def test_get_legacy_prompt_backwards_compatible(self):
        """Legacy format should produce usable prompts."""
        markdown = "Sample document"
        schema = '{"test": "string"}'

        result = get_legacy_prompt(markdown, schema, is_partial=False)

        # Should be a reasonable prompt string
        assert len(result) > len(markdown) + len(schema)
        assert "\n" in result  # Should have some structure


class TestPromptConsistency:
    """Test consistency between prompt formats."""

    def test_get_prompt_and_legacy_contain_same_info(self):
        """Dict and legacy prompts should contain same information."""
        markdown = "Test document"
        schema = '{"field": "string"}'

        dict_result = get_prompt(markdown, schema)
        legacy_result = get_legacy_prompt(markdown, schema)

        combined_dict = dict_result["system"] + dict_result["user"]

        # Both should contain the content
        assert markdown in combined_dict
        assert markdown in legacy_result
        assert schema in combined_dict
        assert schema in legacy_result

    def test_prompt_extraction_instructions_consistent(self):
        """Extraction instructions should be consistent."""
        markdown = "Document"
        schema = '{"f": "t"}'

        dict_result = get_prompt(markdown, schema)

        # System prompt should have clear extraction instructions
        system = dict_result["system"]
        assert "extract" in system.lower()
        assert "json" in system.lower()

    def test_partial_vs_complete_prompts_differ(self):
        """Partial and complete prompts should differ appropriately."""
        markdown = "Test"
        schema = '{"f": "s"}'

        partial = get_prompt(markdown, schema, is_partial=True)
        complete = get_prompt(markdown, schema, is_partial=False)

        # Should be different
        assert partial["system"] != complete["system"]
        assert partial["user"] != complete["user"]
