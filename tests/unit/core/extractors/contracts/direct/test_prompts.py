"""
Unit tests for LLM prompt generation.

Tests the simplified prompt generation for direct extraction:
- get_extraction_prompt() for standard extraction
- No capability-based branching (removed)
- No consolidation prompts (removed)
- No delta/context-aware prompts (removed)
"""

import pytest

from docling_graph.core.extractors.contracts.direct import get_extraction_prompt


class TestGetExtractionPrompt:
    """Test get_extraction_prompt() function."""

    def test_partial_extraction_prompt(self):
        """Test prompt generation for partial document (page/chunk)."""
        markdown = "This is page 1 content."
        schema = '{"title": "Test", "type": "object"}'

        prompt_dict = get_extraction_prompt(markdown, schema, is_partial=True)

        # Verify structure
        assert "system" in prompt_dict
        assert "user" in prompt_dict
        assert isinstance(prompt_dict["system"], str)
        assert isinstance(prompt_dict["user"], str)

        # Verify system prompt content
        system = prompt_dict["system"]
        assert "extract" in system.lower() or "extraction" in system.lower()
        assert "partial" in system.lower() or "page" in system.lower()
        assert "json" in system.lower()

        # Verify user prompt content
        user = prompt_dict["user"]
        assert markdown in user
        assert schema in user
        assert "page" in user.lower() or "document" in user.lower()

    def test_complete_extraction_prompt(self):
        """Test prompt generation for complete document."""
        markdown = "This is the full document content."
        schema = '{"title": "Test", "type": "object"}'

        prompt_dict = get_extraction_prompt(markdown, schema, is_partial=False)

        # Verify structure
        assert "system" in prompt_dict
        assert "user" in prompt_dict
        assert isinstance(prompt_dict["system"], str)
        assert isinstance(prompt_dict["user"], str)

        # Verify system prompt content
        system = prompt_dict["system"]
        assert "extract" in system.lower() or "extraction" in system.lower()
        assert "complete" in system.lower() or "thorough" in system.lower()
        assert "json" in system.lower()

        # Verify user prompt content
        user = prompt_dict["user"]
        assert markdown in user
        assert schema in user
        assert "complete" in user.lower() or "document" in user.lower()

    def test_extraction_instructions_present(self):
        """Test that extraction instructions are present in system prompt."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"]
        # Check for key instruction elements
        assert "read" in system.lower() or "extract" in system.lower()
        assert "json" in system.lower()
        assert "schema" in system.lower()

    def test_relationship_guidance_present(self):
        """Test that relationship handling guidance is present."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"]
        # Check for relationship-specific guidance
        # The instructions should mention not using empty arrays for relationships
        assert "omit" in system.lower() or "missing" in system.lower()

    def test_backward_compatibility_model_config(self):
        """Test that model_config parameter is accepted (backward compatibility)."""
        from types import SimpleNamespace

        markdown = "Test content"
        schema = '{"title": "Test"}'

        # Should work with model_config parameter (even though it's not used)
        model_config = SimpleNamespace(capability="STANDARD")
        prompt_dict = get_extraction_prompt(markdown, schema, model_config=model_config)

        assert "system" in prompt_dict
        assert "user" in prompt_dict

    def test_backward_compatibility_no_model_config(self):
        """Test that function works without model_config parameter."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        # Should work without model_config parameter
        prompt_dict = get_extraction_prompt(markdown, schema)

        assert "system" in prompt_dict
        assert "user" in prompt_dict

    def test_schema_included_in_user_prompt(self):
        """Test that schema is included in user prompt."""
        markdown = "Test content"
        schema = '{"title": "TestSchema", "properties": {"name": {"type": "string"}}}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        user = prompt_dict["user"]
        assert schema in user
        assert "schema" in user.lower()

    def test_markdown_included_in_user_prompt(self):
        """Test that markdown content is included in user prompt."""
        markdown = "This is unique test content 12345"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        user = prompt_dict["user"]
        assert markdown in user

    def test_prompt_structure_consistent(self):
        """Test that prompt structure is consistent across calls."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt1 = get_extraction_prompt(markdown, schema, is_partial=True)
        prompt2 = get_extraction_prompt(markdown, schema, is_partial=True)

        # Same inputs should produce same outputs
        assert prompt1["system"] == prompt2["system"]
        assert prompt1["user"] == prompt2["user"]

    def test_partial_vs_complete_different_prompts(self):
        """Test that partial and complete prompts are different."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        partial_prompt = get_extraction_prompt(markdown, schema, is_partial=True)
        complete_prompt = get_extraction_prompt(markdown, schema, is_partial=False)

        # System prompts should be different
        assert partial_prompt["system"] != complete_prompt["system"]
        # User prompts should be different (different document type)
        assert partial_prompt["user"] != complete_prompt["user"]


class TestPromptQuality:
    """Test prompt quality and best practices."""

    def test_no_empty_default_guidance(self):
        """Test that prompts guide against empty defaults."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"].lower()
        # Should mention not using empty strings/arrays/objects
        assert any(phrase in system for phrase in ["empty", "omit", "not applicable", "missing"])

    def test_json_format_emphasis(self):
        """Test that JSON format is emphasized."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"]
        user = prompt_dict["user"]

        # JSON should be mentioned in both system and user prompts
        assert "json" in system.lower()
        assert "json" in user.lower()

    def test_schema_validation_guidance(self):
        """Test that schema validation is mentioned."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"].lower()
        # Should mention following/matching the schema
        assert "schema" in system

    def test_evidence_based_extraction_guidance(self):
        """Test that evidence-based extraction is emphasized."""
        markdown = "Test content"
        schema = '{"title": "Test"}'

        prompt_dict = get_extraction_prompt(markdown, schema)

        system = prompt_dict["system"].lower()
        # Should mention extracting only what's in the text
        assert any(phrase in system for phrase in ["text", "document", "provided", "evidenced"])


# Note: The following test classes are removed as the functions no longer exist:
# - TestGetConsolidationPrompt (get_consolidation_prompt removed)
# - TestAdaptivePrompting (capability-based prompting removed)
# - TestChainOfDensity (advanced consolidation removed)
# - TestContextAwarePrompts (delta extraction removed)
#
# These were part of the old delta extraction and capability detection system
# which has been replaced by direct extraction.
