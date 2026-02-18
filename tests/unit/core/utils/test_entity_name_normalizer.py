"""Unit tests for entity name normalizer."""

import pytest

from docling_graph.core.utils.entity_name_normalizer import (
    canonicalize_identity_for_dedup,
    normalize_entity_name,
)


def test_john_doe():
    assert normalize_entity_name("John Doe") == "JOHN_DOE"


def test_the_company():
    assert normalize_entity_name("The Company") == "COMPANY"
    assert normalize_entity_name("the company") == "COMPANY"


def test_sarah_chen_whitespace():
    assert normalize_entity_name("  Sarah  Chen  ") == "SARAH_CHEN"


def test_empty_string():
    assert normalize_entity_name("") == ""


def test_whitespace_only():
    assert normalize_entity_name("   ") == ""


def test_single_word():
    assert normalize_entity_name("OpenAI") == "OPENAI"
    assert normalize_entity_name("  Apple  ") == "APPLE"


def test_prefix_a_an():
    assert normalize_entity_name("A Person") == "PERSON"
    assert normalize_entity_name("An Event") == "EVENT"


def test_possessive():
    assert normalize_entity_name("John's") == "JOHN"
    assert normalize_entity_name("Company's Products") == "COMPANY_PRODUCTS"


def test_case_insensitive():
    assert normalize_entity_name("john doe") == "JOHN_DOE"
    assert normalize_entity_name("JOHN DOE") == "JOHN_DOE"


def test_none_input():
    assert normalize_entity_name(None) == ""


def test_the_strips_to_empty():
    # "The " with trailing space strips to empty; single word "The" stays THE
    assert normalize_entity_name("The ") == ""
    assert normalize_entity_name("The  ") == ""


def test_non_string_input_returns_empty():
    """Non-string input (e.g. int) returns empty string."""
    assert normalize_entity_name(123) == ""
    assert normalize_entity_name(0) == ""


def test_single_word_article_returns_empty():
    """Single-word articles 'The', 'A', 'An' (and lowercase) normalize to empty."""
    assert normalize_entity_name("The") == ""
    assert normalize_entity_name("the") == ""
    assert normalize_entity_name("A") == ""
    assert normalize_entity_name("a") == ""
    assert normalize_entity_name("An") == ""
    assert normalize_entity_name("an") == ""


def test_unicode_possessive_right_single_quote():
    """Unicode right single quote (\\u2019) possessive is stripped like ASCII apostrophe."""
    # \u2019 is the Unicode right single quotation mark (e.g. from smart quotes)
    assert normalize_entity_name("Company\u2019s") == "COMPANY"
    assert normalize_entity_name("John\u2019s Report") == "JOHN_REPORT"


def test_prefix_strip_then_multiple_words():
    """Prefix strip (trimmed[len(prefix):].strip() and break) then multiple words normalized."""
    assert normalize_entity_name("A New Hope") == "NEW_HOPE"
    assert normalize_entity_name("The Alpha Beta") == "ALPHA_BETA"


def test_unicode_possessive_word_appended():
    """Unicode possessive branch (word[-2:] == \\u2019s) strips suffix then words.append(word) (line 40)."""
    assert normalize_entity_name("Test\u2019s") == "TEST"


def test_canonicalize_identity_for_dedup_name_fields():
    """name/title/nom use normalize_entity_name."""
    assert canonicalize_identity_for_dedup("name", "John Doe") == "JOHN_DOE"
    assert canonicalize_identity_for_dedup("title", "The Company") == "COMPANY"
    assert canonicalize_identity_for_dedup("nom", "  Alpha  ") == "ALPHA"


def test_canonicalize_identity_for_dedup_identifier_collapse():
    """Identifier-style id fields collapse run_1/run1/Run-1 to same canonical form."""
    assert canonicalize_identity_for_dedup("run_id", "run_1") == "run1"
    assert canonicalize_identity_for_dedup("run_id", "run1") == "run1"
    assert canonicalize_identity_for_dedup("run_id", "Run-1") == "run1"
    assert canonicalize_identity_for_dedup("batch_id", "batch1") == "batch1"
    assert canonicalize_identity_for_dedup("batch_id", "batch_1") == "batch1"
    assert canonicalize_identity_for_dedup("dataset_id", "dataset1") == "dataset1"
    assert canonicalize_identity_for_dedup("dataset_id", "dataset_1") == "dataset1"


def test_canonicalize_identity_for_dedup_none_empty():
    """None and empty return empty string."""
    assert canonicalize_identity_for_dedup("run_id", None) == ""
    assert canonicalize_identity_for_dedup("name", "") == ""
    assert canonicalize_identity_for_dedup("batch_id", "  ") == ""
