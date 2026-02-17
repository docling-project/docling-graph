"""Unit tests for entity name normalizer."""

import pytest

from docling_graph.core.utils.entity_name_normalizer import normalize_entity_name


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
