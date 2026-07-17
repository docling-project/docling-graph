"""Tests for the tolerant ``templategen:`` config-block reader (design §8)."""

import pytest
from pydantic import ValidationError

from docling_graph.templategen.settings import TemplateGenSettings, load_templategen_settings

EXPECTED_DEFAULTS = {
    "input_budget_chars": 24_000,
    "max_models": 30,
    "max_enum_members": 24,
    "ontology_depth": 4,
    "llm_gap_fill": False,
    "strict": False,
    "strategy": "one-shot",
    "workers": 4,
    "max_units": 24,
    "max_windows_per_doc": 6,
    "saturation_stop": True,
}


class TestDefaults:
    def test_design_section_8_defaults(self):
        assert TemplateGenSettings().model_dump() == EXPECTED_DEFAULTS

    def test_none_config_yields_defaults(self):
        # Template commands must not hard-require config.yaml.
        assert load_templategen_settings(None) == TemplateGenSettings()

    def test_config_without_block_yields_defaults(self):
        config = {"defaults": {"backend": "llm"}, "models": {}}
        assert load_templategen_settings(config) == TemplateGenSettings()

    def test_explicit_none_block_yields_defaults(self):
        assert load_templategen_settings({"templategen": None}) == TemplateGenSettings()


class TestPartialBlock:
    def test_partial_block_keeps_other_defaults(self):
        settings = load_templategen_settings(
            {"templategen": {"max_models": 50, "llm_gap_fill": True}}
        )
        assert settings.max_models == 50
        assert settings.llm_gap_fill is True
        assert settings.input_budget_chars == 24_000
        assert settings.ontology_depth == 4
        assert settings.strict is False

    def test_full_block(self):
        block = {
            "input_budget_chars": 8_000,
            "max_models": 10,
            "max_enum_members": 12,
            "ontology_depth": 2,
            "llm_gap_fill": True,
            "strict": True,
            "strategy": "three-pass",
            "workers": 2,
            "max_units": 10,
            "max_windows_per_doc": 3,
            "saturation_stop": False,
        }
        assert load_templategen_settings({"templategen": block}).model_dump() == block


class TestRejection:
    def test_unknown_key_lists_valid_keys(self):
        with pytest.raises(ValueError, match="Unknown templategen setting"):
            load_templategen_settings({"templategen": {"max_modles": 10}})
        # The error must name every valid key (the typo-recovery UX).
        with pytest.raises(ValueError) as exc_info:
            load_templategen_settings({"templategen": {"max_modles": 10}})
        message = str(exc_info.value)
        assert "max_modles" in message
        for key in EXPECTED_DEFAULTS:
            assert key in message

    def test_non_mapping_block_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_templategen_settings({"templategen": ["strict"]})

    def test_wrong_typed_value_raises_validation_error(self):
        with pytest.raises(ValidationError):
            load_templategen_settings({"templategen": {"input_budget_chars": "plenty"}})

    def test_non_positive_int_rejected(self):
        with pytest.raises(ValidationError):
            load_templategen_settings({"templategen": {"max_models": 0}})

    def test_settings_model_forbids_extras(self):
        with pytest.raises(ValidationError):
            TemplateGenSettings(unknown_key=1)
