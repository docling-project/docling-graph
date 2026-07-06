"""Unit tests for root-identity repair (class-name echo + source-stem fallback)."""

from pydantic import BaseModel, Field

from docling_graph.core.utils.root_identity import (
    is_class_name_echo,
    repair_root_identity,
)


class AssuranceMRH(BaseModel):
    """Insurance-shaped root: one string identity field."""

    reference_document: str = Field(default="")
    version: str | None = None

    model_config = {"graph_id_fields": ["reference_document"]}


class NoIdentityRoot(BaseModel):
    title: str = ""


class TestIsClassNameEcho:
    def test_exact_and_spaced_echoes_match(self):
        assert is_class_name_echo("AssuranceMRH", "AssuranceMRH")
        assert is_class_name_echo("assurance mrh", "AssuranceMRH")
        assert is_class_name_echo("Assurance_MRH", "AssuranceMRH")

    def test_real_values_do_not_match(self):
        assert not is_class_name_echo("HABITATION_07-25", "AssuranceMRH")
        assert not is_class_name_echo("", "AssuranceMRH")
        assert not is_class_name_echo(None, "AssuranceMRH")
        assert not is_class_name_echo(42, "AssuranceMRH")


class TestRepairRootIdentity:
    def test_class_name_echo_is_cleared_then_stem_applied(self):
        root = AssuranceMRH(reference_document="AssuranceMRH")
        repair_root_identity(root, document_stem="insurance_terms")
        assert root.reference_document == "insurance_terms"

    def test_empty_identity_falls_back_to_stem(self):
        root = AssuranceMRH(reference_document="")
        repair_root_identity(root, document_stem="insurance_terms")
        assert root.reference_document == "insurance_terms"

    def test_real_identity_is_never_touched(self):
        root = AssuranceMRH(reference_document="HABITATION_07-25")
        repair_root_identity(root, document_stem="insurance_terms")
        assert root.reference_document == "HABITATION_07-25"

    def test_no_stem_leaves_identity_empty(self):
        root = AssuranceMRH(reference_document="AssuranceMRH")
        repair_root_identity(root)
        assert root.reference_document == ""

    def test_template_without_id_fields_is_noop(self):
        root = NoIdentityRoot(title="hello")
        repair_root_identity(root, document_stem="stem")
        assert root.title == "hello"

    def test_unusable_stem_is_rejected(self):
        root = AssuranceMRH(reference_document="")
        repair_root_identity(root, document_stem="   ")
        assert root.reference_document == ""
        repair_root_identity(root, document_stem="x" * 200)
        assert root.reference_document == ""

    def test_multi_field_identity_partially_filled_is_untouched(self):
        class TwoIdRoot(BaseModel):
            code: str = ""
            name: str = ""
            model_config = {"graph_id_fields": ["code", "name"]}

        root = TwoIdRoot(code="", name="Real Name")
        repair_root_identity(root, document_stem="stem")
        assert root.code == ""
        assert root.name == "Real Name"
