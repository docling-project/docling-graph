"""Unit tests for the deterministic identifier utilities (templategen.naming)."""

import pytest

from docling_graph.templategen.naming import (
    BANNED_EDGE_LABELS,
    EDGE_VERB_PREFIXES,
    RESERVED_FIELD_RENAMES,
    RESERVED_NODE_ATTRS,
    RESERVED_TEMPLATE_FIELD_NAMES,
    derive_edge_label,
    is_verb_phrase,
    normalize_edge_label,
    sanitize_class_name,
    sanitize_field_name,
    to_pascal_case,
    to_snake_case,
)

# ---------------------------------------------------------------------------
# Case conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("line_item", "LineItem"),
        ("lineItem", "LineItem"),
        ("line-item", "LineItem"),
        ("Line item", "LineItem"),
        ("LINE_ITEM", "LineItem"),
        ("USB_cable", "USBCable"),  # short acronyms survive
        ("insurance  policy", "InsurancePolicy"),
        ("Invoice", "Invoice"),
    ],
)
def test_to_pascal_case(raw, expected):
    assert to_pascal_case(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LineItem", "line_item"),
        ("hasLineItem", "has_line_item"),
        ("line-item", "line_item"),
        ("Line Item", "line_item"),
        ("HTTPServer", "http_server"),
        ("policy_number", "policy_number"),
    ],
)
def test_to_snake_case(raw, expected):
    assert to_snake_case(raw) == expected


# ---------------------------------------------------------------------------
# Edge-label normalization (relationships.md conventions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "target", "expected"),
    [
        # camelCase verb phrases -> UPPER_SNAKE
        ("hasLineItem", None, "HAS_LINE_ITEM"),
        ("worksFor", None, "WORKS_FOR"),
        ("issuedBy", None, "ISSUED_BY"),
        # kebab-case and spaces
        ("issued-by", None, "ISSUED_BY"),
        ("located at", None, "LOCATED_AT"),
        # banned vague labels -> HAS_<TARGET>
        ("related", "Party", "HAS_PARTY"),
        ("RELATED_TO", "LineItem", "HAS_LINE_ITEM"),
        ("LINK", "Bien", "HAS_BIEN"),
        ("has", "Guarantee", "HAS_GUARANTEE"),
        ("is", "Person", "HAS_PERSON"),
        ("of", "Section", "HAS_SECTION"),
        # bare single-token nouns -> HAS_<NOUN>
        ("address", None, "HAS_ADDRESS"),
        ("premium", None, "HAS_PREMIUM"),
        # single-word verbs stay (including present-tense forms)
        ("covers", None, "COVERS"),
        ("contains", None, "CONTAINS"),
        ("employs", None, "EMPLOYS"),
        ("knows", None, "KNOWS"),
        ("pays", None, "PAYS"),
        ("issues", None, "ISSUES"),  # generated from ISSUED at module load
        # verb-prefixed multi-word labels stay (IS banned only when bare)
        ("IS_PERSON", None, "IS_PERSON"),
        ("HAS_GUARANTEE", None, "HAS_GUARANTEE"),
        ("REFERENCES_ITEM", None, "REFERENCES_ITEM"),
        ("ownsVehicle", None, "OWNS_VEHICLE"),
        ("grantsCoverage", None, "GRANTS_COVERAGE"),
        # multi-token labels with an UNKNOWN first token are kept as-is —
        # user-chosen verb phrases are never mangled (R9 advisory instead)
        ("managedBy", None, "MANAGED_BY"),
        ("line item", None, "LINE_ITEM"),
        ("insurance policy", None, "INSURANCE_POLICY"),
        # already-normalized labels are stable
        ("CONTAINS_LINE", None, "CONTAINS_LINE"),
    ],
)
def test_normalize_edge_label(raw, target, expected):
    assert normalize_edge_label(raw, target=target) == expected


def test_normalize_never_produces_has_prefixed_verb_phrases():
    # The regression finding: OWNS_VEHICLE / KNOWS / GRANTS_COVERAGE /
    # MANAGED_BY must never come out HAS_-mangled.
    assert normalize_edge_label("OWNS_VEHICLE") == "OWNS_VEHICLE"
    assert normalize_edge_label("KNOWS") == "KNOWS"
    assert normalize_edge_label("GRANTS_COVERAGE") == "GRANTS_COVERAGE"
    assert normalize_edge_label("MANAGED_BY") == "MANAGED_BY"


def test_present_tense_forms_generated_from_past_forms():
    # V ending 'ED' -> V[:-2]+'S' and V[:-1]+'S' at module load.
    assert "ISSUES" in EDGE_VERB_PREFIXES  # from ISSUED
    assert "OWNS" in EDGE_VERB_PREFIXES  # from OWNED (and explicit)
    assert "BILLS" in EDGE_VERB_PREFIXES  # from BILLED
    # explicit present-tense entries
    for verb in ("KNOWS", "PAYS", "MANAGES", "GRANTS", "REPRESENTS", "SENDS"):
        assert verb in EDGE_VERB_PREFIXES


@pytest.mark.parametrize(
    ("field_name", "target", "expected"),
    [
        # derivation (label-less edges) DOES prefix multi-token noun phrases
        ("line_items", "LineItem", "HAS_LINE_ITEMS"),
        ("sections", "Section", "HAS_SECTIONS"),
        # verb-phrase field names are kept
        ("owns_vehicle", "Vehicle", "OWNS_VEHICLE"),
        ("issued_by", "Party", "ISSUED_BY"),
        # banned/empty names fall back to HAS_<TARGET>
        ("has", "Party", "HAS_PARTY"),
        ("---", "Line Item", "HAS_LINE_ITEM"),
    ],
)
def test_derive_edge_label(field_name, target, expected):
    assert derive_edge_label(field_name, target) == expected


def test_normalize_edge_label_is_idempotent():
    label = normalize_edge_label("hasLineItem")
    assert normalize_edge_label(label) == label


def test_banned_label_without_target_raises():
    with pytest.raises(ValueError, match="banned"):
        normalize_edge_label("related")


def test_empty_label_raises():
    with pytest.raises(ValueError, match="no letters"):
        normalize_edge_label("---")


def test_ban_list_matches_relationships_doc():
    assert BANNED_EDGE_LABELS == {"HAS", "LINK", "RELATED", "RELATED_TO", "IS", "OF"}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ISSUED_BY", True),
        ("worksFor", True),
        ("OWNS_VEHICLE", True),
        ("GRANTS_COVERAGE", True),
        ("address", False),
        ("LINE_ITEM", False),
        ("MANAGED_BY", False),
    ],
)
def test_is_verb_phrase(label, expected):
    assert is_verb_phrase(label) is expected


# ---------------------------------------------------------------------------
# Keyword / builtin / reserved-node-attr sanitation
# ---------------------------------------------------------------------------


def test_reserved_node_attrs_match_graph_converter():
    # GraphConverter._create_nodes_pass writes exactly these keys on every node.
    assert RESERVED_NODE_ATTRS == {"id", "label", "type", "__class__"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # reserved node-attr keys get the fixed rename map
        ("id", "identifier"),
        ("ID", "identifier"),
        ("label", "name_label"),
        ("type", "category"),
        # Python keywords
        ("class", "class_field"),
        ("for", "for_field"),
        ("import", "import_field"),
        # builtins
        ("sum", "sum_field"),
        ("input", "input_field"),
        # template module-level names (date/datetime annotations, the edge()
        # helper, the module logger) must not be rebound by a field
        ("date", "date_field"),
        ("datetime", "datetime_field"),
        ("edge", "edge_field"),
        ("logger", "logger_field"),
        # ordinary names are snake_cased only
        ("Total Amount", "total_amount"),
        ("taxId", "tax_id"),
        ("unit-price", "unit_price"),
        ("issue_date", "issue_date"),  # only the exact name collides
        # degenerate inputs
        ("2nd_line", "field_2nd_line"),
        ("---", "field"),
    ],
)
def test_sanitize_field_name(raw, expected):
    assert sanitize_field_name(raw) == expected


def test_reserved_template_field_names_pinned():
    assert RESERVED_TEMPLATE_FIELD_NAMES == {"date", "datetime", "edge", "logger"}


def test_rename_map_covers_all_renameable_reserved_attrs():
    for attr in RESERVED_NODE_ATTRS - {"__class__"}:
        assert sanitize_field_name(attr) == RESERVED_FIELD_RENAMES[attr]
        assert sanitize_field_name(attr) not in RESERVED_NODE_ATTRS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("invoice document", "InvoiceDocument"),
        ("line_item", "LineItem"),
        # builtins and template-module names must not be shadowed
        ("Exception", "ExceptionModel"),
        ("Field", "FieldModel"),
        ("Enum", "EnumModel"),
        ("None", "NoneModel"),
        # scalar-type-named classes (JSON-Schema $defs often carry 'date',
        # 'int', ...) are force-suffixed so scalar fields stay unambiguous
        ("date", "DateModel"),
        ("Date", "DateModel"),
        ("datetime", "DatetimeModel"),
        ("int", "IntModel"),
        ("str", "StrModel"),
        ("bool", "BoolModel"),
        ("float", "FloatModel"),
        # degenerate inputs
        ("", "Model"),
        ("3d model", "Model3dModel"),
    ],
)
def test_sanitize_class_name(raw, expected):
    assert sanitize_class_name(raw) == expected
