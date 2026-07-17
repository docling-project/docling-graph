"""
Tests for Cypher exporter.
"""

from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.converters.config import ExportConfig
from docling_graph.core.exporters.cypher_exporter import CypherExporter


@pytest.fixture
def sample_graph():
    """Create a sample graph."""
    graph = nx.DiGraph()
    graph.add_node("n1", label="Person", name="John")
    graph.add_node("n2", label="Company", name="ACME")
    graph.add_edge("n1", "n2", label="works_for", strength=0.9)
    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return nx.DiGraph()


class TestCypherExporterInitialization:
    """Test CypherExporter initialization."""

    def test_initialization_default(self):
        """Should initialize with default config."""
        exporter = CypherExporter()
        assert exporter.config is not None

    def test_initialization_custom_config(self):
        """Should accept custom config."""
        config = ExportConfig()
        exporter = CypherExporter(config=config)
        assert exporter.config is config


class TestCypherExporterValidation:
    """Test graph validation."""

    def test_validate_graph_with_nodes(self, sample_graph):
        """Should return True for non-empty graph."""
        exporter = CypherExporter()
        assert exporter.validate_graph(sample_graph) is True

    def test_validate_graph_empty(self, empty_graph):
        """Should return False for empty graph."""
        exporter = CypherExporter()
        assert exporter.validate_graph(empty_graph) is False


class TestCypherExporterStringEscaping:
    """Test Cypher string escaping."""

    def test_escape_backslash(self):
        """Should escape backslashes."""
        result = CypherExporter._escape_cypher_string("path\\to\\file")
        assert "\\\\" in result

    def test_escape_quotes(self):
        """Should escape quotes."""
        result = CypherExporter._escape_cypher_string('say "hello"')
        assert '\\"' in result

    def test_escape_newlines(self):
        """Should escape newlines."""
        result = CypherExporter._escape_cypher_string("line1\nline2")
        assert "\\n" in result

    def test_escape_single_quotes(self):
        """Should escape single quotes."""
        result = CypherExporter._escape_cypher_string("it's")
        assert "\\'" in result

    def test_escape_non_string(self):
        """Should convert non-strings to string."""
        result = CypherExporter._escape_cypher_string(42)
        assert result == "42"


class TestCypherExporterIdentifierSanitization:
    """Test Cypher identifier sanitization."""

    def test_sanitize_alphanumeric(self):
        """Should keep alphanumeric characters."""
        result = CypherExporter._sanitize_identifier("node123")
        assert result == "node123"

    def test_sanitize_replaces_special_chars(self):
        """Should replace special characters with underscore."""
        result = CypherExporter._sanitize_identifier("node-1_2")
        assert result == "node_1_2"

    def test_sanitize_starts_with_letter(self):
        """Should ensure identifier starts with letter."""
        result = CypherExporter._sanitize_identifier("123node")
        assert result == "n_123node"

    def test_sanitize_empty_becomes_node(self):
        """Empty string should become 'node'."""
        result = CypherExporter._sanitize_identifier("")
        assert result == "node"

    def test_sanitize_only_special_chars(self):
        """String with only special chars becomes underscores."""
        result = CypherExporter._sanitize_identifier("---")
        assert result == "___"

    def test_sanitize_unicode(self):
        """Should handle unicode characters by replacing them."""
        result = CypherExporter._sanitize_identifier("nöde")
        assert result == "n_de"

    def test_sanitize_underscore_preserved(self):
        """Underscores should be preserved."""
        result = CypherExporter._sanitize_identifier("node_name")
        assert result == "node_name"

    def test_sanitize_mixed_content(self):
        """Should handle mixed alphanumeric and special chars."""
        result = CypherExporter._sanitize_identifier("node@#$%name")
        assert result == "node____name"

    def test_sanitize_leading_numbers_prefixed(self):
        """Leading numbers should be prefixed with 'n_'."""
        result = CypherExporter._sanitize_identifier("9to5job")
        assert result == "n_9to5job"


class TestCypherExporterExport:
    """Test Cypher export functionality."""

    def test_export_creates_file(self, sample_graph, tmp_path):
        """Should create Cypher file."""
        exporter = CypherExporter()
        output_file = tmp_path / "graph.cypher"

        exporter.export(sample_graph, output_file)

        assert output_file.exists()

    def test_export_empty_graph_raises_error(self, empty_graph, tmp_path):
        """Should raise error for empty graph."""
        exporter = CypherExporter()

        with pytest.raises(ValueError):
            exporter.export(empty_graph, tmp_path / "output.cypher")

    def test_export_creates_parent_directories(self, sample_graph, tmp_path):
        """Should create parent directories if needed."""
        exporter = CypherExporter()
        output_file = tmp_path / "nested" / "deep" / "graph.cypher"

        exporter.export(sample_graph, output_file)

        assert output_file.exists()

    def test_export_contains_header(self, sample_graph, tmp_path):
        """Exported file should contain header comment."""
        exporter = CypherExporter()
        output_file = tmp_path / "graph.cypher"

        exporter.export(sample_graph, output_file)

        content = output_file.read_text()
        assert "// Cypher script generated" in content

    def test_export_contains_create_nodes(self, sample_graph, tmp_path):
        """Exported file should contain CREATE NODE statements."""
        exporter = CypherExporter()
        output_file = tmp_path / "graph.cypher"

        exporter.export(sample_graph, output_file)

        content = output_file.read_text()
        assert "CREATE" in content
        assert "Person" in content or "Company" in content

    def test_export_contains_create_relationships(self, sample_graph, tmp_path):
        """Exported file should contain relationship creation."""
        exporter = CypherExporter()
        output_file = tmp_path / "graph.cypher"

        exporter.export(sample_graph, output_file)

        content = output_file.read_text()
        assert "CREATE" in content
        assert "->" in content

    def test_export_is_valid_cypher_syntax(self, sample_graph, tmp_path):
        """Exported content should look like valid Cypher."""
        exporter = CypherExporter()
        output_file = tmp_path / "graph.cypher"

        exporter.export(sample_graph, output_file)

        content = output_file.read_text()
        # Check for basic Cypher syntax elements
        assert "CREATE" in content
        assert ":" in content  # Labels
        assert "->" in content or "<-" in content  # Relationships


def _statements(script: str) -> list[str]:
    """Non-comment statements of a script, the way cypher-shell splits them."""
    code = "\n".join(
        line for line in script.splitlines() if line.strip() and not line.strip().startswith("//")
    )
    return [s.strip() for s in code.split(";") if s.strip()]


def _export(graph, tmp_path, **kwargs: str) -> str:
    output_file = tmp_path / "graph.cypher"
    CypherExporter(**kwargs).export(graph, output_file)
    return output_file.read_text()


@pytest.fixture
def typed_graph():
    """Graph exercising every property type."""
    graph = nx.DiGraph()
    graph.add_node(
        "m1",
        label="Measurement",
        id="m1",
        value=3.14,
        count=42,
        verified=True,
        tags=["shear", "yield"],
        note=None,
        provenance={"chunks": [0], "match": "verbatim"},
    )
    graph.add_node("p1", label="Paper", id="p1", title="Alpha")
    graph.add_edge("p1", "m1", label="HAS_MEASUREMENT", weight=0.9)
    return graph


class TestCypherExporterMergeStyle:
    """Test the default idempotent MERGE output."""

    def test_default_style_is_merge(self):
        """Merge is the default: the broken CREATE output was never executable."""
        assert CypherExporter().style == "merge"

    def test_invalid_style_rejected(self):
        """Unknown styles should fail fast at construction."""
        with pytest.raises(ValueError, match="style"):
            CypherExporter(style="upsert")

    def test_nodes_merged_on_id_only(self, typed_graph, tmp_path):
        """Only the identity key may appear in the MERGE pattern."""
        content = _export(typed_graph, tmp_path)
        assert 'MERGE (n:Measurement {id: "m1"})' in content
        assert 'MERGE (n:Paper {id: "p1"})' in content
        # Mutable properties live in SET, never in the MERGE pattern
        assert "value" not in content.split("SET")[0].split("MERGE")[-1]

    def test_properties_set_with_types(self, typed_graph, tmp_path):
        """Numbers, bools and lists must be Cypher literals, not strings."""
        content = _export(typed_graph, tmp_path)
        assert "n.value = 3.14" in content
        assert "n.count = 42" in content
        assert "n.verified = true" in content
        assert 'n.tags = ["shear", "yield"]' in content

    def test_none_properties_dropped(self, typed_graph, tmp_path):
        """None values should not be exported."""
        assert "note" not in _export(typed_graph, tmp_path)

    def test_dict_property_becomes_json_string(self, typed_graph, tmp_path):
        """Dicts (e.g. provenance) are stored as parseable JSON strings."""
        content = _export(typed_graph, tmp_path)
        assert 'n.provenance = "{\\"chunks\\": [0], \\"match\\": \\"verbatim\\"}"' in content

    def test_relationship_matches_by_label_and_id(self, typed_graph, tmp_path):
        """Endpoint MATCHes must be label + id filtered: no cartesian product."""
        content = _export(typed_graph, tmp_path)
        assert 'MATCH (a:Paper {id: "p1"}), (b:Measurement {id: "m1"})' in content
        assert "MERGE (a)-[r:HAS_MEASUREMENT]->(b)" in content
        assert "r.weight = 0.9" in content

    def test_constraint_per_distinct_label(self, typed_graph, tmp_path):
        """One id-uniqueness constraint per label, at the top of the script."""
        content = _export(typed_graph, tmp_path)
        assert (
            "CREATE CONSTRAINT measurement_id IF NOT EXISTS "
            "FOR (n:Measurement) REQUIRE n.id IS UNIQUE;" in content
        )
        assert (
            "CREATE CONSTRAINT paper_id IF NOT EXISTS "
            "FOR (n:Paper) REQUIRE n.id IS UNIQUE;" in content
        )
        assert content.count("CREATE CONSTRAINT") == 2
        assert content.index("CREATE CONSTRAINT") < content.index("MERGE (n:")

    def test_every_statement_terminated(self, typed_graph, tmp_path):
        """cypher-shell splits on ';' — the code must end with a terminator."""
        content = _export(typed_graph, tmp_path)
        code_lines = [
            line for line in content.splitlines() if line.strip() and not line.startswith("//")
        ]
        assert code_lines[-1].endswith(";")
        # constraints x2 + nodes x2 + relationship x1
        assert len(_statements(content)) == 5

    def test_statements_are_self_contained(self, typed_graph, tmp_path):
        """No statement may reference a variable bound in an earlier one."""
        for statement in _statements(_export(typed_graph, tmp_path)):
            if statement.startswith(("MERGE", "SET")):
                continue
            # A relationship statement must bind its own endpoints
            if "]->(" in statement:
                assert statement.startswith("MATCH")

    def test_node_without_id_attr_falls_back_to_node_key(self, tmp_path):
        """Plain nx graphs without id attributes must still key on identity."""
        graph = nx.DiGraph()
        graph.add_node("k1", label="Thing", name="A")
        graph.add_node("k2", label="Thing", name="B")
        graph.add_edge("k1", "k2", label="NEXT")
        content = _export(graph, tmp_path)
        assert 'MERGE (n:Thing {id: "k1"})' in content
        assert 'MATCH (a:Thing {id: "k1"}), (b:Thing {id: "k2"})' in content

    def test_exporter_is_reentrant(self, tmp_path):
        """Exporting two graphs with one instance must not leak state."""
        exporter = CypherExporter()
        first = nx.DiGraph()
        first.add_node("a1", label="Alpha", id="a1")
        second = nx.DiGraph()
        second.add_node("b1", label="Beta", id="b1")

        exporter.export(first, tmp_path / "first.cypher")
        exporter.export(second, tmp_path / "second.cypher")

        content = (tmp_path / "second.cypher").read_text()
        assert "b1" in content
        assert "a1" not in content
        assert not hasattr(exporter, "_node_vars")

    def test_string_values_escaped(self, tmp_path):
        """Quotes inside string values must not break out of the literal."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="Doc", id="n1", title='say "hi"')
        content = _export(graph, tmp_path)
        assert 'n.title = "say \\"hi\\""' in content

    def test_exotic_label_backtick_quoted(self, tmp_path):
        """Labels that are not plain identifiers are backtick-quoted."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="My Label", id="n1")
        content = _export(graph, tmp_path)
        assert "MERGE (n:`My Label` {id:" in content
        assert "FOR (n:`My Label`)" in content
        # ... but the constraint name is a sanitized plain identifier
        assert "CREATE CONSTRAINT my_label_id IF NOT EXISTS" in content

    def test_mixed_type_list_falls_back_to_json_string(self, tmp_path):
        """Neo4j rejects heterogeneous property lists — export as JSON string."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="Thing", id="n1", mixed=[1, "a"], nested=[{"k": 1}])
        content = _export(graph, tmp_path)
        assert 'n.mixed = "[1, \\"a\\"]"' in content
        assert 'n.nested = "[{\\"k\\": 1}]"' in content

    def test_int_float_list_coerced_to_floats(self, tmp_path):
        """Mixed int/float lists become homogeneous float lists."""
        graph = nx.DiGraph()
        graph.add_node("n1", label="Thing", id="n1", values=[1, 2.5])
        content = _export(graph, tmp_path)
        assert "n.values = [1.0, 2.5]" in content


class TestCypherExporterCreateStyle:
    """Test the opt-in non-idempotent CREATE output."""

    def test_create_nodes_inline_props(self, typed_graph, tmp_path):
        """CREATE style inlines typed properties on the node."""
        content = _export(typed_graph, tmp_path, style="create")
        assert (
            'CREATE (:Measurement {id: "m1", value: 3.14, count: 42, '
            'verified: true, tags: ["shear", "yield"]' in content
        )
        assert "MERGE" not in content

    def test_create_relationships_match_by_id(self, typed_graph, tmp_path):
        """Even CREATE style must bind endpoints by label + id."""
        content = _export(typed_graph, tmp_path, style="create")
        assert 'MATCH (a:Paper {id: "p1"}), (b:Measurement {id: "m1"})' in content
        assert "CREATE (a)-[:HAS_MEASUREMENT {weight: 0.9}]->(b);" in content

    def test_create_style_still_terminated_and_constrained(self, typed_graph, tmp_path):
        """CREATE style keeps ';' terminators and uniqueness constraints."""
        content = _export(typed_graph, tmp_path, style="create")
        assert content.count("CREATE CONSTRAINT") == 2
        assert len(_statements(content)) == 5
