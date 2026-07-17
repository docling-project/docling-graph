"""Cypher exporter semantics against a real Neo4j.

The unit tests pin the script's text; these tests pin what actually matters:
that the script executes, is idempotent, loads the right graph, and preserves
property types. Statements are executed one by one exactly the way
``cypher-shell`` consumes the file (split on ``;`` outside quotes).

Backend selection:
    * ``NEO4J_TEST_URI`` (with optional ``NEO4J_TEST_AUTH=user/password``)
      points at a running instance, e.g. a CI docker service.
    * Otherwise a testcontainers Neo4j is started; skipped when Docker is
      unavailable.

The database-backed classes are marked ``neo4j``: the CI matrix deselects them
(``-m "not neo4j"``) to stay Docker-free, and the dedicated ``neo4j-integration``
job runs them against a service container. Run them locally with
``pytest -m neo4j`` once Docker is up.
"""

import os
from pathlib import Path

import networkx as nx
import pytest

from docling_graph.core.exporters.cypher_exporter import CypherExporter

neo4j = pytest.importorskip("neo4j", reason="neo4j driver not installed")

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(600)]

_NEO4J_IMAGE = "neo4j:5.26"
_TC_PASSWORD = "docling-graph-test"


def split_cypher_statements(script: str) -> list[str]:
    """Split a script on ``;`` the way cypher-shell does.

    Quote-aware (``'``, ``"``, backticks, with backslash escapes) and drops
    ``//`` line comments, so a ``;`` inside a string literal never splits.
    """
    statements: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(script):
        ch = script[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote in "'\"" and i + 1 < len(script):
                buf.append(script[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
            buf.append(ch)
        elif script.startswith("//", i):
            end = script.find("\n", i)
            i = len(script) if end == -1 else end
            continue
        elif ch == ";":
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


class TestStatementSplitter:
    """The splitter must mirror cypher-shell, or the suite tests the wrong thing."""

    def test_splits_on_semicolons_outside_strings(self):
        script = 'CREATE (n {a: "x; y"});\nMATCH (n) RETURN n;\n'
        assert split_cypher_statements(script) == [
            'CREATE (n {a: "x; y"})',
            "MATCH (n) RETURN n",
        ]

    def test_drops_comments_and_keeps_escaped_quotes(self):
        script = '// header\nSET n.p = "say \\"hi;\\"";\n'
        assert split_cypher_statements(script) == ['SET n.p = "say \\"hi;\\""']

    def test_exported_script_splits_into_expected_statements(self, tmp_path):
        statements = split_cypher_statements(export(build_graph(), tmp_path).read_text("utf-8"))
        # 3 constraints + 4 nodes + 2 relationships
        assert len(statements) == 9
        assert all(s.startswith(("CREATE CONSTRAINT", "MERGE", "MATCH")) for s in statements)


@pytest.fixture(scope="module")
def driver():
    """Neo4j driver for an external instance or a throwaway container."""
    uri = os.environ.get("NEO4J_TEST_URI")
    if uri:
        user, _, password = os.environ.get("NEO4J_TEST_AUTH", "neo4j/password").partition("/")
        drv = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        yield drv
        drv.close()
        return

    testcontainers_neo4j = pytest.importorskip(
        "testcontainers.neo4j", reason="testcontainers not installed"
    )
    try:
        container = testcontainers_neo4j.Neo4jContainer(_NEO4J_IMAGE, password=_TC_PASSWORD)
        container.start()
    except Exception as exc:
        pytest.skip(f"Cannot start Neo4j container (is Docker running?): {exc}")
    try:
        drv = neo4j.GraphDatabase.driver(
            container.get_connection_url(), auth=("neo4j", _TC_PASSWORD)
        )
        yield drv
        drv.close()
    finally:
        container.stop()


class Neo4jSuite:
    """Base for tests that talk to Neo4j: each starts from an empty graph.

    Subclasses inherit the ``neo4j`` mark, so the Docker-dependent tests are
    deselectable as a group while the splitter tests keep running everywhere.
    """

    pytestmark = pytest.mark.neo4j

    @pytest.fixture(autouse=True)
    def clean_database(self, driver):
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()


def load_script(driver, path: Path) -> None:
    """Execute every statement of an exported script, cypher-shell style."""
    with driver.session() as session:
        for statement in split_cypher_statements(path.read_text("utf-8")):
            session.run(statement).consume()


def counts(driver) -> tuple[int, int]:
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return nodes, rels


def build_graph() -> nx.DiGraph:
    """4 nodes, 2 edges, every property type the exporter serializes."""
    graph = nx.DiGraph()
    graph.add_node(
        "m1",
        label="Measurement",
        id="m1",
        value=3.14,
        count=42,
        verified=True,
        tags=["shear", "yield"],
        provenance={"chunks": [0], "match": "verbatim"},
    )
    graph.add_node("p1", label="Paper", id="p1", title='Alpha "prime"')
    graph.add_node("p2", label="Paper", id="p2", title="Beta")
    graph.add_node("a1", label="Author", id="a1", name="Grace")
    graph.add_edge("p1", "m1", label="HAS_MEASUREMENT", weight=0.9)
    graph.add_edge("p1", "a1", label="AUTHORED_BY")
    return graph


def export(graph: nx.DiGraph, tmp_path: Path, style: str = "merge") -> Path:
    path = tmp_path / f"graph_{style}.cypher"
    CypherExporter(style=style).export(graph, path)
    return path


class TestMergeStyleAgainstNeo4j(Neo4jSuite):
    def test_script_executes(self, driver, tmp_path):
        """The exported script parses and runs — broken before the rewrite."""
        load_script(driver, export(build_graph(), tmp_path))

    def test_first_load_matches_source_graph(self, driver, tmp_path):
        """Against an empty DB, counts must equal the source graph's."""
        graph = build_graph()
        load_script(driver, export(graph, tmp_path))
        assert counts(driver) == (graph.number_of_nodes(), graph.number_of_edges())

    def test_reload_is_idempotent(self, driver, tmp_path):
        """Running the script twice must not duplicate nodes or relationships."""
        graph = build_graph()
        path = export(graph, tmp_path)
        load_script(driver, path)
        first = counts(driver)
        load_script(driver, path)
        assert counts(driver) == first == (graph.number_of_nodes(), graph.number_of_edges())

    def test_no_cartesian_product(self, driver, tmp_path):
        """4 nodes + 2 edges must produce exactly 2 relationships, on the
        declared endpoints — an unfiltered MATCH would explode to 16 rows."""
        load_script(driver, export(build_graph(), tmp_path))
        with driver.session() as session:
            assert (
                session.run(
                    'MATCH (:Paper {id: "p1"})-[r:HAS_MEASUREMENT]->(:Measurement {id: "m1"}) '
                    "RETURN count(r) AS c"
                ).single()["c"]
                == 1
            )
        assert counts(driver)[1] == 2

    def test_property_types_round_trip(self, driver, tmp_path):
        """float/int/bool/list arrive in Neo4j as their native types."""
        load_script(driver, export(build_graph(), tmp_path))
        with driver.session() as session:
            node = session.run('MATCH (n:Measurement {id: "m1"}) RETURN n').single()["n"]
        assert node["value"] == 3.14 and type(node["value"]) is float
        assert node["count"] == 42 and type(node["count"]) is int
        assert node["verified"] is True
        assert node["tags"] == ["shear", "yield"]
        assert node["title"] is None  # no attribute bleed between nodes
        with driver.session() as session:
            paper = session.run('MATCH (p:Paper {id: "p1"}) RETURN p').single()["p"]
        assert paper["title"] == 'Alpha "prime"'

    def test_reload_updates_instead_of_duplicating(self, driver, tmp_path):
        """A changed property must update the node, not create a sibling —
        this is why only the identity key lives in the MERGE pattern."""
        graph = build_graph()
        load_script(driver, export(graph, tmp_path))
        graph.nodes["m1"]["value"] = 2.71
        load_script(driver, export(graph, tmp_path / "updated"))
        assert counts(driver) == (graph.number_of_nodes(), graph.number_of_edges())
        with driver.session() as session:
            value = session.run('MATCH (n:Measurement {id: "m1"}) RETURN n.value AS v').single()
        assert value["v"] == 2.71


class TestCreateStyleAgainstNeo4j(Neo4jSuite):
    def test_create_style_first_load(self, driver, tmp_path):
        """CREATE style loads a correct, typed graph into an empty DB."""
        graph = build_graph()
        load_script(driver, export(graph, tmp_path, style="create"))
        assert counts(driver) == (graph.number_of_nodes(), graph.number_of_edges())
        with driver.session() as session:
            node = session.run('MATCH (n:Measurement {id: "m1"}) RETURN n').single()["n"]
        assert node["value"] == 3.14 and type(node["value"]) is float
        assert node["verified"] is True

    def test_create_style_reload_fails_loudly(self, driver, tmp_path):
        """A second CREATE run must trip the uniqueness constraint instead of
        silently duplicating the graph."""
        path = export(build_graph(), tmp_path, style="create")
        load_script(driver, path)
        with pytest.raises(neo4j.exceptions.ConstraintError):
            load_script(driver, path)
