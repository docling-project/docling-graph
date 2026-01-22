from typing import Any, Dict, List, Optional
import networkx as nx
from neo4j import GraphDatabase, Driver
from rich import print as rich_print

class Neo4jExporter:
    """Exporter for populating a live Neo4j database."""

    def __init__(
        self,
        uri: str,
        auth: Optional[tuple[str, str]] = None,
        database: str = "neo4j",
        batch_size: int = 1000,
        write_mode: str = "merge",  # "merge" or "create"
    ):
        """
        Initialize the Neo4j exporter.
        
        Args:
            uri: Neo4j database URI (e.g., 'bolt://localhost:7687')
            auth: Tuple of (username, password)
            database: Database name to use
            batch_size: Number of records to commit in a single transaction
            write_mode: Strategy for writing nodes ('merge' updates existing, 'create' adds new)
        """
        self.uri = uri
        self.auth = auth
        self.database = database
        self.batch_size = batch_size
        self.write_mode = write_mode.lower()
        self._driver: Optional[Driver] = None

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def export(self, graph: nx.DiGraph) -> None:
        """
        Export the NetworkX graph to Neo4j.
        
        Args:
            graph: The NetworkX directed graph to export
        """
        if graph.number_of_nodes() == 0:
            rich_print("[yellow]Graph is empty. Skipping Neo4j export.[/yellow]")
            return

        driver = self._get_driver()
        
        try:
            with driver.session(database=self.database) as session:
                # 1. Export Nodes
                self._export_nodes(session, graph)
                
                # 2. Export Relationships
                self._export_edges(session, graph)
                
            rich_print(f"[green]Successfully exported graph to Neo4j database '{self.database}'[/green]")
        except Exception as e:
            rich_print(f"[red]Failed to export to Neo4j:[/red] {e}")
            raise
        finally:
            self.close()

    def _export_nodes(self, session, graph: nx.DiGraph) -> None:
        """Batch write nodes to Neo4j."""
        batch: List[Dict[str, Any]] = []
        
        query = (
            "UNWIND $batch AS row "
            f"{'MERGE' if self.write_mode == 'merge' else 'CREATE'} (n:Node {{id: row.id}}) "
            "SET n += row.properties, n.label = row.label "
            "WITH n, row "
            "CALL apoc.create.addLabels(n, [row.label]) YIELD node "  # Optional: requires APOC, fallback to simple label setting if needed
            "RETURN count(*)"
        )
        
        # Simplified query without APOC dependency
        query = (
            "UNWIND $batch AS row "
            f"{'MERGE' if self.write_mode == 'merge' else 'CREATE'} (n:Node {{id: row.id}}) "
            "SET n += row.properties "
        )

        # Strategy: Group nodes by label to allow static label assignment
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        
        for node_id, data in graph.nodes(data=True):
            label = data.get("label", "Entity")
            # Sanitize label
            label = "".join(x for x in label if x.isalnum() or x == "_")
            if not label: 
                label = "Entity"
                
            props = {k: v for k, v in data.items() if k != "label"}
            props["id"] = node_id  # Ensure ID is a property
            
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(props)

        total_nodes = 0
        for label, nodes in nodes_by_label.items():
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i : i + self.batch_size]
                cypher = (
                    f"UNWIND $batch AS row "
                    f"{'MERGE' if self.write_mode == 'merge' else 'CREATE'} (n:{label} {{id: row.id}}) "
                    "SET n += row.properties"
                )
                session.run(cypher, batch=batch)
                total_nodes += len(batch)
                
        rich_print(f"   - Exported {total_nodes} nodes")

    def _export_edges(self, session, graph: nx.DiGraph) -> None:
        """Batch write edges to Neo4j."""
        edges_by_type: Dict[str, List[Dict[str, Any]]] = {}

        for u, v, data in graph.edges(data=True):
            rel_type = data.get("label", "RELATED_TO").upper()
            # Sanitize relationship type
            rel_type = "".join(x for x in rel_type if x.isalnum() or x == "_")
            if not rel_type:
                rel_type = "RELATED_TO"
            
            props = {k: v for k, v in data.items() if k != "label"}
            props["source_id"] = u
            props["target_id"] = v
            
            if rel_type not in edges_by_type:
                edges_by_type[rel_type] = []
            edges_by_type[rel_type].append(props)

        total_edges = 0
        for rel_type, edges in edges_by_type.items():
            for i in range(0, len(edges), self.batch_size):
                batch = edges[i : i + self.batch_size]
                cypher = (
                    "UNWIND $batch AS row "
                    "MATCH (source {id: row.source_id}) "
                    "MATCH (target {id: row.target_id}) "
                    f"{'MERGE' if self.write_mode == 'merge' else 'CREATE'} (source)-[r:{rel_type}]->(target) "
                    "SET r += row "  # This sets source_id/target_id on rel too, which is harmless but redundant
                )
                session.run(cypher, batch=batch)
                total_edges += len(batch)

        rich_print(f"   - Exported {total_edges} edges")