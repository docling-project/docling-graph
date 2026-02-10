"""
Example 06: Export Formats - CSV, Cypher, and JSON

Description:
    Demonstrates different export formats for knowledge graphs. Processes the same
    billing document with three different export formats to show Neo4j integration options
    and general-purpose data exchange.

Use Cases:
    - Neo4j bulk import (CSV)
    - Neo4j script execution (Cypher)
    - API integration (JSON)
    - Data archival and exchange

Prerequisites:
    - Installation: uv sync
    - No API keys required (uses local VLM)
    - Optional: Neo4j for testing imports

Key Concepts:
    - CSV Export: Neo4j-compatible nodes and edges files
    - Cypher Export: CREATE statements for Neo4j
    - JSON Export: Complete graph structure
    - Format Selection: Based on use case

Expected Output:
    - CSV: nodes.csv + edges.csv (Neo4j bulk import)
    - Cypher: graph.cypher (Neo4j script)
    - JSON: graph.json (general purpose)

Related Examples:
    - Example 01: Basic VLM extraction
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/graph-management/export-formats/
"""

import sys
from pathlib import Path

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.billing_document import BillingDocument

    from docling_graph import PipelineConfig, run_pipeline
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg"
TEMPLATE_CLASS = BillingDocument

console = Console()


def export_csv() -> None:
    """Export as CSV for Neo4j bulk import."""
    console.print("\n[bold cyan]1. CSV Export (Neo4j Bulk Import)[/bold cyan]")

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        backend="vlm",
        inference="local",
        processing_mode="one-to-one",
        docling_config="vision",
        export_format="csv",  # CSV export
    )

    console.print("  • Processing with CSV export...")
    context = run_pipeline(config)
    console.print("  • [green]✓ Complete[/green]")
    graph = context.knowledge_graph
    console.print(f"  • Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def export_cypher() -> None:
    """Export as Cypher for Neo4j script execution."""
    console.print("\n[bold cyan]2. Cypher Export (Neo4j Script)[/bold cyan]")

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        backend="vlm",
        inference="local",
        processing_mode="one-to-one",
        docling_config="vision",
        export_format="cypher",  # Cypher export
    )

    console.print("  • Processing with Cypher export...")
    context = run_pipeline(config)
    console.print("  • [green]✓ Complete[/green]")
    graph = context.knowledge_graph
    console.print(f"  • Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def main() -> None:
    """Execute export format demonstrations."""
    console.print(
        Panel.fit(
            "[bold blue]Example 06: Export Formats[/bold blue]\n"
            "[dim]Compare CSV, Cypher, and JSON export formats[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Overview:[/yellow]")
    console.print("  Process the same billing document with different export formats:")
    console.print("  1. CSV - Neo4j bulk import")
    console.print("  2. Cypher - Neo4j script execution")
    console.print("  3. JSON - General purpose (always included)")

    try:
        export_csv()
        export_cypher()

        # Format comparison
        console.print("\n[bold]📊 Format Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Format")
        table.add_column("Files")
        table.add_column("Best For")
        table.add_column("Neo4j Import")

        table.add_row("CSV", "nodes.csv, edges.csv", "Production bulk import", "neo4j-admin import")
        table.add_row(
            "Cypher", "graph.cypher", "Development, incremental", "cypher-shell < file.cypher"
        )
        table.add_row("JSON", "graph.json", "API integration, archival", "Custom import script")

        console.print(table)

        console.print("\n[bold]💡 When to Use Each Format:[/bold]")
        console.print("\n[cyan]CSV (Bulk Import):[/cyan]")
        console.print("  • Large-scale production imports")
        console.print("  • Best performance for big datasets")
        console.print("  • Requires Neo4j admin access")
        console.print("  • Command: [dim]neo4j-admin database import full[/dim]")

        console.print("\n[cyan]Cypher (Script):[/cyan]")
        console.print("  • Development and testing")
        console.print("  • Incremental updates")
        console.print("  • Works with any Neo4j instance")
        console.print("  • Command: [dim]cypher-shell < graph.cypher[/dim]")

        console.print("\n[cyan]JSON (General Purpose):[/cyan]")
        console.print("  • API integration")
        console.print("  • Data archival")
        console.print("  • Custom processing")
        console.print("  • Always included with other formats")

        console.print("\n[bold]📊 Output Locations:[/bold]")
        console.print("  • CSV: [cyan]outputs/06_export_formats/csv/docling_graph/[/cyan]")
        console.print("  • Cypher: [cyan]outputs/06_export_formats/cypher/docling_graph/[/cyan]")

        console.print("\n[bold]🔧 Neo4j Import Examples:[/bold]")
        console.print("\n[cyan]CSV Bulk Import:[/cyan]")
        console.print("  [dim]neo4j-admin database import full \\[/dim]")
        console.print(
            "    [dim]--nodes=outputs/06_export_formats/csv/docling_graph/nodes.csv \\[/dim]"
        )
        console.print(
            "    [dim]--relationships=outputs/06_export_formats/csv/docling_graph/edges.csv[/dim]"
        )

        console.print("\n[cyan]Cypher Script:[/cyan]")
        console.print(
            "  [dim]cat outputs/06_export_formats/cypher/docling_graph/graph.cypher | \\[/dim]"
        )
        console.print("    [dim]cypher-shell -u neo4j -p password[/dim]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure dependencies: [cyan]uv sync[/cyan]")
        console.print("  • Check source file exists")
        console.print("  • Verify template is correctly defined")
        sys.exit(1)


if __name__ == "__main__":
    main()
