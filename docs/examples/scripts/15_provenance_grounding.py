"""
Example 15: Data Grounding & Provenance

Description:
    Demonstrates deterministic node-to-source grounding: after extraction, every
    graph node carries a `__provenance__` attribute pointing back to the source
    chunk(s) and page(s) it was extracted from. This is fully deterministic (no
    extra LLM calls) and works with both the "direct" and "dense" extraction
    contracts.

Use Cases:
    - Auditing extracted data against the source document
    - Citation / "show your work" for downstream RAG or review UIs
    - Debugging why a field has a particular value

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Data: Sample rheology research included in repository

Key Concepts:
    - provenance="detailed": embeds character-offset spans in __provenance__
    - "verbatim" match: the node's identifier was found literally in the text
    - "observed" match: dense skeleton saw the node in a chunk, but not verbatim
      (approximate=True)
    - provenance.json: the full ledger, including chunk text, next to graph.json

Expected Output:
    - graph.json: nodes annotated with __provenance__
    - provenance.json: full grounding ledger with chunk text

Related Examples:
    - Example 02: Basic LLM extraction (this script builds on it)
    - Example 08: Chunking and consolidation
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/graph-management/provenance/
"""

import json
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
    from examples.templates.rheology_research import ScholarlyRheologyPaper

    from docling_graph import PipelineConfig, run_pipeline
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper
console = Console()


def main() -> None:
    """Extract with grounding enabled, then inspect node-level and ledger-level provenance."""
    console.print(
        Panel.fit(
            "[bold blue]Example 15: Data Grounding & Provenance[/bold blue]\n"
            "[dim]Trace every extracted node back to its source chunk and page[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Contract: [cyan]dense[/cyan] (richest grounding)")
    console.print("  • Provenance: [cyan]detailed[/cyan] (adds character spans)")

    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            provider_override="mistral",
            model_override="mistral-large-latest",
            processing_mode="many-to-one",
            extraction_contract="dense",
            use_chunking=True,
            provenance="detailed",
            dump_to_disk=True,
        )

        console.print("\n[yellow]⚙️  Processing...[/yellow]")
        context = run_pipeline(config)

        graph = context.knowledge_graph
        console.print(
            f"\n[green]✓ Extracted[/green] [cyan]{graph.number_of_nodes()} nodes[/cyan], "
            f"[cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        # 1. Node-level provenance: already attached to every entity node
        console.print("\n[bold]📍 Node-level grounding (__provenance__):[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Node")
        table.add_column("Match")
        table.add_column("Pages")

        for node_id, data in list(graph.nodes(data=True))[:8]:
            prov = data.get("__provenance__", {})
            if "scope" in prov:
                match, pages = "document", "-"
            else:
                match = prov.get("match") or prov.get("status", "?")
                pages = ", ".join(str(p) for p in prov.get("pages", [])) or "-"
            table.add_row(node_id, match, pages)

        console.print(table)

        # 2. Full ledger: self-contained, includes chunk TEXT
        graph_dir = context.output_manager.get_docling_graph_dir()
        ledger_path = graph_dir / "provenance.json"

        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            console.print(f"\n[bold]📖 Full ledger:[/bold] [cyan]{ledger_path}[/cyan]")
            console.print(f"  • Resolution: [cyan]{ledger['resolution']}[/cyan]")
            console.print(f"  • Chunks indexed: [cyan]{len(ledger['chunks'])}[/cyan]")
            console.print(f"  • Nodes grounded: [cyan]{len(ledger['nodes'])}[/cyan]")
            console.print(f"  • Bind stats: [cyan]{ledger['bind_stats']}[/cyan]")

            # Resolve one verbatim anchor down to its source text snippet
            for entry in ledger["nodes"].values():
                verbatim = [a for a in entry["anchors"] if a["kind"] == "verbatim"]
                if not verbatim:
                    continue
                anchor = verbatim[0]
                chunk = ledger["chunks"][str(anchor["chunk_id"])]
                start, end = anchor["span"]
                snippet = chunk["text"][start:end]
                console.print(
                    f"\n[bold]🔍 Example resolution:[/bold] {entry['node_type']} {entry['ids']}"
                )
                console.print(f'  • Found verbatim: [green]"{snippet}"[/green]')
                console.print(f"  • Page(s): [cyan]{chunk['page_numbers']}[/cyan]")
                console.print(f"  • Chunk text: [dim]{chunk['text'][:120]}...[/dim]")
                break

        console.print("\n[bold]💡 What Happened:[/bold]")
        console.print("  • Grounding ran as deterministic post-processing (zero extra LLM calls)")
        console.print("  • Nodes whose identifier appears verbatim got an exact chunk/page")
        console.print("  • Nodes only seen by the skeleton phase got an approximate location")
        console.print("  • provenance.json carries the full ledger, including chunk text,")
        console.print("    so it's self-contained for auditing without re-running extraction")

    except FileNotFoundError:
        console.print(f"\n[red]Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you're running from the project root directory")
        console.print("  • Check that the sample data exists in docs/examples/data/")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure dependencies installed: [cyan]uv sync[/cyan]")
        console.print(
            "  • Set your Mistral API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]"
        )
        console.print("  • Or switch to local inference: see Example 07")
        sys.exit(1)


if __name__ == "__main__":
    main()
