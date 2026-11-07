"""
Example 05: Processing Modes - One-to-One vs Many-to-One

Description:
    Side-by-side comparison of one-to-one and many-to-one processing modes.
    Uses a multi-page PDF with French ID cards to demonstrate when to use each mode.

Use Cases:
    - Batch documents (billing documents, forms): Use one-to-one
    - Continuous documents (reports, papers): Use many-to-one
    - Understanding mode selection
    - Comparing output structures

Prerequisites:
    - Installation: uv sync
    - Local LLM: Ollama with llama3:8b model
    - Data: Multi-page ID card PDF included

Key Concepts:
    - One-to-One: Each page → separate model instance
    - Many-to-One: All pages → single merged model
    - Mode Selection: Based on document structure
    - Output Differences: Multiple vs single graph

Expected Output:
    - Two output directories for comparison
    - One-to-one: Multiple ID card nodes
    - Many-to-one: Single merged result
    - Comparison report

Related Examples:
    - Example 02: Many-to-one for rheology researchs
    - Example 06: Export format comparison
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/pipeline-configuration/processing-modes/
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
    from docling_graph import PipelineConfig
    from examples.templates.id_card import IDCard
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# Configuration
SOURCE_FILE = "docs/examples/data/id_card/multi_french_id_cards.pdf"
TEMPLATE_CLASS = IDCard

console = Console()


def process_one_to_one() -> None:
    """Process with one-to-one mode: each page separately."""
    console.print("\n[bold cyan]1. One-to-One Mode[/bold cyan]")
    console.print("  • Each page processed independently")
    console.print("  • Creates separate ID card for each page")
    console.print("  • Best for: Batch documents, independent pages")

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        backend="llm",
        inference="local",
        provider_override="ollama",
        model_override="llama3:8b",
        # Key setting: one-to-one mode
        processing_mode="one-to-one",
        # No chunking in one-to-one (each page is separate)
        use_chunking=False,
    )

    console.print("  • Processing...")
    config.run()
    console.print("  • [green]✓ Complete[/green]")


def process_many_to_one() -> None:
    """Process with many-to-one mode: all pages merged."""
    console.print("\n[bold cyan]2. Many-to-One Mode[/bold cyan]")
    console.print("  • All pages processed together")
    console.print("  • Attempts to merge into single ID card")
    console.print("  • Best for: Continuous documents, related pages")

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        backend="llm",
        inference="local",
        provider_override="ollama",
        model_override="llama3:8b",
        # Key setting: many-to-one mode
        processing_mode="many-to-one",
        # Enable chunking for many-to-one
        use_chunking=True,
        # Use programmatic merge
    )

    console.print("  • Processing...")
    config.run()
    console.print("  • [green]✓ Complete[/green]")


def main() -> None:
    """Execute processing mode comparison."""
    console.print(
        Panel.fit(
            "[bold blue]Example 05: Processing Modes Comparison[/bold blue]\n"
            "[dim]Compare one-to-one vs many-to-one processing on multi-page document[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Overview:[/yellow]")
    console.print("  This example processes the same multi-page PDF twice:")
    console.print("  1. One-to-One: Each page becomes a separate ID card")
    console.print("  2. Many-to-One: All pages merged into single result")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Ollama running: [cyan]ollama serve[/cyan]")
    console.print("  • Model pulled: [cyan]ollama pull llama3:8b[/cyan]")
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")

    try:
        # Process with both modes
        process_one_to_one()
        process_many_to_one()

        # Comparison table
        console.print("\n[bold]📊 Mode Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Aspect")
        table.add_column("One-to-One")
        table.add_column("Many-to-One")

        table.add_row("Pages", "Processed separately", "Processed together")
        table.add_row("Output", "Multiple models", "Single merged model")
        table.add_row("Chunking", "Not used", "Used for large docs")
        table.add_row("Best For", "Batch documents", "Continuous documents")
        table.add_row("Examples", "Billing documents, forms, ID cards", "Papers, reports, books")

        console.print(table)

        console.print("\n[bold]💡 When to Use Each Mode:[/bold]")
        console.print("\n[cyan]Use One-to-One when:[/cyan]")
        console.print("  • Each page is independent (batch of billing documents)")
        console.print("  • You need separate outputs per page")
        console.print("  • Pages contain different entities")
        console.print("  • Processing forms or structured documents")

        console.print("\n[cyan]Use Many-to-One when:[/cyan]")
        console.print("  • Document spans multiple pages")
        console.print("  • Information flows across pages")
        console.print("  • You want a unified view")
        console.print("  • Processing reports or rheology researchs")

        console.print("\n[bold]📊 Output Locations:[/bold]")
        console.print("  • One-to-One: [cyan]outputs/05_processing_modes/one_to_one/[/cyan]")
        console.print("  • Many-to-One: [cyan]outputs/05_processing_modes/many_to_one/[/cyan]")

        console.print("\n[bold]🔍 Compare Results:[/bold]")
        console.print("  [cyan]# Count nodes in each mode[/cyan]")
        console.print(
            "  [dim]wc -l outputs/05_processing_modes/one_to_one/docling_graph/nodes.csv[/dim]"
        )
        console.print(
            "  [dim]wc -l outputs/05_processing_modes/many_to_one/docling_graph/nodes.csv[/dim]"
        )

        console.print("\n[bold]🎯 Expected Results:[/bold]")
        console.print("  • One-to-One: 3 separate ID card nodes (one per page)")
        console.print("  • Many-to-One: 1 merged ID card node (or 3 if merge fails)")

    except FileNotFoundError:
        console.print(f"\n[red]Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you're running from project root")
        console.print("  • Check sample data exists in docs/examples/data/")
        sys.exit(1)

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "ollama" in error_msg or "connection" in error_msg:
            console.print("  • Start Ollama: [cyan]ollama serve[/cyan]")
            console.print("  • Pull model: [cyan]ollama pull llama3:8b[/cyan]")
            console.print("  • Check Ollama is running: [cyan]ollama list[/cyan]")
        else:
            console.print("  • Install dependencies: [cyan]uv sync[/cyan]")
            console.print("  • Check template is correctly defined")
            console.print("  • Try with a smaller document first")

        sys.exit(1)


if __name__ == "__main__":
    main()
