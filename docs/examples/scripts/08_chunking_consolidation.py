"""
Example 08: Advanced Chunking and Consolidation Strategies

Description:
    Demonstrates different chunking and consolidation approaches for large documents.
    Compares programmatic merge vs LLM consolidation, and shows chunking configuration.

Use Cases:
    - Large documents exceeding LLM context limits
    - Optimizing extraction quality vs speed
    - Understanding consolidation trade-offs
    - Fine-tuning chunking parameters

Prerequisites:
    - Installation: uv sync --extra remote
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Data: Multi-page rheology research

Key Concepts:
    - Chunking: Splitting documents for LLM context
    - Programmatic Merge: Fast, rule-based consolidation
    - LLM Consolidation: Intelligent, LLM-based merge
    - Trade-offs: Speed vs quality

Expected Output:
    - Two outputs for comparison
    - Processing time differences
    - Quality comparison

Related Examples:
    - Example 02: Basic LLM extraction
    - Example 05: Processing modes
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/extraction-process/chunking-strategies/
"""

import sys
import time
from pathlib import Path

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.rheology_research import Research

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = Research

console = Console()


def process_programmatic_merge() -> float:
    """Process with programmatic merge (fast)."""
    console.print("\n[bold cyan]1. Programmatic Merge (Fast)[/bold cyan]")
    console.print("  • Rule-based consolidation")
    console.print("  • No additional LLM calls")
    console.print("  • Faster but may miss semantic duplicates")

    start_time = time.time()

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        output_dir="outputs/08_chunking_consolidation/programmatic",
        backend="llm",
        inference="remote",
        provider_override="mistral",
        model_override="mistral-small-latest",
        processing_mode="many-to-one",
        use_chunking=True,
        llm_consolidation=False,  # Programmatic merge
    )

    console.print("  • Processing...")
    config.run()

    elapsed = time.time() - start_time
    console.print(f"  • [green]✓ Complete[/green] in {elapsed:.1f}s")
    return elapsed


def process_llm_consolidation() -> float:
    """Process with LLM consolidation (intelligent)."""
    console.print("\n[bold cyan]2. LLM Consolidation (Intelligent)[/bold cyan]")
    console.print("  • LLM-based intelligent merge")
    console.print("  • Additional LLM call for consolidation")
    console.print("  • Slower but better semantic understanding")

    start_time = time.time()

    config = PipelineConfig(
        source=SOURCE_FILE,
        template=TEMPLATE_CLASS,
        output_dir="outputs/08_chunking_consolidation/llm_consolidation",
        backend="llm",
        inference="remote",
        provider_override="mistral",
        model_override="mistral-small-latest",
        processing_mode="many-to-one",
        use_chunking=True,
        llm_consolidation=True,  # LLM consolidation
    )

    console.print("  • Processing...")
    config.run()

    elapsed = time.time() - start_time
    console.print(f"  • [green]✓ Complete[/green] in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    """Execute chunking and consolidation comparison."""
    console.print(
        Panel.fit(
            "[bold blue]Example 08: Chunking & Consolidation[/bold blue]\n"
            "[dim]Compare programmatic merge vs LLM consolidation strategies[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Overview:[/yellow]")
    console.print("  This example compares two consolidation strategies:")
    console.print("  1. Programmatic Merge: Fast, rule-based")
    console.print("  2. LLM Consolidation: Intelligent, LLM-based")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install remote extras: [cyan]uv sync --extra remote[/cyan]")

    try:
        # Process with both strategies
        time_programmatic = process_programmatic_merge()
        time_llm = process_llm_consolidation()

        # Comparison table
        console.print("\n[bold]📊 Strategy Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Aspect")
        table.add_column("Programmatic Merge")
        table.add_column("LLM Consolidation")

        table.add_row("Speed", f"{time_programmatic:.1f}s", f"{time_llm:.1f}s")
        table.add_row("API Calls", "N chunks", "N chunks + 1")
        table.add_row("Cost", "Lower", "Higher")
        table.add_row("Quality", "Good", "Better")
        table.add_row("Duplicates", "May miss semantic", "Handles semantic")

        console.print(table)

        console.print("\n[bold]💡 How Each Works:[/bold]")
        console.print("\n[cyan]Programmatic Merge:[/cyan]")
        console.print("  1. Extract from each chunk independently")
        console.print("  2. Merge results using rules:")
        console.print("     • Lists: Concatenate and deduplicate")
        console.print("     • Scalars: First non-null value")
        console.print("     • Objects: Recursive merge")
        console.print("  3. Fast but may miss semantic duplicates")

        console.print("\n[cyan]LLM Consolidation:[/cyan]")
        console.print("  1. Extract from each chunk independently")
        console.print("  2. Merge results programmatically (initial)")
        console.print("  3. Send merged result to LLM for refinement")
        console.print("  4. LLM resolves semantic duplicates")
        console.print("  5. Slower but higher quality")

        console.print("\n[bold]🎯 When to Use Each:[/bold]")
        console.print("\n[cyan]Use Programmatic Merge when:[/cyan]")
        console.print("  • Speed is priority")
        console.print("  • Cost is a concern")
        console.print("  • Documents have clear structure")
        console.print("  • Minimal semantic duplicates expected")

        console.print("\n[cyan]Use LLM Consolidation when:[/cyan]")
        console.print("  • Quality is priority")
        console.print("  • Complex semantic relationships")
        console.print("  • Handling ambiguous duplicates")
        console.print("  • Final production extraction")

        console.print("\n[bold]📊 Output Locations:[/bold]")
        console.print(
            "  • Programmatic: [cyan]outputs/08_chunking_consolidation/programmatic/[/cyan]"
        )
        console.print("  • LLM: [cyan]outputs/08_chunking_consolidation/llm_consolidation/[/cyan]")

        console.print("\n[bold]🔍 Compare Results:[/bold]")
        console.print("  [cyan]# Compare node counts[/cyan]")
        console.print(
            "  [dim]wc -l outputs/08_chunking_consolidation/*/docling_graph/nodes.csv[/dim]"
        )

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "api" in error_msg or "key" in error_msg:
            console.print("  • Set API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
            console.print("  • Get key at: https://console.mistral.ai/")
        else:
            console.print("  • Install dependencies: [cyan]uv sync --extra remote[/cyan]")
            console.print("  • Check internet connection")

        sys.exit(1)


if __name__ == "__main__":
    main()
