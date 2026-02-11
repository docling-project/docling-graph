"""
Example 08: Chunking and Programmatic Consolidation

Description:
    Demonstrates chunking and programmatic merge for large documents
    that exceed LLM context limits.

Use Cases:
    - Large documents exceeding LLM context limits
    - Optimizing extraction with configurable chunking
    - Understanding programmatic merge behavior

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Data: Multi-page rheology research

Key Concepts:
    - Chunking: Splitting documents for LLM context
    - Programmatic Merge: Rule-based consolidation of chunk results

Expected Output:
    - Single output with consolidated graph
    - Processing time

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

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.rheology_research import ScholarlyRheologyPaper

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper

console = Console()


def main() -> None:
    """Execute chunking with programmatic consolidation."""
    console.print(
        Panel.fit(
            "[bold blue]Example 08: Chunking & Consolidation[/bold blue]\n"
            "[dim]Process a large document with chunking and programmatic merge[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Overview:[/yellow]")
    console.print("  This example processes a multi-page document with:")
    console.print("  • Chunking: Document split to fit LLM context")
    console.print("  • Programmatic merge: Chunk results consolidated by rules")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")

    try:
        console.print("\n[bold cyan]Processing with chunking...[/bold cyan]")
        start_time = time.time()

        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            provider_override="mistral",
            model_override="mistral-small-latest",
            processing_mode="many-to-one",
            use_chunking=True,
            output_dir="outputs/08_chunking_consolidation",
        )

        config.run()
        elapsed = time.time() - start_time
        console.print(f"  • [green]✓ Complete[/green] in {elapsed:.1f}s")

        console.print("\n[bold]📊 Output:[/bold]")
        console.print("  • [cyan]outputs/08_chunking_consolidation/[/cyan]")

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "api" in error_msg or "key" in error_msg:
            console.print("  • Set API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
            console.print("  • Get key at: https://console.mistral.ai/")
        else:
            console.print("  • Install dependencies: [cyan]uv sync[/cyan]")
            console.print("  • Check internet connection")

        sys.exit(1)


if __name__ == "__main__":
    main()
