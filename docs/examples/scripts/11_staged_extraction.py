"""
Example 11: Staged Extraction - Multi-Pass LLM Extraction

Description:
    Demonstrates staged extraction (extraction_contract="staged") for many-to-one
    LLM processing. Staged runs: catalog from template → ID pass (skeleton) →
    fill pass (bottom-up) → merge. Useful for complex nested templates.

Use Cases:
    - Nested Pydantic templates with lists and sub-objects
    - Identity-first extraction (IDs from document, then fill)
    - When direct single-pass extraction is inconsistent

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key" (or use local Ollama)
    - Data: Sample PDF (e.g. rheology research)

Key Concepts:
    - extraction_contract="staged": Enables ID pass + fill pass + merge
    - staged_tuning_preset: "standard" or "advanced"
    - Optional overrides: staged_workers, staged_nodes_fill_cap, staged_id_shard_size

Expected Output:
    - Same graph/output structure as direct extraction
    - Processing may take longer due to multiple passes

Related Examples:
    - Example 02: Direct many-to-one (default)
    - Example 05: Processing modes
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/extraction-process/staged-extraction/
"""

import sys
from pathlib import Path

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel

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
    """Execute staged extraction (multi-pass many-to-one LLM)."""
    console.print(
        Panel.fit(
            "[bold blue]Example 11: Staged Extraction[/bold blue]\n"
            "[dim]Multi-pass extraction: ID pass → fill pass → merge[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]LLM[/cyan]")
    console.print("  • Mode: [cyan]many-to-one[/cyan]")
    console.print("  • Extraction contract: [cyan]staged[/cyan]")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print(
        "  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan] (or use Ollama locally)"
    )
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")
    console.print("  • Sample data: [cyan]docs/examples/data/research_paper/rheology.pdf[/cyan]")

    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            provider_override="mistral",
            model_override="mistral-small-latest",
            processing_mode="many-to-one",
            extraction_contract="staged",
            staged_tuning_preset="standard",
            use_chunking=True,
        )

        console.print("\n[yellow]⚙️  Processing (staged passes may take 1-3 minutes)...[/yellow]")
        console.print("  • Catalog from template")
        console.print("  • ID pass: build node skeleton")
        console.print("  • Fill pass: fill content per path (bottom-up)")
        console.print("  • Merge into root model")
        console.print("  • Build knowledge graph")

        context = run_pipeline(config)

        console.print("\n[green]✓ Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"\n[bold]Extracted:[/bold] [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        console.print("\n[bold]💡 Staged vs direct:[/bold]")
        console.print("  • Direct (default): single extraction pass, then programmatic merge")
        console.print("  • Staged: ID pass first (skeleton), then fill pass per path, then merge")
        console.print("  • Use staged for complex nested templates when direct is inconsistent")

    except FileNotFoundError:
        console.print(f"\n[red]Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("  • Run from project root and ensure sample data exists.")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        if "api" in error_msg or "key" in error_msg:
            console.print("  • Set Mistral API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
            console.print(
                "  • Or switch to local: [cyan]inference='local', provider_override='ollama'[/cyan]"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
