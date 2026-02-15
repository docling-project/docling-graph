"""
Example 13: Delta Extraction - Chunk-Based Graph Extraction

Description:
    Demonstrates delta extraction (extraction_contract="delta") for many-to-one
    LLM processing. Delta runs: chunk → token-bounded batches → per-batch LLM
    (flat graph IR: nodes + relationships) → IR normalize → merge → projection.
    Useful for long documents and graph-first extraction.

Use Cases:
    - Long documents with token-bounded batching
    - Prefer graph-first representation (nodes with path, ids, parent) then project to template
    - Optional post-merge resolvers (fuzzy/semantic) for near-duplicate entities

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key" (or use local Ollama)
    - Data: Sample PDF (e.g. billing or rheology)
    - Chunking must be enabled (default for many-to-one)

Key Concepts:
    - extraction_contract="delta": Enables chunk → batch → graph merge → projection
    - llm_batch_token_size: max input tokens per LLM batch (default 2048)
    - parallel_workers: parallel delta batch calls
    - Delta normalizer and optional resolvers

Expected Output:
    - Same graph/output structure as direct extraction
    - Processing depends on document length and batch count

Related Examples:
    - Example 11: Staged extraction (ID pass → fill pass)
    - Example 02: Direct many-to-one (default)
    - Documentation: Delta Extraction (fundamentals/extraction-process/delta-extraction.md)
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
    from examples.templates.billing_document import BillingDocument

    from docling_graph import PipelineConfig, run_pipeline
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

SOURCE_FILE = "docs/examples/data/billing/sample_invoice.pdf"
TEMPLATE_CLASS = BillingDocument
console = Console()


def main() -> None:
    """Execute delta extraction (chunk → batch → graph merge → projection)."""
    console.print(
        Panel.fit(
            "[bold blue]Example 13: Delta Extraction[/bold blue]\n"
            "[dim]Chunk-based graph extraction: batches → flat graph IR → merge → projection[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]LLM[/cyan]")
    console.print("  • Mode: [cyan]many-to-one[/cyan]")
    console.print("  • Extraction contract: [cyan]delta[/cyan]")
    console.print("  • Chunking: [cyan]enabled[/cyan] (required for delta)")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print(
        "  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan] (or use Ollama locally)"
    )
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")
    console.print("  • Sample data: [cyan]docs/examples/data/billing/sample_invoice.pdf[/cyan]")

    try:
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            backend="llm",
            inference="remote",
            provider_override="mistral",
            model_override="mistral-small-latest",
            processing_mode="many-to-one",
            extraction_contract="delta",
            use_chunking=True,
            llm_batch_token_size=2048,
            parallel_workers=1,
        )

        console.print(
            "\n[yellow]⚙️  Processing (delta: chunk → batch → merge → project)...[/yellow]"
        )
        console.print("  • Chunk document and plan token-bounded batches")
        console.print("  • Per-batch LLM: extract flat graph (nodes + relationships)")
        console.print("  • IR normalize → merge by identity → project to template")
        console.print("  • Build knowledge graph")

        context = run_pipeline(config)

        console.print("\n[green]✓ Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"\n[bold]Extracted:[/bold] [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        console.print("\n[bold]💡 Delta vs direct vs staged:[/bold]")
        console.print("  • Direct: single pass per chunk, then programmatic merge")
        console.print(
            "  • Delta: token-bounded batches → graph IR → normalize → merge → projection"
        )
        console.print("  • Staged: ID pass → fill pass → merge (no chunk batching)")
        console.print("  • Use delta for long documents when you want batched graph extraction")

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
