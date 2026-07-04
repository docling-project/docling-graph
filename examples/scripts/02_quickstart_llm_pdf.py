"""
Example 02: Quickstart - LLM Extraction from PDF

Description:
    Basic LLM extraction from a multi-page rheology research PDF using a remote API.
    Demonstrates the standard workflow for text-heavy documents with automatic chunking.

Use Cases:
    - Rheology researchs and academic documents
    - Technical reports and whitepapers
    - Multi-page business documents
    - Any text-heavy PDF content

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Data: Sample rheology research included in repository

Key Concepts:
    - LLM Backend: Processes text extracted from PDFs
    - Many-to-One Mode: All pages merged into single output
    - Chunking: Automatically splits large documents for LLM context limits
    - Remote Inference: Uses Mistral API for extraction
    - Programmatic Merge: Combines chunk results without additional LLM call

Expected Output:
    - nodes.csv: Extracted research data (authors, experiments, results)
    - edges.csv: Relationships between research entities
    - graph.html: Interactive knowledge graph visualization
    - document.md: Markdown version of the PDF
    - report.md: Extraction statistics and summary

Related Examples:
    - Example 01: VLM extraction from images
    - Example 07: Local LLM inference
    - Example 08: Advanced chunking strategies
    - Documentation: https://ibm.github.io/docling-graph/usage/examples/research-paper/
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

# Configuration
SOURCE_FILE = "docs/examples/data/research_paper/rheology.pdf"
TEMPLATE_CLASS = ScholarlyRheologyPaper
console = Console()


def main() -> None:
    """Execute LLM extraction from rheology research PDF."""
    console.print(
        Panel.fit(
            "[bold blue]Example 02: Quickstart - LLM from PDF[/bold blue]\n"
            "[dim]Extract structured data from a rheology research using Large Language Model[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]LLM (Large Language Model)[/cyan]")
    console.print("  • Provider: [cyan]Mistral AI[/cyan]")
    console.print("  • Mode: [cyan]many-to-one[/cyan]")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Mistral API key must be set: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")

    try:
        # Configure the pipeline
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            # LLM backend for text-based extraction
            backend="llm",
            # Remote inference using API
            inference="remote",
            # Use Mistral AI provider
            provider_override="mistral",
            # Use a capable model for complex extraction
            model_override="mistral-large-latest",
            # Many-to-one: merge all pages into single result
            processing_mode="many-to-one",
            # extraction_contract="direct" (default); use "dense" for richer chunk-aware extraction
            use_chunking=True,
        )

        # Execute the pipeline
        console.print("\n[yellow]⚙️  Processing (this may take 1-2 minutes)...[/yellow]")
        console.print("  • Converting PDF to markdown")
        console.print("  • Chunking document for LLM context")
        console.print("  • Extracting data from each chunk")
        console.print("  • Merging results programmatically")
        console.print("  • Building knowledge graph")

        context = run_pipeline(config)

        # Success message
        console.print("\n[green]✓ Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"\n[bold]Extracted:[/bold] [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        console.print("\n[bold]💡 What Happened:[/bold]")
        console.print("  • PDF converted to markdown using Docling")
        console.print("  • Document split into chunks respecting context limits")
        console.print("  • Each chunk processed by Mistral LLM")
        console.print("  • Results merged programmatically (no LLM consolidation)")
        console.print("  • Knowledge graph built from extracted entities")

        console.print("\n[bold]🎯 Key Differences from Example 01:[/bold]")
        console.print("  • LLM vs VLM: Text-based vs vision-based extraction")
        console.print("  • Remote vs Local: API call vs local model")
        console.print("  • Many-to-one vs One-to-one: Merged vs separate outputs")
        console.print("  • Chunking: Enabled for large documents")

    except FileNotFoundError:
        console.print(f"\n[red]Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you're running from the project root directory")
        console.print("  • Check that the sample data exists in docs/examples/data/")
        sys.exit(1)

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            console.print(
                "  • Set your Mistral API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]"
            )
            console.print("  • Get a key at: https://console.mistral.ai/")
            console.print("  • Or use local inference: see Example 07")
        else:
            console.print("  • Ensure dependencies installed: [cyan]uv sync[/cyan]")
            console.print("  • Check your internet connection")
            console.print("  • Verify the template class is correctly defined")
            console.print("  • Try with a smaller document first")

        sys.exit(1)


if __name__ == "__main__":
    main()
