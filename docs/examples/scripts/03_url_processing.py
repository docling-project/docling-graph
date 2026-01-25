"""
Example 03: URL Processing - Extract from Remote Documents

Description:
    Download and process a research paper directly from a URL (arXiv).
    Demonstrates how docling-graph can fetch and process remote documents
    in a single pipeline execution.

Use Cases:
    - Processing papers from arXiv, PubMed, or other repositories
    - Extracting data from web-hosted PDFs
    - Automated document ingestion from URLs
    - Building knowledge bases from online sources

Prerequisites:
    - Installation: uv sync --extra remote
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Internet connection required for URL download

Key Concepts:
    - URL Input: Automatic download and processing
    - Remote Inference: Uses API for extraction
    - Many-to-One: Merges multi-page document
    - Chunking: Handles large documents automatically
    - Caching: Downloaded files cached in output directory

Expected Output:
    - nodes.csv: Extracted research entities
    - edges.csv: Relationships between entities
    - graph.html: Interactive visualization
    - document.pdf: Downloaded PDF (cached)
    - document.md: Markdown conversion
    - report.md: Processing statistics

Related Examples:
    - Example 02: LLM extraction from local PDF
    - Example 04: Other input formats (text, markdown)
    - Example 08: Advanced chunking strategies
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/pipeline-configuration/input-formats/#urls
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
    from examples.templates.rheology_research import Research

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# Configuration
SOURCE_URL = "https://arxiv.org/pdf/2207.02720"  # Example arXiv paper
TEMPLATE_CLASS = Research
OUTPUT_DIR = "outputs/03_url_processing"

console = Console()


def main() -> None:
    """Execute LLM extraction from URL-hosted PDF."""
    console.print(
        Panel.fit(
            "[bold blue]Example 03: URL Processing[/bold blue]\n"
            "[dim]Download and extract structured data from a remote research paper[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_URL}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]LLM (Large Language Model)[/cyan]")
    console.print("  • Provider: [cyan]Mistral AI[/cyan]")
    console.print("  • Mode: [cyan]many-to-one[/cyan]")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Internet connection required")
    console.print("  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install remote extras: [cyan]uv sync --extra remote[/cyan]")

    try:
        # Configure the pipeline
        config = PipelineConfig(
            source=SOURCE_URL,
            template=TEMPLATE_CLASS,
            output_dir=OUTPUT_DIR,
            # LLM backend for text extraction
            backend="llm",
            # Remote inference via API
            inference="remote",
            # Use Mistral provider
            provider_override="mistral",
            # Use powerful model for complex extraction
            model_override="mistral-large-latest",
            # Many-to-one: merge all pages
            processing_mode="many-to-one",
            # Enable chunking for large documents
            use_chunking=True,
            # Programmatic merge (no extra LLM call)
            llm_consolidation=False,
        )

        # Execute the pipeline
        console.print("\n[yellow]⚙️  Processing (this may take 2-3 minutes)...[/yellow]")
        console.print("  • Downloading PDF from URL")
        console.print("  • Converting to markdown")
        console.print("  • Chunking document")
        console.print("  • Extracting with LLM")
        console.print("  • Building knowledge graph")

        config.run()

        # Success message
        console.print("\n[green]✓ Success![/green]")
        console.print(f"\n[bold]Output Location:[/bold] [cyan]{OUTPUT_DIR}[/cyan]")

        console.print("\n[bold]📊 Next Steps:[/bold]")
        console.print(
            f"  1. View interactive graph: [cyan]uv run docling-graph inspect {OUTPUT_DIR}[/cyan]"
        )
        console.print(f"  2. Check extracted data: [cyan]cat {OUTPUT_DIR}/docling_graph/nodes.csv[/cyan]")
        console.print(f"  3. View cached PDF: [cyan]ls {OUTPUT_DIR}/docling/[/cyan]")
        console.print(f"  4. Read markdown: [cyan]cat {OUTPUT_DIR}/docling/document.md[/cyan]")

        console.print("\n[bold]💡 What Happened:[/bold]")
        console.print("  • PDF downloaded from arXiv and cached locally")
        console.print("  • Document converted to markdown")
        console.print("  • Content chunked for LLM processing")
        console.print("  • Research entities extracted (authors, experiments, results)")
        console.print("  • Knowledge graph constructed with relationships")

        console.print("\n[bold]🎯 URL Processing Benefits:[/bold]")
        console.print("  • No manual download required")
        console.print("  • Downloaded files cached for reprocessing")
        console.print("  • Same pipeline works for local and remote files")
        console.print("  • Supports arXiv, PubMed, and direct PDF URLs")

        console.print("\n[bold]🔄 Reprocessing Tip:[/bold]")
        console.print(
            "  To reprocess without re-downloading, use the cached PDF:"
        )
        console.print(
            f"  [cyan]uv run docling-graph convert {OUTPUT_DIR}/docling/*.pdf ...[/cyan]"
        )

    except FileNotFoundError:
        console.print(f"\n[red]✗ Error:[/red] Could not download from URL: {SOURCE_URL}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Check your internet connection")
        console.print("  • Verify the URL is accessible")
        console.print("  • Try downloading manually first")
        console.print("  • Check for firewall or proxy issues")
        sys.exit(1)

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "timeout" in error_msg or "connection" in error_msg:
            console.print("  • Check your internet connection")
            console.print("  • Try again (network might be slow)")
            console.print("  • Download PDF manually and use local path")
        elif "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            console.print("  • Set your Mistral API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
            console.print("  • Get a key at: https://console.mistral.ai/")
        else:
            console.print("  • Ensure dependencies: [cyan]uv sync --extra remote[/cyan]")
            console.print("  • Check URL is a valid PDF")
            console.print("  • Try with a smaller document first")

        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
