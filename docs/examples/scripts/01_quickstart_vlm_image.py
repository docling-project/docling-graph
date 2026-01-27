"""
Example 01: Quickstart - VLM Extraction from Image

Description:
    The simplest possible example demonstrating VLM (Vision-Language Model) extraction
    from a single billing document image. This is the "Hello World" of docling-graph.

Use Cases:
    - Extracting data from scanned billing documents
    - Processing ID cards, badges, or forms
    - Single-page structured documents
    - Image files (JPG, PNG)

Prerequisites:
    - Installation: uv sync
    - Data: Sample billing document from Wikimedia Commons
    - No API keys required (uses local VLM)

Key Concepts:
    - VLM Backend: Processes images directly without text conversion
    - One-to-One Mode: Each image becomes one extracted model
    - Vision Pipeline: Uses Docling's vision capabilities for layout understanding
    - Local Inference: Runs entirely on your machine

Expected Output:
    - nodes.csv: Extracted billing document data in tabular format
    - edges.csv: Relationships between entities
    - graph.html: Interactive visualization
    - graph.json: Complete graph structure
    - report.md: Summary statistics

Related Examples:
    - Example 02: LLM extraction from PDF
    - Example 04: Multiple input formats
    - Documentation: https://ibm.github.io/docling-graph/usage/examples/billing-document-extraction/
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

# Configuration
SOURCE_FILE = "https://upload.wikimedia.org/wikipedia/commons/9/9f/Swiss_QR-Bill_example.jpg"
TEMPLATE_CLASS = BillingDocument
console = Console()


def main() -> None:
    """Execute VLM extraction from billing document image."""
    console.print(
        Panel.fit(
            "[bold blue]Example 01: Quickstart - VLM from Image[/bold blue]\n"
            "[dim]Extract structured data from a billing document image using Vision-Language Model[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Configuration:[/yellow]")
    console.print(f"  • Source: [cyan]{SOURCE_FILE}[/cyan]")
    console.print(f"  • Template: [cyan]{TEMPLATE_CLASS.__name__}[/cyan]")
    console.print("  • Backend: [cyan]VLM (Vision-Language Model)[/cyan]")
    console.print("  • Mode: [cyan]one-to-one[/cyan]")

    try:
        # Configure the pipeline
        config = PipelineConfig(
            source=SOURCE_FILE,
            template=TEMPLATE_CLASS,
            # VLM backend processes images directly
            backend="vlm",
            # VLM only supports local inference
            inference="local",
            # One-to-one: each image becomes one model instance
            processing_mode="one-to-one",
            # Vision pipeline required for VLM
            docling_config="vision",
        )

        # Execute the pipeline
        console.print("\n[yellow]⚙️  Processing...[/yellow]")
        context = run_pipeline(config)

        # Success message
        console.print("\n[green]✓ Success![/green]")
        graph = context.knowledge_graph
        console.print(
            f"\n[bold]Extracted:[/bold] [cyan]{graph.number_of_nodes()} nodes[/cyan] "
            f"and [cyan]{graph.number_of_edges()} edges[/cyan]"
        )

        console.print("\n[bold]💡 What Happened:[/bold]")
        console.print("  • Image was processed by Docling's vision model")
        console.print("  • Billing document structure extracted using VLM")
        console.print("  • Data validated against BillingDocument template")
        console.print("  • Knowledge graph constructed with entities and relationships")
        console.print("  • Multiple export formats generated")

    except FileNotFoundError:
        console.print(f"\n[red]Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you're running from the project root directory")
        console.print("  • Check that the sample data exists in docs/examples/data/")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure dependencies are installed: [cyan]uv sync[/cyan]")
        console.print("  • Check that you have sufficient disk space")
        console.print("  • Verify the template class is correctly defined")
        console.print(
            "  • For GPU issues, see: https://ibm.github.io/docling-graph/fundamentals/installation/gpu-setup/"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
