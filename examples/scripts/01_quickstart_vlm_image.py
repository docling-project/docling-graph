"""
Example 01: Quickstart - VLM Extraction from Image

Description:
    The simplest possible example demonstrating VLM (Vision-Language Model) extraction
    from a single invoice image. This is the "Hello World" of docling-graph.

Use Cases:
    - Extracting data from scanned invoices
    - Processing ID cards, badges, or forms
    - Single-page structured documents
    - Image files (JPG, PNG)

Prerequisites:
    - Installation: uv sync --extra all
    - Data: Sample invoice image included in repository
    - No API keys required (uses local VLM)

Key Concepts:
    - VLM Backend: Processes images directly without text conversion
    - One-to-One Mode: Each image becomes one extracted model
    - Vision Pipeline: Uses Docling's vision capabilities for layout understanding
    - Local Inference: Runs entirely on your machine

Expected Output:
    - nodes.csv: Extracted invoice data in tabular format
    - edges.csv: Relationships between entities
    - graph.html: Interactive visualization
    - graph.json: Complete graph structure
    - report.md: Summary statistics

Related Examples:
    - Example 02: LLM extraction from PDF
    - Example 04: Multiple input formats
    - Documentation: https://ibm.github.io/docling-graph/usage/examples/invoice-extraction/
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
    from examples.templates.invoice import Invoice

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# Configuration
SOURCE_FILE = "docs/examples/data/invoice/sample_invoice.jpg"
TEMPLATE_CLASS = Invoice
OUTPUT_DIR = "outputs/01_quickstart_vlm_image"

console = Console()


def main() -> None:
    """Execute VLM extraction from invoice image."""
    console.print(
        Panel.fit(
            "[bold blue]Example 01: Quickstart - VLM from Image[/bold blue]\n"
            "[dim]Extract structured data from an invoice image using Vision-Language Model[/dim]",
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
            output_dir=OUTPUT_DIR,
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
        config.run()

        # Success message
        console.print("\n[green]✓ Success![/green]")
        console.print(f"\n[bold]Output Location:[/bold] [cyan]{OUTPUT_DIR}[/cyan]")

        console.print("\n[bold]📊 Next Steps:[/bold]")
        console.print(
            f"  1. View interactive graph: [cyan]uv run docling-graph inspect {OUTPUT_DIR}[/cyan]"
        )
        console.print(
            f"  2. Check extracted data: [cyan]cat {OUTPUT_DIR}/docling_graph/nodes.csv[/cyan]"
        )
        console.print(
            f"  3. View relationships: [cyan]cat {OUTPUT_DIR}/docling_graph/edges.csv[/cyan]"
        )
        console.print(
            f"  4. Read summary report: [cyan]cat {OUTPUT_DIR}/docling_graph/report.md[/cyan]"
        )

        console.print("\n[bold]💡 What Happened:[/bold]")
        console.print("  • Image was processed by Docling's vision model")
        console.print("  • Invoice structure extracted using VLM")
        console.print("  • Data validated against Invoice template")
        console.print("  • Knowledge graph constructed with entities and relationships")
        console.print("  • Multiple export formats generated")

    except FileNotFoundError:
        console.print(f"\n[red]✗ Error:[/red] Source file not found: {SOURCE_FILE}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure you're running from the project root directory")
        console.print("  • Check that the sample data exists in docs/examples/data/")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Ensure all dependencies are installed: [cyan]uv sync --extra all[/cyan]")
        console.print("  • Check that you have sufficient disk space")
        console.print("  • Verify the template class is correctly defined")
        console.print(
            "  • For GPU issues, see: https://ibm.github.io/docling-graph/fundamentals/installation/gpu-setup/"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
