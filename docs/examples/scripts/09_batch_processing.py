"""
Example 09: Batch Processing Multiple Documents

Description:
    Demonstrates how to process multiple documents efficiently in a batch.
    Shows error handling, progress tracking, and result aggregation.

Use Cases:
    - Processing document collections
    - Bulk data extraction
    - Automated pipelines
    - Production workflows

Prerequisites:
    - Installation: uv sync --extra all
    - Multiple sample documents
    - Optional: API keys for remote inference

Key Concepts:
    - Batch Processing: Multiple documents in sequence
    - Error Handling: Continue on failures
    - Progress Tracking: Monitor batch progress
    - Result Aggregation: Combine outputs

Expected Output:
    - Separate output directory per document
    - Batch processing summary
    - Error log if failures occur

Related Examples:
    - Example 01-03: Single document processing
    - Documentation: https://ibm.github.io/docling-graph/usage/api/batch-processing/
"""

import sys
from pathlib import Path
from typing import List, Tuple

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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

console = Console()


def get_sample_documents() -> List[Tuple[str, type]]:
    """Get list of sample documents to process."""
    return [
        ("docs/examples/data/invoice/sample_invoice.jpg", Invoice),
        # Add more documents here as needed
        # ("docs/examples/data/invoice/another_invoice.pdf", Invoice),
    ]


def process_document(source: str, template: type, output_base: str) -> Tuple[bool, str]:
    """
    Process a single document.

    Returns:
        Tuple of (success, message)
    """
    try:
        # Create output directory based on source filename
        source_path = Path(source)
        output_dir = f"{output_base}/{source_path.stem}"

        config = PipelineConfig(
            source=source,
            template=template,
            output_dir=output_dir,
            backend="vlm",  # Use VLM for images
            inference="local",
            processing_mode="one-to-one",
            docling_config="vision",
        )

        config.run()
        return True, f"Success: {source_path.name}"

    except FileNotFoundError:
        return False, f"File not found: {source}"
    except Exception as e:
        return False, f"Error processing {source}: {e!s}"


def main() -> None:
    """Execute batch processing."""
    console.print(
        Panel.fit(
            "[bold blue]Example 09: Batch Processing[/bold blue]\n"
            "[dim]Process multiple documents efficiently with error handling[/dim]",
            border_style="blue",
        )
    )

    # Get documents to process
    documents = get_sample_documents()
    output_base = "outputs/09_batch_processing"

    console.print("\n[yellow]📋 Batch Configuration:[/yellow]")
    console.print(f"  • Documents to process: [cyan]{len(documents)}[/cyan]")
    console.print(f"  • Output directory: [cyan]{output_base}[/cyan]")
    console.print("  • Backend: [cyan]VLM (local)[/cyan]")

    console.print("\n[yellow]⚙️  Processing batch...[/yellow]")

    # Track results
    results = []
    successful = 0
    failed = 0

    # Process with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing documents...", total=len(documents))

        for source, template in documents:
            source_name = Path(source).name
            progress.update(task, description=f"[cyan]Processing {source_name}...")

            success, message = process_document(source, template, output_base)
            results.append((source_name, success, message))

            if success:
                successful += 1
                console.print(f"  [green]✓[/green] {source_name}")
            else:
                failed += 1
                console.print(f"  [red]✗[/red] {source_name}: {message}")

            progress.advance(task)

    # Summary
    console.print("\n[bold]📊 Batch Processing Summary:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Count")

    table.add_row("Total Documents", str(len(documents)))
    table.add_row("Successful", f"[green]{successful}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    table.add_row(
        "Success Rate", f"{(successful/len(documents)*100):.1f}%" if documents else "N/A"
    )

    console.print(table)

    # Detailed results
    if failed > 0:
        console.print("\n[bold red]Failed Documents:[/bold red]")
        for name, success, message in results:
            if not success:
                console.print(f"  • {name}: {message}")

    console.print("\n[bold]💡 Batch Processing Tips:[/bold]")
    console.print("  • Process similar documents together (same template)")
    console.print("  • Use error handling to continue on failures")
    console.print("  • Monitor progress for long-running batches")
    console.print("  • Consider parallel processing for large batches")
    console.print("  • Log errors for debugging")

    console.print("\n[bold]🔧 Advanced Batch Processing:[/bold]")
    console.print("  • Use Python's multiprocessing for parallelization")
    console.print("  • Implement retry logic for transient failures")
    console.print("  • Add rate limiting for API-based processing")
    console.print("  • Save intermediate results for resumability")

    console.print("\n[bold]📊 Output Locations:[/bold]")
    console.print(f"  • Base directory: [cyan]{output_base}/[/cyan]")
    console.print("  • Each document in separate subdirectory")
    console.print(f"  • View all: [cyan]ls {output_base}/[/cyan]")

    # Exit with error code if any failed
    if failed > 0:
        console.print(f"\n[yellow]⚠️  {failed} document(s) failed processing[/yellow]")
        sys.exit(1)
    else:
        console.print("\n[green]✓ All documents processed successfully![/green]")


if __name__ == "__main__":
    main()

# Made with Bob
