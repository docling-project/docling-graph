"""
Example 04: Multiple Input Formats

Description:
    Demonstrates how docling-graph handles different input formats: plain text,
    Markdown, and DoclingDocument JSON. Shows format-specific optimizations and
    when to use each format.

Use Cases:
    - Processing documentation and notes (Markdown)
    - Extracting from plain text files
    - Reprocessing with different templates (DoclingDocument)
    - Skipping OCR for text-only inputs

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Sample files: Create test files or use provided examples

Key Concepts:
    - Input Normalization: Automatic format detection
    - OCR Skipping: Text inputs bypass visual processing
    - DoclingDocument: Reprocess without re-running OCR
    - Format-Specific Routing: Optimized pipeline per format

Expected Output:
    - Three separate output directories (one per format)
    - Comparison of processing times
    - Same template applied to different formats

Related Examples:
    - Example 02: PDF processing (requires OCR)
    - Example 03: URL processing
    - Documentation: https://ibm.github.io/docling-graph/fundamentals/pipeline-configuration/input-formats/
"""

import sys
from pathlib import Path
from typing import Any

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from pydantic import BaseModel, Field

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

console = Console()


# Simple template for demonstration
class SimpleDocument(BaseModel):
    """A simple document template for testing different input formats."""

    model_config = {"is_entity": True, "graph_id_fields": ["title"]}

    title: str = Field(
        description="Document title", examples=["Rheology Research", "Technical Report"]
    )
    summary: str = Field(
        description="Brief summary of the document", examples=["This paper discusses..."]
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Main points or findings",
        examples=[["Point 1", "Point 2", "Point 3"]],
    )


def create_sample_text() -> str:
    """Create sample plain text content."""
    return """
Title: Introduction to Knowledge Graphs

Summary: This document provides an overview of knowledge graphs and their applications
in modern AI systems. Knowledge graphs represent information as entities and relationships,
enabling more structured and queryable data representations.

Key Points:
- Knowledge graphs use nodes and edges to represent information
- They enable semantic search and reasoning
- Applications include recommendation systems and question answering
- Graph databases like Neo4j are commonly used for storage
"""


def create_sample_markdown() -> str:
    """Create sample Markdown content."""
    return """# Introduction to Knowledge Graphs

## Summary

This document provides an overview of knowledge graphs and their applications
in modern AI systems. Knowledge graphs represent information as entities and relationships,
enabling more structured and queryable data representations.

## Key Points

- Knowledge graphs use nodes and edges to represent information
- They enable semantic search and reasoning
- Applications include recommendation systems and question answering
- Graph databases like Neo4j are commonly used for storage

## Conclusion

Knowledge graphs are a powerful tool for representing structured information
and enabling advanced AI capabilities.
"""


def process_text_input() -> None:
    """Process plain text input."""
    console.print("\n[bold cyan]1. Processing Plain Text Input[/bold cyan]")

    # Create sample text file
    text_file = Path("outputs/04_input_formats/sample.txt")
    text_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text(create_sample_text())

    config = PipelineConfig(
        source=str(text_file),
        template=SimpleDocument,
        backend="llm",  # Text requires LLM backend
        inference="remote",
        provider_override="mistral",
        model_override="mistral-small-latest",
        processing_mode="many-to-one",
        use_chunking=False,  # Small text, no chunking needed
    )

    console.print("  • Input: Plain text file (.txt)")
    console.print("  • OCR: [green]Skipped[/green] (text-only input)")
    console.print("  • Processing...")

    config.run()
    console.print("  • [green]✓ Complete[/green]")


def process_markdown_input() -> None:
    """Process Markdown input."""
    console.print("\n[bold cyan]2. Processing Markdown Input[/bold cyan]")

    # Create sample markdown file
    md_file = Path("outputs/04_input_formats/sample.md")
    md_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text(create_sample_markdown())

    config = PipelineConfig(
        source=str(md_file),
        template=SimpleDocument,
        backend="llm",  # Markdown requires LLM backend
        inference="remote",
        provider_override="mistral",
        model_override="mistral-small-latest",
        processing_mode="many-to-one",
        use_chunking=False,
    )

    console.print("  • Input: Markdown file (.md)")
    console.print("  • OCR: [green]Skipped[/green] (text-only input)")
    console.print("  • Processing...")

    config.run()
    console.print("  • [green]✓ Complete[/green]")


def process_docling_document() -> None:
    """Process DoclingDocument JSON (reprocessing scenario)."""
    console.print("\n[bold cyan]3. Processing DoclingDocument JSON[/bold cyan]")
    console.print("  • [yellow]Note:[/yellow] This requires a pre-existing DoclingDocument JSON")
    console.print("  • Use case: Reprocess with different template without re-running OCR")
    console.print("  • [dim]Skipping (requires existing DoclingDocument from previous run)[/dim]")


def main() -> None:
    """Execute input format demonstrations."""
    console.print(
        Panel.fit(
            "[bold blue]Example 04: Multiple Input Formats[/bold blue]\n"
            "[dim]Process text, Markdown, and DoclingDocument inputs[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]📋 Overview:[/yellow]")
    console.print("  This example demonstrates three input formats:")
    console.print("  1. Plain Text (.txt) - Simple text files")
    console.print("  2. Markdown (.md) - Formatted documentation")
    console.print("  3. DoclingDocument (.json) - Reprocessing without OCR")

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Mistral API key: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")

    try:
        # Process different formats
        process_text_input()
        process_markdown_input()
        process_docling_document()

        # Summary table
        console.print("\n[bold]📊 Format Comparison:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Format")
        table.add_column("OCR Required")
        table.add_column("Backend Support")
        table.add_column("Best For")

        table.add_row("PDF/Image", "✅ Yes", "LLM + VLM", "Scanned documents, forms")
        table.add_row("Plain Text", "❌ No", "LLM only", "Simple text files")
        table.add_row("Markdown", "❌ No", "LLM only", "Documentation, notes")
        table.add_row("DoclingDocument", "❌ No", "LLM only", "Reprocessing, experimentation")

        console.print(table)

        console.print("\n[bold]💡 Key Takeaways:[/bold]")
        console.print("  • Text/Markdown inputs skip OCR → faster processing")
        console.print("  • DoclingDocument enables template experimentation")
        console.print("  • Same pipeline handles all formats automatically")
        console.print("  • Format detection is automatic based on file extension")

        console.print("\n[bold]📊 Output Locations:[/bold]")
        console.print("  • Text: [cyan]outputs/04_input_formats/text_output/[/cyan]")
        console.print("  • Markdown: [cyan]outputs/04_input_formats/markdown_output/[/cyan]")

        console.print("\n[bold]🔄 Reprocessing Workflow:[/bold]")
        console.print("  1. Process PDF with [cyan]export_docling_json=True[/cyan]")
        console.print("  2. Reprocess JSON with different template (no OCR)")
        console.print("  3. Compare results from different templates")

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "api" in error_msg or "key" in error_msg:
            console.print("  • Set API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
        elif "vlm" in error_msg or "backend" in error_msg:
            console.print("  • Text/Markdown require LLM backend (not VLM)")
            console.print("  • Use [cyan]--backend llm[/cyan] for text inputs")
        else:
            console.print("  • Ensure dependencies: [cyan]uv sync[/cyan]")
            console.print("  • Check file permissions")

        sys.exit(1)


if __name__ == "__main__":
    main()
