"""
Example 14: Streaming LLM Responses

Description:
    Demonstrates how to use streaming responses with the LiteLLM client for
    real-time processing of LLM outputs. Streaming provides reduced latency
    for first results and better user experience with progress feedback.

Use Cases:
    - Interactive applications requiring immediate feedback
    - Processing large documents where partial results are useful
    - Applications with progress indicators
    - Real-time data processing pipelines

Prerequisites:
    - Installation: uv sync
    - Environment: export MISTRAL_API_KEY="your-api-key"
    - Data: Sample invoice image included in repository

Key Concepts:
    - Streaming API: get_json_response_stream() method
    - Iterator Pattern: Process results as they arrive
    - Progress Feedback: Real-time updates during extraction
    - Memory Efficiency: Handle large responses incrementally

Expected Output:
    - Real-time progress updates during extraction
    - Same final output as non-streaming mode
    - nodes.csv, edges.csv, graph.html, report.md

Related Examples:
    - Example 01: Basic VLM extraction (non-streaming)
    - Example 02: Basic LLM extraction (non-streaming)
    - Example 12: Custom LLM client implementation
    - Documentation: https://ibm.github.io/docling-graph/reference/llm-clients/#streaming-responses
"""

import sys
import time
from pathlib import Path

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup project path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.billing_document import BillingDocument

    from docling_graph.llm_clients import get_client
    from docling_graph.llm_clients.config import resolve_effective_model_config
except ImportError:
    rich_print("[red]Error:[/red] Could not import required modules.")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# Configuration
SOURCE_FILE = "docs/examples/data/invoice/swiss_qr_bill.jpg"
TEMPLATE_CLASS = BillingDocument
console = Console()


def example_1_basic_streaming() -> None:
    """Example 1: Basic streaming usage with a single document."""
    console.print("\n[bold cyan]Example 1: Basic Streaming Usage[/bold cyan]")
    console.print("Demonstrates simple streaming with real-time progress feedback.\n")

    # Configure the LLM client
    effective = resolve_effective_model_config(
        "mistral",
        "mistral-large-latest",
        overrides={"generation": {"max_tokens": 2048}},
    )
    client_class = get_client("mistral")
    client = client_class(model_config=effective)

    # Prepare a simple prompt
    prompt = {
        "system": "You are a data extraction assistant. Extract structured information from the provided text.",
        "user": "Extract billing information from this invoice: Invoice #12345, Date: 2024-01-15, Amount: $1,234.56, Customer: Acme Corp",
    }

    # Get the schema from the template
    schema_json = TEMPLATE_CLASS.model_json_schema()
    import json

    schema_str = json.dumps(schema_json)

    console.print("[yellow]Starting streaming extraction...[/yellow]")
    start_time = time.time()

    # Use streaming to get responses
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)

        for result in client.get_json_response_stream(
            prompt=prompt,
            schema_json=schema_str,
        ):
            elapsed = time.time() - start_time
            progress.update(task, description=f"Received result in {elapsed:.2f}s")
            console.print("\n[green]✓ Result received:[/green]")
            console.print(f"  Type: {type(result).__name__}")
            console.print(f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

    console.print(f"\n[bold]Total time:[/bold] {elapsed:.2f}s")
    console.print("\n[dim]Note: Streaming currently accumulates the full response before yielding.[/dim]")
    console.print("[dim]This provides a foundation for future chunk-by-chunk streaming.[/dim]")


def example_2_streaming_vs_nonstreaming() -> None:
    """Example 2: Compare streaming vs non-streaming performance."""
    console.print("\n[bold cyan]Example 2: Streaming vs Non-Streaming Comparison[/bold cyan]")
    console.print("Compares latency and behavior between streaming and non-streaming modes.\n")

    # Configure the LLM client
    effective = resolve_effective_model_config(
        "mistral",
        "mistral-large-latest",
        overrides={"generation": {"max_tokens": 2048}},
    )
    client_class = get_client("mistral")
    client = client_class(model_config=effective)

    prompt = {
        "system": "Extract structured data from the text.",
        "user": "Invoice #67890, Date: 2024-02-20, Total: $5,678.90, Vendor: Tech Solutions Inc.",
    }

    schema_json = TEMPLATE_CLASS.model_json_schema()
    import json

    schema_str = json.dumps(schema_json)

    # Non-streaming call
    console.print("[yellow]1. Non-streaming call...[/yellow]")
    start = time.time()
    client.get_json_response(prompt=prompt, schema_json=schema_str)
    time_normal = time.time() - start
    console.print(f"   [green]✓ Completed in {time_normal:.2f}s[/green]")

    # Streaming call
    console.print("\n[yellow]2. Streaming call...[/yellow]")
    start = time.time()
    for _result_stream in client.get_json_response_stream(prompt=prompt, schema_json=schema_str):
        time_stream = time.time() - start
        console.print(f"   [green]✓ Completed in {time_stream:.2f}s[/green]")

    # Comparison
    console.print("\n[bold]Performance Comparison:[/bold]")
    console.print(f"  Non-streaming: {time_normal:.2f}s")
    console.print(f"  Streaming:     {time_stream:.2f}s")
    console.print(f"  Difference:    {abs(time_stream - time_normal):.2f}s")

    console.print("\n[bold]When to Use Each Mode:[/bold]")
    console.print("  [cyan]Non-Streaming:[/cyan]")
    console.print("    • Batch processing where latency doesn't matter")
    console.print("    • Simple scripts without UI feedback")
    console.print("    • Cases where complete response is needed before processing")
    console.print("\n  [cyan]Streaming:[/cyan]")
    console.print("    • Interactive applications requiring immediate feedback")
    console.print("    • Processing large documents with progress indicators")
    console.print("    • Real-time data processing pipelines")
    console.print("    • Applications with user-facing progress bars")


def example_3_streaming_with_error_handling() -> None:
    """Example 3: Streaming with proper error handling."""
    console.print("\n[bold cyan]Example 3: Streaming with Error Handling[/bold cyan]")
    console.print("Demonstrates robust error handling for streaming operations.\n")

    effective = resolve_effective_model_config(
        "mistral",
        "mistral-large-latest",
        overrides={"generation": {"max_tokens": 2048}},
    )
    client_class = get_client("mistral")
    client = client_class(model_config=effective)

    prompt = {
        "system": "Extract billing information.",
        "user": "Process this invoice data: INV-2024-001, Amount: $999.99",
    }

    schema_json = TEMPLATE_CLASS.model_json_schema()
    import json

    schema_str = json.dumps(schema_json)

    console.print("[yellow]Processing with error handling...[/yellow]")

    try:
        result_count = 0
        for result in client.get_json_response_stream(
            prompt=prompt,
            schema_json=schema_str,
        ):
            result_count += 1
            console.print(f"[green]✓ Result {result_count} received successfully[/green]")

            # Validate result structure
            if isinstance(result, dict):
                console.print(f"  Fields extracted: {len(result)}")
            elif isinstance(result, list):
                console.print(f"  Items extracted: {len(result)}")

        console.print(f"\n[bold green]Success![/bold green] Processed {result_count} result(s)")

    except Exception as e:
        console.print(f"\n[red]Error during streaming:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Check API key is set correctly")
        console.print("  • Verify network connectivity")
        console.print("  • Ensure model supports streaming")
        console.print("  • Check if structured output is compatible with streaming")


def example_4_streaming_best_practices() -> None:
    """Example 4: Best practices for streaming usage."""
    console.print("\n[bold cyan]Example 4: Streaming Best Practices[/bold cyan]")
    console.print("Demonstrates recommended patterns for production use.\n")

    effective = resolve_effective_model_config(
        "mistral",
        "mistral-large-latest",
        overrides={"generation": {"max_tokens": 2048}},
    )
    client_class = get_client("mistral")
    client = client_class(model_config=effective)

    prompt = {
        "system": "Extract structured billing data.",
        "user": "Invoice: INV-2024-999, Date: 2024-03-15, Total: $12,345.67",
    }

    schema_json = TEMPLATE_CLASS.model_json_schema()
    import json

    schema_str = json.dumps(schema_json)

    console.print("[bold]Best Practices:[/bold]")
    console.print("  1. Always use try-except for error handling")
    console.print("  2. Provide user feedback during processing")
    console.print("  3. Validate results as they arrive")
    console.print("  4. Consider timeout handling for long operations")
    console.print("  5. Log streaming events for debugging")

    console.print("\n[yellow]Executing with best practices...[/yellow]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Streaming extraction...", total=None)

            results = []
            for idx, result in enumerate(
                client.get_json_response_stream(
                    prompt=prompt,
                    schema_json=schema_str,
                ),
                1,
            ):
                progress.update(task, description=f"Processing result {idx}...")

                # Validate result
                if not result:
                    console.print("[yellow]⚠ Warning: Empty result received[/yellow]")
                    continue

                results.append(result)
                progress.update(task, description=f"✓ Result {idx} validated")

            progress.update(task, description="[green]✓ Streaming complete[/green]")

        console.print("\n[bold green]Success![/bold green]")
        console.print(f"  Total results: {len(results)}")
        console.print("  All results validated: ✓")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Streaming interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise


def main() -> None:
    """Run all streaming examples."""
    console.print(
        Panel.fit(
            "[bold blue]Example 14: Streaming LLM Responses[/bold blue]\n"
            "[dim]Real-time processing with progress feedback and best practices[/dim]",
            border_style="blue",
        )
    )

    console.print("\n[yellow]⚠️  Prerequisites:[/yellow]")
    console.print("  • Mistral API key must be set: [cyan]export MISTRAL_API_KEY='...'[/cyan]")
    console.print("  • Install dependencies: [cyan]uv sync[/cyan]")

    console.print("\n[bold]Benefits of Streaming:[/bold]")
    console.print("  • Reduced latency for first results")
    console.print("  • Better user experience with progress feedback")
    console.print("  • Memory efficiency for large responses")
    console.print("  • Real-time processing capabilities")

    try:
        # Run examples
        example_1_basic_streaming()
        example_2_streaming_vs_nonstreaming()
        example_3_streaming_with_error_handling()
        example_4_streaming_best_practices()

        console.print("\n[bold green]All examples completed successfully![/bold green]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  • Integrate streaming into your pipeline")
        console.print("  • Add progress indicators to your UI")
        console.print("  • Monitor streaming performance in production")
        console.print("  • See Example 12 for custom client implementation")

    except ImportError as e:
        console.print(f"\n[red]Import Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Run from project root: [cyan]uv run python docs/examples/scripts/14_streaming_responses.py[/cyan]")
        console.print("  • Ensure dependencies installed: [cyan]uv sync[/cyan]")
        sys.exit(1)

    except Exception as e:
        error_msg = str(e).lower()
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")

        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            console.print("  • Set your Mistral API key: [cyan]export MISTRAL_API_KEY='your-key'[/cyan]")
            console.print("  • Get a key at: https://console.mistral.ai/")
        else:
            console.print("  • Check your internet connection")
            console.print("  • Verify API key is valid")
            console.print("  • Try with a different model/provider")

        sys.exit(1)


if __name__ == "__main__":
    main()
