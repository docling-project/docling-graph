"""
Example 09: Exporting to Cypher for Neo4j
"""

import sys
from pathlib import Path

from rich import print as rich_print

# --- 1. Add project root to sys.path ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from examples.templates.invoice import Invoice

    from docling_graph import PipelineConfig
except ImportError:
    rich_print("[red]Error:[/red] Could not import modules. ")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# --- 2. Define Paths (relative to project root) ---
SOURCE_FILE = "docs/examples/data/invoice/sample_invoice.jpg"
TEMPLATE_CLASS = Invoice
OUTPUT_DIR = "outputs/09_export_to_cypher"

rich_print("[blue]--- Running Example 09: Export to Cypher ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,
    # Use simple VLM for this example
    backend="vlm",
    inference="local",
    processing_mode="one-to-one",
    docling_config="vision",
    # --- Key Setting ---
    export_format="cypher",
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    cypher_path = Path(OUTPUT_DIR) / f"{Path(SOURCE_FILE).stem}_graph.cypher"
    rich_print(f"Find your Cypher script at: {cypher_path}")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")
