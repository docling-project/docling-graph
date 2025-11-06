"""
Example 01: Basic VLM Extraction from an Image
"""

import sys
from pathlib import Path

from rich import print as rich_print

# --- 1. Add project root to sys.path ---
# This allows the script to import 'docling_graph' and 'examples'
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
OUTPUT_DIR = "outputs/01_vlm_from_image"

rich_print("[blue]--- Running Example 01: VLM from Image ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,
    # --- Key Settings ---
    backend="vlm",
    inference="local",  # VLM backend must be 'local'
    processing_mode="one-to-one",  # Best for single-page/image docs
    docling_config="vision",  # 'vision' is required for VLM
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")
