"""
Example 02: VLM Extraction from a Single-Page PDF
"""

import sys
from pathlib import Path
from rich import print as rich_print


# --- 1. Add project root to sys.path ---
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from docling_graph import PipelineConfig
    from examples.templates.id_card import IDCard
except ImportError:
    rich_print("[red]Error:[/red] Could not import modules. ")
    rich_print("Please run this script from the project root directory.")
    sys.exit(1)

# --- 2. Define Paths (relative to project root) ---
SOURCE_FILE = "docs/examples/data/id_card/french_id_card_sp.pdf"
TEMPLATE_CLASS = IDCard
OUTPUT_DIR = "outputs/02_vlm_from_pdf"

rich_print(f"[blue]--- Running Example 02: VLM from PDF ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,

    # --- Key Settings ---
    backend="vlm",
    inference="local",
    processing_mode="one-to-one",
    docling_config="vision", # 'vision' is required for VLM
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")