"""
Example 06: Page-by-Page Extraction ('one-to-one' on PDF)
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
SOURCE_FILE = "docs/examples/data/id_card/multi_french_id_cards.pdf"
TEMPLATE_CLASS = IDCard
OUTPUT_DIR = "outputs/06_llm_one_to_one"

rich_print(f"[blue]--- Running Example 06: 'one-to-one' on PDF ---[/blue]")

# --- 3. Configure the Pipeline ---
config = PipelineConfig(
    source=SOURCE_FILE,
    template=TEMPLATE_CLASS,
    output_dir=OUTPUT_DIR,

    # --- Key Settings ---
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3:8b",
    
    # Process each page as a separate document
    processing_mode="one-to-one",
    
    # Chunking is not used in 'one-to-one' mode
    use_chunking=False,
)

# --- 4. Run the pipeline ---
try:
    config.run()
    rich_print(f"\n[green]Success![/green] Graph data saved to: {OUTPUT_DIR}")
    rich_print(f"Explore: {OUTPUT_DIR}/{Path(SOURCE_FILE).stem}_graph.html")
except Exception as e:
    rich_print(f"\n[red]Pipeline Failed:[/red] {e}")