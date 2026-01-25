#!/usr/bin/env python3
"""
Migration script to replace invoice references with billing_document references.
Handles template paths, class names, and field names across documentation.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Define replacement patterns
REPLACEMENTS = [
    # Template imports and paths
    (r"from examples\.templates\.invoice import BillingDocument",
     r"from examples.templates.billing_document import BillingDocument"),
    (r"templates\.invoice\.BillingDocument",
     r"templates.billing_document.BillingDocument"),
    (r"my_templates\.BillingDocument",
     r"templates.BillingDocument"),
    (r'"Invoice"',
     r'"BillingDocument"'),

    # Class names in code examples
    (r"\bInvoice\(",
     r"BillingDocument("),
    (r"= Invoice\b",
     r"= BillingDocument"),
    (r": Invoice\b",
     r": BillingDocument"),
    (r"\[Invoice\]",
     r"[BillingDocument]"),

    # Field names
    (r"invoice_number",
     r"document_no"),
    (r"invoice\.invoice_number",
     r"billing_doc.document_no"),

    # File references
    (r"invoice\.py",
     r"billing_document.py"),
    (r"invoice_template\.py",
     r"billing_document_template.py"),

    # Documentation text (be careful with context)
    (r"Invoice extraction template",
     r"BillingDocument extraction template"),
    (r"Invoice document",
     r"BillingDocument document"),
    (r"Invoice entity",
     r"BillingDocument entity"),
    (r"Invoice model",
     r"BillingDocument model"),
    (r"Invoice template",
     r"BillingDocument template"),
    (r"Invoice \(node\)",
     r"BillingDocument (node)"),
    (r"Invoice-(\d+)",
     r"BillingDocument-\1"),

    # Node/label references
    (r"'Invoice'",
     r"'BillingDocument'"),
    (r'label == "Invoice"',
     r'label == "BillingDocument"'),
    (r"label: Invoice",
     r"label: BillingDocument"),
    (r":Invoice\b",
     r":BillingDocument"),

    # Comments and descriptions
    (r"# Invoice",
     r"# BillingDocument"),
    (r"## Invoice",
     r"## BillingDocument"),
    (r"### Invoice",
     r"### BillingDocument"),
]

# Files to process (Phase 3 and 4)
PHASE_3_FILES = [
    "docs/fundamentals/schema-definition/index.md",
    "docs/fundamentals/schema-definition/best-practices.md",
    "docs/fundamentals/schema-definition/relationships.md",
    "docs/fundamentals/schema-definition/field-definitions.md",
    "docs/fundamentals/schema-definition/validation.md",
    "docs/fundamentals/schema-definition/entities-vs-components.md",
    "docs/fundamentals/graph-management/graph-conversion.md",
    "docs/fundamentals/graph-management/neo4j-integration.md",
    "docs/fundamentals/graph-management/visualization.md",
    "docs/fundamentals/graph-management/graph-analysis.md",
    "docs/fundamentals/graph-management/index.md",
]

PHASE_4_FILES = [
    "docs/fundamentals/pipeline-configuration/configuration-basics.md",
    "docs/fundamentals/pipeline-configuration/configuration-examples.md",
    "docs/fundamentals/pipeline-configuration/model-configuration.md",
    "docs/fundamentals/pipeline-configuration/processing-modes.md",
    "docs/fundamentals/pipeline-configuration/export-configuration.md",
    "docs/fundamentals/pipeline-configuration/input-formats.md",
    "docs/fundamentals/pipeline-configuration/docling-settings.md",
    "docs/fundamentals/pipeline-configuration/index.md",
    "docs/fundamentals/extraction-process/extraction-backends.md",
    "docs/fundamentals/extraction-process/model-merging.md",
    "docs/fundamentals/extraction-process/index.md",
    "docs/fundamentals/extraction-process/chunking-strategies.md",
    "docs/usage/examples/docling-document-input.md",
    "docs/usage/api/run-pipeline.md",
    "docs/usage/api/programmatic-examples.md",
    "docs/usage/api/batch-processing.md",
    "docs/usage/api/index.md",
    "docs/usage/cli/inspect-command.md",
    "docs/usage/cli/index.md",
    "docs/usage/advanced/error-handling.md",
]

def process_file(file_path: Path, replacements: List[Tuple[str, str]]) -> Tuple[int, int]:
    """
    Process a single file with all replacements.
    Returns (lines_changed, total_replacements)
    """
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return 0, 0

    content = file_path.read_text(encoding="utf-8")
    original_content = content
    total_replacements = 0

    for pattern, replacement in replacements:
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            total_replacements += matches

    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        lines_changed = len([line for line in content.split("\n")
                           if line not in original_content.split("\n")])
        return lines_changed, total_replacements

    return 0, 0

def main() -> None:
    """Run the migration."""
    print("🚀 Starting migration: invoice → billing_document\n")

    all_files = PHASE_3_FILES + PHASE_4_FILES
    total_files = len(all_files)
    processed = 0
    total_changes = 0
    total_replacements = 0

    for file_path_str in all_files:
        file_path = Path(file_path_str)
        lines_changed, replacements = process_file(file_path, REPLACEMENTS)

        if replacements > 0:
            processed += 1
            total_changes += lines_changed
            total_replacements += replacements
            print(f"✅ {file_path.name}: {replacements} replacements")
        else:
            print(f"⏭️  {file_path.name}: No changes needed")

    print("\n📊 Migration Summary:")
    print(f"   Files processed: {processed}/{total_files}")
    print(f"   Total replacements: {total_replacements}")
    print(f"   Lines changed: {total_changes}")
    print("\n✅ Migration complete!")
    print("\n⚠️  Manual review recommended for:")
    print("   - Context-specific 'invoice' references (keep lowercase when referring to document type)")
    print("   - Comments that discuss invoices conceptually")
    print("   - Example data that should remain as 'invoice'")

if __name__ == "__main__":
    main()

# Made with Bob
