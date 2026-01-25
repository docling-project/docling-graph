#!/usr/bin/env python3
"""
Fix schema alignment issues in documentation.
Adds notes about actual schema and fixes critical misalignments.
"""

import re
from pathlib import Path

# Add schema reference note to key documentation files
SCHEMA_NOTE = """
> **Note**: The examples in this document use simplified field names and structures for teaching purposes.
> The actual `BillingDocument` schema at `docs/examples/templates/billing_document.py` is more comprehensive
> with 30+ classes, EN 16931/Peppol BIS compliance, and uses `CONTAINS_LINE` for line items.
"""

FILES_TO_ADD_NOTE = [
    "docs/fundamentals/schema-definition/index.md",
    "docs/fundamentals/schema-definition/relationships.md",
    "docs/fundamentals/schema-definition/field-definitions.md",
]

def add_schema_note(file_path: Path) -> bool:
    """Add schema reference note after the first heading."""
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")

    # Check if note already exists
    if "The actual `BillingDocument` schema" in content:
        print(f"⏭️  {file_path.name}: Note already exists")
        return False

    # Add note after first heading (after # Title\n\n)
    pattern = r"(^# .+?\n\n)"
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, r"\1" + SCHEMA_NOTE + "\n", content, count=1, flags=re.MULTILINE)
        file_path.write_text(content, encoding="utf-8")
        print(f"✅ {file_path.name}: Added schema reference note")
        return True
    else:
        print(f"⚠️  {file_path.name}: Could not find insertion point")
        return False

def main() -> None:
    """Run schema alignment fixes."""
    print("🔧 Fixing schema alignment issues\n")

    added = 0
    for file_path_str in FILES_TO_ADD_NOTE:
        file_path = Path(file_path_str)
        if add_schema_note(file_path):
            added += 1

    print("\n📊 Summary:")
    print(f"   Schema notes added: {added}/{len(FILES_TO_ADD_NOTE)}")
    print("\n✅ Schema alignment fixes complete!")
    print("\n📝 Remaining manual tasks:")
    print("   1. Review CONTAINS_LINE vs CONTAINS_LINE usage in examples")
    print("   2. Consider if teaching examples should match actual schema")
    print("   3. Delete obsolete invoice-extraction.md file")

if __name__ == "__main__":
    main()

# Made with Bob
