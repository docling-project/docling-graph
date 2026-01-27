# Debug Mode

## Overview

Debug mode provides comprehensive visibility into the atomic extraction pipeline's intermediate stages, enabling debugging, performance analysis, and quality assurance. When enabled, all extraction artifacts are persisted to disk for inspection and replay.

**What's Captured:**
- **Slot Metadata**: Chunks/pages with token counts and text hashes
- **Atom Extractions**: Raw LLM outputs and validated results for each attempt
- **Field Catalog**: Global catalog and per-slot selections
- **Reducer State**: Applied/rejected facts, conflicts, and arbitration decisions
- **Best-Effort Model**: Final model output with validation status
- **Provenance**: Document path and configuration for replay

---

## Quick Start

### Enable Debug Mode

Debug mode is controlled by a single flag - no levels, no complexity.

**CLI:**
```bash
# Enable debug mode
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --debug
```

**API:**
```python
from docling_graph import run_pipeline, PipelineConfig

# Enable debug mode
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True  # Single flag - all or nothing
)

context = run_pipeline(config)
```

---

## Output Structure

When debug mode is enabled, all artifacts are saved under `outputs/{document}_{timestamp}/debug/`:

```
outputs/invoice_pdf_20260206_094500/
├── metadata.json                    # Pipeline metadata
├── docling/                         # Docling conversion output
│   ├── document.json
│   └── document.md
├── docling_graph/                   # Graph outputs
│   ├── graph.json
│   ├── nodes.csv
│   ├── edges.csv
│   └── ...
└── debug/                           # 🔍 Debug artifacts
    ├── slots.jsonl                  # Slot metadata (one per line)
    ├── atoms_all.jsonl              # All atomic facts (one per line)
    ├── field_catalog.json           # Global field catalog
    ├── reducer_report.json          # Reducer decisions and conflicts
    ├── best_effort_model.json       # Final model output
    ├── provenance.json              # Document path and config for replay
    │
    ├── slots_text/                  # Full slot text for replay
    │   ├── slot_0.txt
    │   ├── slot_1.txt
    │   └── ...
    │
    ├── atoms/                       # Per-slot extraction attempts
    │   ├── slot_0_attempt1.json
    │   ├── slot_0_attempt2.json     # Retry if first failed
    │   ├── slot_1_attempt1.json
    │   └── ...
    │
    ├── field_catalog_selected/      # Per-slot field selections
    │   ├── slot_0.json
    │   ├── slot_1.json
    │   └── ...
    │
    └── arbitration/                 # Conflict resolution
        ├── request.json             # Conflicts sent to LLM
        └── response.json            # LLM arbitration decisions
```

---

## Debug Artifacts Explained

### 1. Slot Metadata (`slots.jsonl`)

One line per slot with compact metadata:

```json
{"slot_id": "slot_0", "chunk_id": 0, "page_numbers": [1, 2], "token_count": 1500, "text_hash": "a3f2e1d4c5b6...", "text_length": 8234}
{"slot_id": "slot_1", "chunk_id": 1, "page_numbers": [3, 4], "token_count": 1450, "text_hash": "b4e3f2d5c6a7...", "text_length": 7891}
```

**Use Cases:**
- Verify chunking strategy
- Identify token count issues
- Check page coverage

### 2. Slot Text (`slots_text/*.txt`)

Full text content for each slot, enabling replay without re-processing the document.

**Use Cases:**
- Replay extraction with different prompts
- Analyze what the LLM actually saw
- Debug extraction errors

### 3. Atom Extractions (`atoms/*.json`)

Raw LLM outputs and validated results for each extraction attempt:

```json
{
  "slot_id": "slot_0",
  "attempt": 1,
  "timestamp": "2026-02-06T09:45:00",
  "raw_output": "{\"atoms\": [...]}",
  "validation_success": true,
  "error": null,
  "validated_result": {
    "atoms": [
      {
        "field_path": "invoice_number",
        "value": "INV-2024-001",
        "confidence": 0.95,
        "source_quote": "Invoice #INV-2024-001"
      }
    ]
  }
}
```

**Use Cases:**
- Debug validation failures
- Analyze LLM output quality
- Track retry attempts
- Identify prompt issues

### 4. All Atoms (`atoms_all.jsonl`)

All atomic facts in one file (one per line) for easy analysis:

```json
{"field_path": "invoice_number", "value": "INV-2024-001", "confidence": 0.95, "source_quote": "Invoice #INV-2024-001", "slot_id": "slot_0"}
{"field_path": "total_amount", "value": 1250.50, "confidence": 0.90, "source_quote": "Total: $1,250.50", "slot_id": "slot_0"}
```

**Use Cases:**
- Grep for specific fields
- Analyze confidence distributions
- Compare facts across slots

### 5. Field Catalog (`field_catalog.json`)

Global catalog of all extractable fields from the template:

```json
[
  {
    "field_path": "invoice_number",
    "field_type": "str",
    "description": "Invoice number",
    "is_required": true,
    "is_list": false
  },
  {
    "field_path": "line_items[].description",
    "field_type": "str",
    "description": "Item description",
    "is_required": true,
    "is_list": true
  }
]
```

**Use Cases:**
- Verify template structure
- Check field descriptions
- Debug field selection

### 6. Selected Field Catalog (`field_catalog_selected/*.json`)

Per-slot field selections showing which fields were targeted:

```json
[
  {
    "field_path": "invoice_number",
    "field_type": "str",
    "description": "Invoice number",
    "is_required": true,
    "is_list": false
  }
]
```

**Use Cases:**
- Verify field selection logic
- Debug missing extractions
- Optimize field targeting

### 7. Reducer Report (`reducer_report.json`)

Complete reducer state with applied/rejected facts and conflicts:

```json
{
  "applied_facts": [
    {
      "field_path": "invoice_number",
      "value": "INV-2024-001",
      "confidence": 0.95,
      "slot_id": "slot_0"
    }
  ],
  "rejected_facts": [
    {
      "field_path": "invoice_number",
      "value": "INV-2024-002",
      "confidence": 0.60,
      "slot_id": "slot_1",
      "reason": "Lower confidence than existing fact"
    }
  ],
  "conflicts": [
    {
      "field_path": "total_amount",
      "conflicting_values": [
        {"value": 1250.50, "confidence": 0.90, "slot_id": "slot_0"},
        {"value": 1250.00, "confidence": 0.85, "slot_id": "slot_1"}
      ],
      "resolution": "arbitration",
      "winner": {"value": 1250.50, "confidence": 0.90, "slot_id": "slot_0"}
    }
  ]
}
```

**Use Cases:**
- Debug conflict resolution
- Analyze fact quality
- Verify arbitration decisions

### 8. Arbitration (`arbitration/*.json`)

LLM arbitration requests and responses for conflicts:

**Request:**
```json
{
  "conflicts": [
    {
      "field_path": "total_amount",
      "options": [
        {"value": 1250.50, "quote": "Total: $1,250.50", "slot_id": "slot_0"},
        {"value": 1250.00, "quote": "Amount Due: $1,250", "slot_id": "slot_1"}
      ]
    }
  ]
}
```

**Response:**
```json
{
  "decisions": [
    {
      "field_path": "total_amount",
      "selected_value": 1250.50,
      "reasoning": "First value includes cents, more precise"
    }
  ]
}
```

**Use Cases:**
- Debug arbitration logic
- Analyze LLM reasoning
- Improve conflict resolution

### 9. Best-Effort Model (`best_effort_model.json`)

Final model output with validation status:

```json
{
  "template": "BillingDocument",
  "timestamp": "2026-02-06T09:45:30",
  "validation_success": true,
  "model": {
    "invoice_number": "INV-2024-001",
    "total_amount": 1250.50,
    "line_items": [
      {"description": "Service A", "amount": 500.00},
      {"description": "Service B", "amount": 750.50}
    ]
  }
}
```

**Use Cases:**
- Verify final output
- Debug validation failures
- Compare with expected results

### 10. Provenance (`provenance.json`)

Document path and configuration for deterministic replay:

```json
{
  "document_path": "/path/to/invoice.pdf",
  "docling_config": {"pipeline": "ocr"},
  "chunker_config": {"max_tokens": 2000},
  "timestamp": "2026-02-06T09:45:00"
}
```

**Use Cases:**
- Replay extraction with same config
- Track configuration changes
- Debug environment issues

---

## Common Debugging Patterns

### Pattern 1: Find Validation Failures

```bash
# Check atom extraction attempts for errors
cd outputs/invoice_pdf_20260206_094500/debug/atoms/
grep -l '"validation_success": false' *.json

# View specific failure
cat slot_0_attempt1.json | jq '.error'
```

### Pattern 2: Analyze Confidence Distribution

```bash
# Extract all confidence scores
cd outputs/invoice_pdf_20260206_094500/debug/
cat atoms_all.jsonl | jq -r '.confidence' | sort -n
```

### Pattern 3: Find Conflicts

```bash
# Check reducer report for conflicts
cd outputs/invoice_pdf_20260206_094500/debug/
cat reducer_report.json | jq '.conflicts'
```

### Pattern 4: Verify Field Coverage

```bash
# List all extracted fields
cd outputs/invoice_pdf_20260206_094500/debug/
cat atoms_all.jsonl | jq -r '.field_path' | sort -u
```

### Pattern 5: Compare Slot Extractions

```bash
# Compare what each slot extracted
cd outputs/invoice_pdf_20260206_094500/debug/atoms/
for f in slot_*_attempt1.json; do
  echo "=== $f ==="
  cat "$f" | jq '.validated_result.atoms | length'
done
```

---

## Programmatic Analysis

### Load Debug Artifacts in Python

```python
import json
from pathlib import Path

debug_dir = Path("outputs/invoice_pdf_20260206_094500/debug")

# Load slot metadata
slots = []
with open(debug_dir / "slots.jsonl") as f:
    for line in f:
        slots.append(json.loads(line))

# Load all atoms
atoms = []
with open(debug_dir / "atoms_all.jsonl") as f:
    for line in f:
        atoms.append(json.loads(line))

# Load reducer report
with open(debug_dir / "reducer_report.json") as f:
    report = json.load(f)

# Analyze
print(f"Total slots: {len(slots)}")
print(f"Total atoms: {len(atoms)}")
print(f"Conflicts: {len(report['conflicts'])}")
```

### Analyze Extraction Quality

```python
import json
from pathlib import Path
from collections import Counter

debug_dir = Path("outputs/invoice_pdf_20260206_094500/debug")

# Load all atoms
atoms = []
with open(debug_dir / "atoms_all.jsonl") as f:
    for line in f:
        atoms.append(json.loads(line))

# Confidence distribution
confidences = [a['confidence'] for a in atoms]
print(f"Average confidence: {sum(confidences) / len(confidences):.2f}")
print(f"Min confidence: {min(confidences):.2f}")
print(f"Max confidence: {max(confidences):.2f}")

# Field coverage
fields = Counter(a['field_path'] for a in atoms)
print("\nField extraction counts:")
for field, count in fields.most_common():
    print(f"  {field}: {count}")
```

### Find Problematic Slots

```python
import json
from pathlib import Path

debug_dir = Path("outputs/invoice_pdf_20260206_094500/debug")
atoms_dir = debug_dir / "atoms"

# Find slots with validation failures
failed_slots = []
for attempt_file in atoms_dir.glob("*_attempt*.json"):
    with open(attempt_file) as f:
        attempt = json.load(f)
        if not attempt['validation_success']:
            failed_slots.append({
                'slot_id': attempt['slot_id'],
                'attempt': attempt['attempt'],
                'error': attempt['error']
            })

print(f"Found {len(failed_slots)} failed attempts:")
for failure in failed_slots:
    print(f"  {failure['slot_id']} (attempt {failure['attempt']}): {failure['error']}")
```

---

## Best Practices

### 1. Enable Debug Mode During Development

```python
# ✅ Good - Debug enabled during development
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True  # Enable for development
)
```

### 2. Disable Debug Mode in Production

```python
# ✅ Good - Debug disabled in production
import os

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)
```

### 3. Use Debug Artifacts for Testing

```python
def test_extraction_quality():
    """Test extraction quality using debug artifacts."""
    config = PipelineConfig(
        source="test_document.pdf",
        template="templates.BillingDocument",
        debug=True
    )
    
    context = run_pipeline(config)
    
    # Load debug artifacts
    debug_dir = Path(context.output_dir) / "debug"
    
    # Verify no validation failures
    atoms_dir = debug_dir / "atoms"
    for attempt_file in atoms_dir.glob("*_attempt*.json"):
        with open(attempt_file) as f:
            attempt = json.load(f)
            assert attempt['validation_success'], f"Validation failed: {attempt['error']}"
    
    # Verify confidence thresholds
    with open(debug_dir / "atoms_all.jsonl") as f:
        for line in f:
            atom = json.loads(line)
            assert atom['confidence'] >= 0.7, f"Low confidence: {atom}"
```

### 4. Archive Debug Artifacts

```bash
# Archive debug artifacts for later analysis
cd outputs/
tar -czf invoice_debug_20260206.tar.gz invoice_pdf_20260206_094500/debug/
```

---

## Troubleshooting

### 🐛 No Debug Artifacts Generated

**Problem:** Debug directory is empty or missing.

**Solution:**
```python
# Ensure debug=True is set
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    debug=True  # Must be True
)
```

### 🐛 Validation Failures

**Problem:** Many atom extraction attempts show `validation_success: false`.

**Solution:**
1. Check `atoms/*_attempt*.json` for error messages
2. Review field descriptions in `field_catalog.json`
3. Examine slot text in `slots_text/*.txt`
4. Adjust template or prompts based on errors

### 🐛 Low Confidence Scores

**Problem:** Atoms have consistently low confidence scores.

**Solution:**
1. Review `atoms_all.jsonl` for confidence distribution
2. Check `slots_text/*.txt` for text quality
3. Verify field descriptions are clear
4. Consider adjusting chunking strategy

### 🐛 Many Conflicts

**Problem:** Reducer report shows many conflicts.

**Solution:**
1. Review `reducer_report.json` for conflict patterns
2. Check `arbitration/request.json` and `response.json`
3. Analyze overlapping slot content in `slots_text/`
4. Consider adjusting chunking to reduce overlap

---

## Performance Considerations

### Disk Space

Debug mode saves all intermediate artifacts, which can use significant disk space:

- **Small documents** (1-5 pages): ~1-5 MB
- **Medium documents** (10-50 pages): ~10-50 MB
- **Large documents** (100+ pages): ~100+ MB

**Recommendation:** Clean up old debug artifacts regularly.

### Processing Time

Debug mode adds minimal overhead (~1-2% slower) since artifacts are written asynchronously.

---

## Next Steps

- **[CLI Documentation](../../usage/cli/convert-command.md)** - CLI usage with debug flag
- **[API Documentation](../../usage/api/pipeline-config.md)** - API usage with debug mode
- **[Testing Guide](../advanced/testing.md)** - Using debug artifacts in tests
- **[Batch Processing](../api/batch-processing.md)** - Process multiple documents