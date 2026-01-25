# Model Merging


## Overview

**Model merging** is the process of consolidating multiple Pydantic model instances into a single unified model. This is essential when extracting from multiple chunks or pages.

**In this guide:**
- Why merging is needed
- Programmatic vs LLM merging
- Chain of Density consolidation
- Zero data loss strategies
- Deduplication strategies
- Conflict resolution
- Best practices

!!! tip "New: Chain of Density Consolidation"
    For ADVANCED tier models (13B+), LLM consolidation now uses a multi-turn "Chain of Density" approach for higher quality results. See [Model Capabilities](model-capabilities.md) for details.

---

## Why Merging Matters

### The Multi-Extraction Problem

When processing large documents:

```python
# Document split into 3 chunks
chunk_1 = "Invoice INV-001, Issued by: Acme Corp"
chunk_2 = "Line items: Product A ($50), Product B ($100)"
chunk_3 = "Total: $150, Due date: 2024-01-31"

# Each chunk produces a partial model
model_1 = BillingDocument(document_no="INV-001", issued_by=Organization(name="Acme Corp"))
model_2 = BillingDocument(line_items=[LineItem(...), LineItem(...)])
model_3 = BillingDocument(total=150, due_date="2024-01-31")

# Need to merge into one complete model
final_model = merge(model_1, model_2, model_3)
```

**Without merging:** Incomplete, fragmented data  
**With merging:** Complete, unified model

---

## Merging Strategies

### Quick Comparison

| Strategy | Speed | Accuracy | Cost | Use Case |
|:---------|:------|:---------|:-----|:---------|
| **Programmatic** | ⚡ Fast | 🟡 Good | Free | Default, simple merging |
| **LLM (Standard)** | 🐢 Slow | 🟢 Better | $ API cost | High accuracy needs |
| **LLM (Chain of Density)** | 🐌 Slower | 💎 Best | $$$ 3x API cost | Critical documents |

!!! info "Zero Data Loss"
    All merging strategies now implement zero data loss - if merging fails, the system returns partial models instead of empty results.

---

## Programmatic Merging

### What is Programmatic Merging?

**Programmatic merging** uses rule-based algorithms to combine models without LLM calls. It's fast, free, and works well for most cases.

### How It Works

--8<-- "docs/assets/flowcharts/programmatic_merge.md"

### Basic Usage

```python
from docling_graph.core.utils import merge_pydantic_models

# Multiple partial models
models = [
    BillingDocument(document_no="INV-001", issued_by=Organization(name="Acme")),
    BillingDocument(line_items=[LineItem(description="Product A", total=50)]),
    BillingDocument(total=150, due_date="2024-01-31")
]

# Merge programmatically
merged = merge_pydantic_models(models, Invoice)

print(merged)
# BillingDocument(
#     document_no="INV-001",
#     issued_by=Organization(name="Acme"),
#     line_items=[LineItem(description="Product A", total=50)],
#     total=150,
#     due_date="2024-01-31"
# )
```

---

### Merge Rules

#### 1. Field Overwriting

**Rule:** Non-empty values overwrite empty ones

```python
# Model 1
BillingDocument(document_no="INV-001", total=None)

# Model 2
BillingDocument(document_no=None, total=150)

# Merged
BillingDocument(document_no="INV-001", total=150)
```

#### 2. List Concatenation

**Rule:** Lists are concatenated and deduplicated

```python
# Model 1
BillingDocument(line_items=[LineItem(description="Product A")])

# Model 2
BillingDocument(line_items=[LineItem(description="Product B")])

# Merged
BillingDocument(line_items=[
    LineItem(description="Product A"),
    LineItem(description="Product B")
])
```

#### 3. Nested Object Merging

**Rule:** Nested objects are recursively merged

```python
# Model 1
BillingDocument(issued_by=Organization(name="Acme"))

# Model 2
BillingDocument(issued_by=Organization(address=Address(city="Paris")))

# Merged
BillingDocument(issued_by=Organization(
    name="Acme",
    address=Address(city="Paris")
))
```

#### 4. Entity Deduplication

**Rule:** Duplicate entities are detected and removed

```python
# Model 1
BillingDocument(line_items=[LineItem(description="Product A", total=50)])

# Model 2 (duplicate)
BillingDocument(line_items=[LineItem(description="Product A", total=50)])

# Merged (deduplicated)
BillingDocument(line_items=[LineItem(description="Product A", total=50)])
```

---

### Deduplication Algorithm

#### Content-Based Hashing

Entities are deduplicated using content hashing:

```python
def entity_hash(entity: dict) -> str:
    """Compute content hash for entity."""
    # Use stable fields (exclude id, __class__)
    stable_fields = {
        k: v for k, v in entity.items() 
        if k not in {"id", "__class__"} and v is not None
    }
    
    # Create stable JSON representation
    content = json.dumps(stable_fields, sort_keys=True)
    
    # Hash content
    return hashlib.blake2b(content.encode()).hexdigest()[:16]
```

**Example:**
```python
# These are considered duplicates
entity_1 = {"name": "Acme Corp", "city": "Paris"}
entity_2 = {"name": "Acme Corp", "city": "Paris"}

# These are different
entity_3 = {"name": "Acme Corp", "city": "London"}
```

---

## LLM Consolidation

### What is LLM Consolidation?

**LLM consolidation** uses an LLM to intelligently merge models, resolving conflicts and improving accuracy. For ADVANCED tier models (13B+), it uses a multi-turn "Chain of Density" approach.

### Consolidation Modes

#### Standard Consolidation (SIMPLE/STANDARD tiers)

Single-turn consolidation for models < 13B parameters:

```python
# Automatic for SIMPLE/STANDARD tier models
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local",
    provider_override="ollama",
    model_override="llama3.1:8b",  # STANDARD tier
    llm_consolidation=True  # Single-turn consolidation
)
```

**Process:**
1. Receive raw models and programmatic merge
2. Create consolidated model in one LLM call
3. Validate and return

**Performance:**
- Speed: ~3-5 seconds
- Token usage: ~1000-2000 tokens
- Accuracy: 95%

#### Chain of Density Consolidation (ADVANCED tier)

Multi-turn consolidation for models ≥ 13B parameters:

```python
# Automatic for ADVANCED tier models
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True  # Chain of Density (3 turns)
)
```

**Process:**
1. **Initial Merge** (Turn 1): Create first consolidated version
2. **Refinement** (Turn 2): Identify and resolve conflicts
3. **Final Polish** (Turn 3): Ensure completeness and accuracy

**Performance:**
- Speed: ~10-15 seconds (3x slower)
- Token usage: ~3000-6000 tokens (3x more)
- Accuracy: 98%

### When to Use

✅ **Use Standard LLM consolidation when:**
- High accuracy is needed (95%)
- Complex conflict resolution required
- Using SIMPLE/STANDARD tier models
- Budget allows API calls

✅ **Use Chain of Density when:**
- Critical accuracy required (98%)
- Legal or financial documents
- Using ADVANCED tier models (13B+)
- Budget allows 3x API costs

❌ **Don't use LLM consolidation when:**
- Speed is priority
- Cost is primary concern
- Simple merging sufficient
- Processing high volume (>1000 docs)

---

### How It Works

--8<-- "docs/assets/flowcharts/llm_backend.md"

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    llm_consolidation=True  # Enable LLM consolidation
)

run_pipeline(config)
```

---

### LLM Consolidation Process

#### Step 1: Programmatic Merge

```python
# First, merge programmatically
programmatic_model = merge_pydantic_models(raw_models, template)
```

#### Step 2: LLM Review

```python
# Then, LLM reviews and improves
final_model = backend.consolidate_from_pydantic_models(
    raw_models=raw_models,
    programmatic_model=programmatic_model,
    template=template
)
```

#### Step 3: Validation

```python
# LLM output is validated against schema
validated_model = template.model_validate(llm_output)
```

---

### Chain of Density Process

For ADVANCED tier models, consolidation uses three turns:

#### Turn 1: Initial Merge

```python
# Prompt includes:
# - Schema definition
# - All raw models
# - Programmatic merge result

# LLM creates initial consolidated version
initial_model = llm.consolidate(
    schema=schema,
    raw_models=raw_models,
    draft=programmatic_merge
)
```

#### Turn 2: Refinement

```python
# Prompt includes:
# - Initial model from Turn 1
# - Identified conflicts
# - Missing information

# LLM refines and resolves conflicts
refined_model = llm.refine(
    initial=initial_model,
    conflicts=identified_conflicts,
    raw_models=raw_models
)
```

#### Turn 3: Final Polish

```python
# Prompt includes:
# - Refined model from Turn 2
# - Completeness checklist
# - Accuracy verification

# LLM ensures completeness and accuracy
final_model = llm.polish(
    refined=refined_model,
    schema=schema,
    raw_models=raw_models
)
```

**Benefits:**
- 💎 Highest accuracy (98% vs 95%)
- 🔍 Better conflict resolution
- ✅ More complete data extraction
- 🎯 Fewer validation errors

**Trade-offs:**
- 🐌 3x slower processing
- 💰 3x higher API costs
- 📊 3x more token usage

See [Model Capabilities: Chain of Density](model-capabilities.md#chain-of-density-consolidation) for complete details.

---

### LLM Consolidation Prompt

The LLM receives:

1. **Schema:** Pydantic model structure
2. **Raw models:** All partial extractions
3. **Draft model:** Programmatic merge result

**Task:** Create the best possible consolidated model

**For Chain of Density (ADVANCED tier):**
- Turn 1: Initial consolidation
- Turn 2: Conflict resolution
- Turn 3: Completeness verification

---

## Complete Examples

### 📍 Basic Programmatic Merge

```python
from docling_graph.core.utils import merge_pydantic_models
from templates.billing_document import BillingDocument, Organization, LineItem

# Partial models from chunks
models = [
    BillingDocument(
        document_no="INV-001",
        issued_by=Organization(name="Acme Corp")
    ),
    BillingDocument(
        line_items=[
            LineItem(description="Product A", quantity=2, unit_price=50, total=100),
            LineItem(description="Product B", quantity=1, unit_price=150, total=150)
        ]
    ),
    BillingDocument(
        subtotal=250,
        tax=25,
        total=275,
        due_date="2024-01-31"
    )
]

# Merge
merged = merge_pydantic_models(models, Invoice)

print(f"Invoice: {merged.document_no}")
print(f"Issued by: {merged.issued_by.name}")
print(f"Line items: {len(merged.line_items)}")
print(f"Total: ${merged.total}")
```

### 📍 With LLM Consolidation

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    
    # Enable LLM consolidation
    backend="llm",
    inference="remote",
    provider_override="mistral",
    llm_consolidation=True,  # Extra accuracy
    
    # Chunking settings
    use_chunking=True,
    processing_mode="many-to-one",
    
    output_dir="outputs/consolidated"
)

run_pipeline(config)
```

### 📍 Manual Consolidation

```python
from docling_graph.core.extractors.backends import LlmBackend
from docling_graph.core.utils import merge_pydantic_models
from docling_graph.llm_clients import MistralClient

# Extract from chunks
models = []
for chunk in chunks:
    model = backend.extract_from_markdown(chunk, template, is_partial=True)
    if model:
        models.append(model)

# Programmatic merge
programmatic = merge_pydantic_models(models, template)

# LLM consolidation
client = MistralClient(model="mistral-large-latest")
backend = LlmBackend(llm_client=client)

final = backend.consolidate_from_pydantic_models(
    raw_models=models,
    programmatic_model=programmatic,
    template=template
)

print(f"Consolidated {len(models)} models into 1")
```

### 📍 Handling Merge Failures

```python
from docling_graph.core.utils import merge_pydantic_models

try:
    merged = merge_pydantic_models(models, template)
    print("✅ Merge successful")
    
except Exception as e:
    print(f"❌ Merge failed: {e}")
    
    # Fallback: use first model
    merged = models[0] if models else template()
    print("Using first model as fallback")
```

---

## Zero Data Loss

### What is Zero Data Loss?

**Zero data loss** ensures that extraction failures never result in completely empty results. Instead, the system returns partial models with whatever data was successfully extracted.

### How It Works

#### Before (Old Behavior)

```python
# If merging failed
try:
    merged = merge_pydantic_models(models, template)
except Exception:
    return []  # ❌ All data lost!
```

#### After (Zero Data Loss)

```python
# If merging fails, return partial models
try:
    merged = merge_pydantic_models(models, template)
    return [merged]
except Exception:
    # ✅ Return partial models instead of empty list
    return models if models else [template()]
```

### Benefits

**Data Preservation:**
```python
# Even if consolidation fails, you get partial data
models = [
    BillingDocument(document_no="INV-001"),  # From chunk 1
    BillingDocument(line_items=[...]),          # From chunk 2
    BillingDocument(total=150)                  # From chunk 3
]

# If merge fails, you still have all 3 partial models
# Better than nothing!
```

**Graceful Degradation:**
```python
# System continues processing even with errors
for document in documents:
    try:
        result = process_document(document)
        # May return merged model or partial models
        print(f"Extracted {len(result)} model(s)")
    except Exception as e:
        print(f"Error: {e}")
        # But still got partial data
```

### Configuration

Zero data loss is automatic - no configuration needed:

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    processing_mode="many-to-one"
    # Zero data loss is always enabled
)
```

### Example: Handling Partial Results

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    processing_mode="many-to-one"
)

results = run_pipeline(config)

# Check if we got merged or partial models
if len(results) == 1:
    print("✅ Successfully merged into single model")
    merged = results[0]
else:
    print(f"⚠ Got {len(results)} partial models")
    # Still useful! Can manually merge or use as-is
    for i, model in enumerate(results, 1):
        print(f"  Model {i}: {model.document_no or 'N/A'}")
```

---

## Conflict Resolution

### Common Conflicts

#### 1. Duplicate Entities

**Problem:** Same entity appears multiple times

```python
# Chunk 1
Organization(name="Acme Corp", city="Paris")

# Chunk 2
Organization(name="Acme Corp", city="Paris")

# Solution: Deduplicated automatically
Organization(name="Acme Corp", city="Paris")  # Only one
```

#### 2. Conflicting Values

**Problem:** Different values for same field

```python
# Chunk 1
BillingDocument(total=150)

# Chunk 2
BillingDocument(total=275)

# Programmatic: Last value wins
BillingDocument(total=275)

# LLM: Intelligent resolution
BillingDocument(total=275)  # LLM chooses correct value
```

#### 3. Partial Information

**Problem:** Information spread across chunks

```python
# Chunk 1
Organization(name="Acme Corp")

# Chunk 2
Organization(address=Address(city="Paris"))

# Solution: Merged recursively
Organization(
    name="Acme Corp",
    address=Address(city="Paris")
)
```

---

## Performance Comparison

### Speed Benchmark

| Strategy | Models | Time | Throughput |
|:---------|:-------|:-----|:-----------|
| **Programmatic** | 5 | 0.01s | 500 merges/s |
| **LLM Consolidation** | 5 | 3s | 0.3 merges/s |
| **Programmatic** | 20 | 0.05s | 400 merges/s |
| **LLM Consolidation** | 20 | 8s | 0.1 merges/s |

### Accuracy Comparison

| Document Type | Programmatic | LLM Consolidation | Improvement |
|:--------------|:-------------|:------------------|:------------|
| **Simple invoice** | 95% | 96% | +1% |
| **Complex contract** | 88% | 94% | +6% |
| **Multi-page form** | 90% | 95% | +5% |
| **Rheology research** | 85% | 92% | +7% |

---

## Best Practices

### 👍 Use Programmatic by Default

```python
# ✅ Good - Fast and free
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    llm_consolidation=False  # Default (90% accuracy)
)
```

### 👍 Enable Standard LLM for High Accuracy

```python
# ✅ Good - Better accuracy for important documents
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    backend="llm",
    inference="local",
    model_override="llama3.1:8b",  # STANDARD tier
    llm_consolidation=True  # 95% accuracy
)
```

### 👍 Use Chain of Density for Critical Documents

```python
# ✅ Good - Highest accuracy for critical data
config = PipelineConfig(
    source="legal_contract.pdf",
    template="templates.LegalContract",
    backend="llm",
    inference="remote",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True  # Chain of Density (98% accuracy)
)
```

!!! warning "Cost Consideration"
    Chain of Density uses 3x more tokens than standard consolidation. For 100 documents:
    
    - Standard LLM: ~$10-20
    - Chain of Density: ~$30-60
    
    Use only when accuracy justifies the cost.

### 👍 Validate Merged Results

```python
# ✅ Good - Always validate
merged = merge_pydantic_models(models, template)

# Check completeness
if not merged.document_no:
    print("Warning: Missing invoice number")

if not merged.line_items:
    print("Warning: No line items")
```

### 👍 Handle Empty Model Lists

```python
# ✅ Good - Handle edge cases
if not models:
    print("No models to merge")
    merged = template()  # Empty model
else:
    merged = merge_pydantic_models(models, template)
```

---

## Troubleshooting

### 🐛 Duplicate Entities

**Solution:**
```python
# Deduplication is automatic
# If duplicates persist, check entity fields

# Ensure entities have stable identifiers
class Organization(BaseModel):
    name: str  # Used for deduplication
    address: Address | None = None
```

### 🐛 Lost Information

**Solution:**
```python
# Check if fields are being overwritten
# Use LLM consolidation for better merging

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    llm_consolidation=True  # Better preservation
)
```

### 🐛 Merge Validation Fails

**Solution:**
```python
# Check merged data structure
try:
    merged = merge_pydantic_models(models, template)
except ValidationError as e:
    print(f"Validation errors: {e.errors()}")
    
    # Inspect raw merged data
    dicts = [m.model_dump() for m in models]
    print(f"Raw data: {dicts}")
```

### 🐛 Slow Consolidation

**Solution:**
```python
# Disable LLM consolidation for speed
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    llm_consolidation=False  # Faster
)
```

---

## Advanced Techniques

### Custom Merge Logic

For special cases, implement custom merging:

```python
from docling_graph.core.utils import merge_pydantic_models

def custom_merge(models, template):
    """Custom merge with special rules."""
    
    # Start with programmatic merge
    base = merge_pydantic_models(models, template)
    
    # Apply custom logic
    if base.total is None and base.line_items:
        # Calculate total from line items
        base.total = sum(item.total for item in base.line_items)
    
    return base

# Use custom merge
merged = custom_merge(models, Invoice)
```

---

## Performance Comparison

### Consolidation Strategy Comparison

| Strategy | Time | Tokens | Accuracy | Cost (100 docs) | Best For |
|:---------|:-----|:-------|:---------|:----------------|:---------|
| **Programmatic** | 0.01s | 0 | 90% | $0 | Default, high volume |
| **LLM Standard** | 3s | 1500 | 95% | $15 | Important documents |
| **Chain of Density** | 10s | 4500 | 98% | $45 | Critical documents |

### When to Use Each

```python
# High volume, good accuracy needed
if document_count > 1000:
    llm_consolidation = False  # Programmatic

# Important documents, high accuracy needed
elif document_importance == "high":
    llm_consolidation = True  # Standard LLM
    model_override = "llama3.1:8b"  # STANDARD tier

# Critical documents, maximum accuracy needed
elif document_importance == "critical":
    llm_consolidation = True  # Chain of Density
    model_override = "gpt-4-turbo"  # ADVANCED tier
```

---

## Next Steps

Now that you understand model merging:

1. **[Model Capabilities →](model-capabilities.md)** - Learn about Chain of Density
2. **[Batch Processing →](batch-processing.md)** - Optimize chunk processing
3. **[Extraction Backends →](extraction-backends.md)** - Understand backends
4. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Optimize consolidation
5. **[Graph Management →](../graph-management/index.md)** - Work with knowledge graphs