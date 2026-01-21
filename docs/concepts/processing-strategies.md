# Processing Strategies

Docling Graph offers two processing strategies for handling multi-page documents: **One-to-One** and **Many-to-One**. Understanding these strategies is crucial for choosing the right approach for your use case.

## Overview

| Strategy | Pages → Models | Use Case | Output |
|:---------|:--------------|:---------|:-------|
| **One-to-One** | 1 page → 1 model | Page-specific analysis | N models (one per page) |
| **Many-to-One** | N pages → 1 model | Document-level extraction | 1 merged model |

## One-to-One Strategy

### Concept

The One-to-One strategy processes each page independently, producing a separate Pydantic model instance for each page.

**Location**: `docling_graph/core/extractors/strategies/one_to_one.py`

### How It Works

```
Page 1  →  Extract  →  Model 1
Page 2  →  Extract  →  Model 2
Page 3  →  Extract  →  Model 3
...
Page N  →  Extract  →  Model N

Result: [Model 1, Model 2, Model 3, ..., Model N]
```

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    processing_mode="one-to-one",  # Process each page separately
    backend="llm",
    output_dir="outputs"
)
```

### Characteristics

✅ **Independent Processing**: Each page is processed in isolation  
✅ **Page-Level Granularity**: Preserves page-specific information  
✅ **Parallel Potential**: Pages can be processed in parallel (future optimization)  
✅ **Simple Logic**: No merging or consolidation required  

⚠️ **Multiple Models**: Returns N models instead of 1  
⚠️ **No Cross-Page Context**: Cannot capture relationships across pages  
⚠️ **Redundant Data**: May duplicate information present on multiple pages  

### Ideal Use Cases

#### 1. Page-Specific Analysis

When each page contains distinct, independent information:

```python
# Example: Multi-page invoice batch
# Each page is a separate invoice
config = PipelineConfig(
    source="invoice_batch.pdf",
    template=Invoice,
    processing_mode="one-to-one",
    output_dir="outputs/invoices"
)
```

#### 2. Document Collections

When a single PDF contains multiple independent documents:

```python
# Example: Scanned ID cards
# Each page is a different person's ID
config = PipelineConfig(
    source="id_cards_batch.pdf",
    template=IDCard,
    processing_mode="one-to-one",
    output_dir="outputs/id_cards"
)
```

#### 3. Page-Level Metadata

When you need to track which page information came from:

```python
# Each model represents one page
# Useful for citation or reference tracking
models = extractor.extract(source, template)
for i, model in enumerate(models, 1):
    print(f"Page {i}: {model}")
```

### Example Output

```python
# Input: 3-page document
# Output: List of 3 models

[
    Invoice(invoice_number="INV-001", total=100.00, ...),
    Invoice(invoice_number="INV-002", total=250.00, ...),
    Invoice(invoice_number="INV-003", total=175.00, ...)
]
```

### Graph Construction

Each model becomes a separate subgraph:

```
Graph:
  Invoice_INV001 (from page 1)
    ├─ ISSUED_BY → Organization_A
    └─ SENT_TO → Customer_X
  
  Invoice_INV002 (from page 2)
    ├─ ISSUED_BY → Organization_A
    └─ SENT_TO → Customer_Y
  
  Invoice_INV003 (from page 3)
    ├─ ISSUED_BY → Organization_B
    └─ SENT_TO → Customer_Z
```

## Many-to-One Strategy

### Concept

The Many-to-One strategy processes all pages together, producing a single merged Pydantic model instance for the entire document.

**Location**: `docling_graph/core/extractors/strategies/many_to_one.py`

### How It Works

```
Page 1 ┐
Page 2 ├─→  Extract  →  Merge  →  Single Model
Page 3 ┘

Result: [Merged Model]
```

### Processing Modes

#### Without Chunking

```python
# Process entire document as one markdown
Full Markdown  →  Extract  →  Single Model
```

#### With Chunking

```python
# Split into chunks, extract from each, then merge
Chunk 1  →  Extract  →  Partial Model 1 ┐
Chunk 2  →  Extract  →  Partial Model 2 ├─→  Merge  →  Final Model
Chunk 3  →  Extract  →  Partial Model 3 ┘
```

### Configuration

#### Basic (No Chunking)

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    processing_mode="many-to-one",
    use_chunking=False,  # Process full document
    backend="llm",
    output_dir="outputs"
)
```

#### With Chunking (Recommended for Large Documents)

```python
config = PipelineConfig(
    source="large_document.pdf",
    template=YourTemplate,
    processing_mode="many-to-one",
    use_chunking=True,           # Enable chunking
    llm_consolidation=False,     # Programmatic merge (faster)
    backend="llm",
    output_dir="outputs"
)
```

#### With LLM Consolidation

```python
config = PipelineConfig(
    source="complex_document.pdf",
    template=YourTemplate,
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=True,  # Use LLM to intelligently merge
    backend="llm",
    output_dir="outputs"
)
```

### Characteristics

✅ **Single Model**: Returns one consolidated model  
✅ **Cross-Page Context**: Captures relationships across pages  
✅ **Deduplication**: Automatically merges duplicate information  
✅ **Document-Level View**: Represents the entire document holistically  

⚠️ **More Complex**: Requires merging logic  
⚠️ **Context Limits**: May need chunking for large documents  
⚠️ **Slower**: Additional consolidation step  

### Merging Strategies

#### 1. Programmatic Merge (Default)

Fast, rule-based merging:

- **Lists**: Concatenate and deduplicate
- **Scalars**: Keep first non-null value
- **Nested Objects**: Recursively merge

```python
# Automatic merging of extracted data
config = PipelineConfig(
    processing_mode="many-to-one",
    llm_consolidation=False  # Use programmatic merge
)
```

#### 2. LLM Consolidation

Intelligent, context-aware merging:

- Uses LLM to resolve conflicts
- Better handles semantic duplicates
- Slower but more accurate

```python
# LLM-based intelligent merging
config = PipelineConfig(
    processing_mode="many-to-one",
    llm_consolidation=True  # Use LLM to merge
)
```

### Ideal Use Cases

#### 1. Multi-Page Documents

When a document spans multiple pages with related content:

```python
# Example: Research paper
# Title on page 1, abstract on page 1-2, content on pages 2-10
config = PipelineConfig(
    source="research_paper.pdf",
    template=ResearchPaper,
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs/research"
)
```

#### 2. Narrative Documents

When information flows across pages:

```python
# Example: Insurance policy
# Coverage details span multiple pages
config = PipelineConfig(
    source="insurance_policy.pdf",
    template=InsurancePolicy,
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs/policies"
)
```

#### 3. Documents with Relationships

When entities and relationships span pages:

```python
# Example: Contract
# Parties on page 1, terms on pages 2-5, signatures on page 6
config = PipelineConfig(
    source="contract.pdf",
    template=Contract,
    processing_mode="many-to-one",
    use_chunking=True,
    output_dir="outputs/contracts"
)
```

### Example Output

```python
# Input: 10-page research paper
# Output: Single merged model

ResearchPaper(
    title="Advanced Battery Technology",
    authors=[
        Author(name="Dr. Smith", affiliation="MIT"),
        Author(name="Dr. Jones", affiliation="Stanford")
    ],
    abstract="This paper presents...",
    sections=[
        Section(title="Introduction", content="..."),
        Section(title="Methodology", content="..."),
        Section(title="Results", content="...")
    ],
    references=[...]
)
```

### Graph Construction

Single unified graph:

```
Graph:
  ResearchPaper_AdvancedBattery
    ├─ HAS_AUTHOR → Author_DrSmith
    ├─ HAS_AUTHOR → Author_DrJones
    ├─ HAS_SECTION → Section_Introduction
    ├─ HAS_SECTION → Section_Methodology
    └─ HAS_SECTION → Section_Results
  
  Author_DrSmith
    └─ AFFILIATED_WITH → Organization_MIT
  
  Author_DrJones
    └─ AFFILIATED_WITH → Organization_Stanford
```

## Chunking Deep Dive

### Why Chunking?

Large documents may exceed LLM context limits. Chunking splits the document into manageable pieces while preserving semantic coherence.

### Hybrid Chunking Strategy

Docling Graph uses a hybrid approach:

1. **Docling Segmentation**: Respects document structure (sections, tables, lists)
2. **Semantic Chunking**: Groups related content together
3. **Token-Aware**: Respects LLM context limits

```python
# Chunking configuration
config = PipelineConfig(
    use_chunking=True,
    # Chunking happens automatically based on:
    # - Document structure (from Docling)
    # - LLM context limit
    # - Semantic boundaries
)
```

### Chunk Processing

```
Document (50 pages)
    ↓
Docling Segmentation
    ↓
Semantic Chunks (respecting structure)
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Chunk 1 │ Chunk 2 │ Chunk 3 │ Chunk 4 │
│ (10 pg) │ (15 pg) │ (12 pg) │ (13 pg) │
└─────────┴─────────┴─────────┴─────────┘
    ↓         ↓         ↓         ↓
 Extract   Extract   Extract   Extract
    ↓         ↓         ↓         ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Model 1 │ Model 2 │ Model 3 │ Model 4 │
└─────────┴─────────┴─────────┴─────────┘
    └─────────┬─────────┬─────────┘
              ↓
         Consolidate
              ↓
        Final Model
```

### Consolidation Methods

#### Programmatic (Fast)

```python
config = PipelineConfig(
    use_chunking=True,
    llm_consolidation=False  # Default
)

# Merging rules:
# - Lists: Concatenate + deduplicate
# - Scalars: First non-null wins
# - Objects: Recursive merge
```

#### LLM-Based (Intelligent)

```python
config = PipelineConfig(
    use_chunking=True,
    llm_consolidation=True
)

# LLM receives:
# - All partial models
# - Original template schema
# - Consolidation prompt
# Returns: Intelligently merged model
```

## Choosing the Right Strategy

### Decision Tree

```
Is each page independent?
├─ Yes → One-to-One
│   └─ Examples: Invoice batches, ID card scans
│
└─ No → Many-to-One
    │
    ├─ Is document < 5 pages?
    │   └─ Yes → Many-to-One (no chunking)
    │
    └─ Is document ≥ 5 pages?
        └─ Yes → Many-to-One (with chunking)
            │
            ├─ Simple merging needed?
            │   └─ Yes → llm_consolidation=False
            │
            └─ Complex merging needed?
                └─ Yes → llm_consolidation=True
```

### Quick Reference

| Document Type | Strategy | Chunking | Consolidation |
|:-------------|:---------|:---------|:--------------|
| Invoice batch | One-to-One | N/A | N/A |
| ID card scan | One-to-One | N/A | N/A |
| Short report (1-5 pages) | Many-to-One | No | N/A |
| Research paper (10+ pages) | Many-to-One | Yes | Programmatic |
| Complex contract (20+ pages) | Many-to-One | Yes | LLM |
| Insurance policy (15+ pages) | Many-to-One | Yes | Programmatic |

## Performance Considerations

### One-to-One

- **Speed**: Fast (pages processed independently)
- **Memory**: Low (one page at a time)
- **Accuracy**: High (no merging errors)

### Many-to-One (No Chunking)

- **Speed**: Moderate (single extraction)
- **Memory**: High (entire document in context)
- **Accuracy**: High (full context available)

### Many-to-One (With Chunking)

- **Speed**: Moderate to Slow (multiple extractions + merge)
- **Memory**: Low to Moderate (chunks processed separately)
- **Accuracy**: Good (depends on consolidation method)

## Best Practices

### For One-to-One

1. **Verify Independence**: Ensure pages are truly independent
2. **Handle Duplicates**: Be prepared for duplicate entities across pages
3. **Page Tracking**: Consider adding page metadata to models

### For Many-to-One

1. **Enable Chunking**: For documents > 5 pages
2. **Choose Consolidation**: Programmatic for speed, LLM for quality
3. **Test Merging**: Verify merged output matches expectations
4. **Monitor Context**: Watch for context limit warnings

## Troubleshooting

### One-to-One Issues

**Problem**: Duplicate entities across pages  
**Solution**: Use graph deduplication or post-process models

**Problem**: Missing cross-page relationships  
**Solution**: Switch to Many-to-One strategy

### Many-to-One Issues

**Problem**: Context length exceeded  
**Solution**: Enable chunking with `use_chunking=True`

**Problem**: Poor merge quality  
**Solution**: Switch to `llm_consolidation=True`

**Problem**: Slow processing  
**Solution**: Use programmatic consolidation or reduce document size

## Next Steps

- Learn about [Graph Construction](graph-construction.md)
- Understand [Pydantic Templates](pydantic-templates.md)
- Explore [Extraction Backends](extraction-backends.md)