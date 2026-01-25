# Key Concepts

## Core Terminology

### Entity

An **entity** is a unique, identifiable object that you want to track individually in your knowledge graph.

**Characteristics**:
- Has a stable identity (defined by `graph_id_fields`)
- Represents real-world objects (people, organizations, documents)
- Tracked individually even if properties are similar

**Example**:
```python
class Person(BaseModel):
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name', 'date_of_birth']
    }
    name: str
    date_of_birth: str
    email: str
```

Two persons with the same name but different birth dates are **different entities**.

### Component

A **component** is a value object that is deduplicated by its content.

**Characteristics**:
- No unique identity (set `is_entity=False`)
- Represents shared values (addresses, amounts, measurements)
- Identical components share the same graph node

**Example**:
```python
class Address(BaseModel):
    model_config = {'is_entity': False'}
    street: str
    city: str
    postal_code: str
```

Two people at "123 Main St, Boston, 02101" share the **same Address node**.

### Node

A **node** is a vertex in the knowledge graph. Every Pydantic model instance becomes a node.

**Node Properties**:
- **ID**: Unique identifier (generated from `graph_id_fields` or content hash)
- **Type**: The Pydantic class name (e.g., "Person", "Organization")
- **Attributes**: All field values from the Pydantic model

**Example Node**:
```
ID: Person_JohnDoe_1990-01-15
Type: Person
Attributes:
  - name: "John Doe"
  - date_of_birth: "1990-01-15"
  - email: "john@example.com"
```

### Edge

An **edge** is a directed relationship between two nodes in the graph.

**Edge Properties**:
- **Source**: The node where the relationship starts
- **Target**: The node where the relationship ends
- **Label**: The relationship type (e.g., "ISSUED_BY", "SENT_TO")
- **Direction**: Edges are directional (A → B is different from B → A)

**Example Edge**:
```
Source: Document_INV001
Label: ISSUED_BY
Target: Organization_AcmeCorp
```

This represents: "Document INV001 was issued by Acme Corp"

### Graph

A **graph** is the complete network of nodes and edges representing your extracted knowledge.

**Graph Structure**:
```
Document_INV001
  ├─ ISSUED_BY → Organization_AcmeCorp
  ├─ SENT_TO → Person_JohnDoe
  └─ CONTAINS_LINE → LineItem_001
      └─ HAS_PRODUCT → Product_Widget
```

## Pydantic Templates

### What is a Template?

A **Pydantic template** is a Python class that serves three purposes:

1. **Extraction Schema**: Tells the LLM/VLM what data to extract
2. **Validation Rules**: Ensures data quality and consistency
3. **Graph Structure**: Defines how entities and relationships map to nodes and edges

### Template Example

```python
from pydantic import BaseModel, Field
from typing import List

def edge(label: str, **kwargs):
    """Helper to define graph relationships"""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class Organization(BaseModel):
    """An organization entity"""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['name']
    }
    
    name: str = Field(
        description="Legal name of the organization",
        examples=["Acme Corp", "Tech Solutions Ltd"]
    )
    
    tax_id: str = Field(
        description="Tax identification number",
        examples=["123456789", "FR12345678901"]
    )

class Invoice(BaseModel):
    """An invoice document"""
    model_config = {
        'is_entity': True,
        'graph_id_fields': ['invoice_number']
    }
    
    invoice_number: str = Field(
        description="Unique invoice identifier",
        examples=["INV-2024-001", "INV-123456"]
    )
    
    # This creates an edge in the graph
    issuer: Organization = edge(
        label="ISSUED_BY",
        description="Organization that issued this invoice"
    )
```

**Result**: When extracted, this creates:
- An `Invoice` node
- An `Organization` node
- An `ISSUED_BY` edge connecting them

## Extraction Backends

### VLM (Vision-Language Model)

**What**: Uses Docling's NuExtract models to extract data directly from document images.

**Best For**:
- Structured forms (invoices, ID cards, receipts)
- Documents with clear key-value pairs
- Small documents (1-3 pages)

**Characteristics**:
- Processes documents directly (no markdown conversion)
- Local inference only
- Fast for small documents
- Excellent for forms

### LLM (Large Language Model)

**What**: Uses language models to extract data from markdown/text representations.

**Best For**:
- Complex narratives (rheology researchs, reports)
- Large documents (5+ pages)
- Documents requiring deep understanding

**Characteristics**:
- Requires markdown conversion first
- Local (vLLM, Ollama) or remote (OpenAI, Mistral, Gemini, WatsonX)
- Supports chunking for large documents
- Better for complex extraction

## Processing Modes

### One-to-One

**What**: Process each page independently, producing one Pydantic model per page.

**When to Use**:
- Each page contains independent information
- Document is a batch of separate items (e.g., multiple invoices in one PDF)
- You need page-level granularity

**Example**:
```
3-page PDF with 3 invoices
→ 3 separate Invoice models
→ 3 separate subgraphs
```

### Many-to-One

**What**: Process all pages together, producing one merged Pydantic model for the entire document.

**When to Use**:
- Document spans multiple pages with related content
- Information flows across pages
- You want a document-level view

**Example**:
```
10-page rheology research
→ 1 merged ResearchPaper model
→ 1 unified graph
```

## Chunking

### Why Chunking?

Large documents may exceed LLM context limits. **Chunking** splits the document into manageable pieces while preserving semantic coherence.

### Hybrid Chunking Strategy

Docling Graph uses a sophisticated approach:

1. **Docling Segmentation**: Respects document structure (sections, tables, lists)
2. **Semantic Boundaries**: Groups related content together
3. **Token-Aware**: Respects LLM context limits
4. **Context Preservation**: Maintains coherence across chunks

### Consolidation

After extracting from multiple chunks, results must be merged:

**Programmatic Merge** (Fast):
- Lists: Concatenate and deduplicate
- Scalars: First non-null value wins
- Objects: Recursive merge

**LLM Consolidation** (Intelligent):
- Uses LLM to intelligently merge results
- Better handles semantic duplicates
- Slower but more accurate

## Graph Construction

### Node ID Generation

**For Entities** (with `graph_id_fields`):
```python
# Person with graph_id_fields=['name', 'dob']
Person(name="John Doe", dob="1990-01-15")
→ Node ID: "Person_JohnDoe_1990-01-15"
```

**For Components** (content-based):
```python
# Address with is_entity=False
Address(street="123 Main St", city="Boston")
→ Node ID: "Address_{content_hash}"
```

### Deduplication

**Entities**: Deduplicated by `graph_id_fields`
- Same ID fields → Same node

**Components**: Deduplicated by content
- Same field values → Same node

### Example Graph

Given these models:

```python
author1 = Author(name="Dr. Smith")
author2 = Author(name="Dr. Jones")
paper = Paper(
    title="Advanced AI",
    authors=[author1, author2]
)
```

Resulting graph:

```
Paper_AdvancedAI
  ├─ HAS_AUTHOR → Author_DrSmith
  └─ HAS_AUTHOR → Author_DrJones
```

## Export Formats

### CSV
- **Purpose**: Neo4j bulk import
- **Files**: `nodes.csv`, `edges.csv`
- **Best For**: Production database loading

### Cypher
- **Purpose**: Neo4j script execution
- **Files**: `document_graph.cypher`
- **Best For**: Development and incremental updates

### JSON
- **Purpose**: General-purpose data exchange
- **Files**: `graph.json`
- **Best For**: API integration, archival

### HTML
- **Purpose**: Interactive visualization
- **Files**: `graph.html`
- **Best For**: Exploration and presentation

## Next Steps

Now that you understand the core concepts:

1. **[Use Cases](use-cases.md)** - See domain-specific examples
2. **[Architecture](architecture.md)** - Understand system design
3. **[Installation](../fundamentals/installation/index.md)** - Set up your environment