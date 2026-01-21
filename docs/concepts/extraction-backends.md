# Extraction Backends

Docling Graph supports two families of extraction backends: **Vision-Language Models (VLM)** and **Large Language Models (LLM)**. Each has distinct characteristics, strengths, and ideal use cases.

## Overview

| Aspect | VLM Backend | LLM Backend |
|:-------|:------------|:------------|
| **Input** | Raw documents (PDF, images) | Markdown/text |
| **Processing** | Direct visual understanding | Text-based extraction |
| **Inference** | Local only | Local or remote |
| **Best For** | Structured forms, key-value pairs | Complex documents, narratives |
| **Speed** | Fast for small documents | Varies by provider |
| **Context** | Limited to page/document | Supports chunking strategies |

## VLM Backend

### Overview

The VLM backend uses Docling's NuExtract models to extract structured information directly from document images or PDFs.

**Location**: `docling_graph/core/extractors/backends/vlm_backend.py`

### How It Works

1. **Direct Processing**: Processes documents without markdown conversion
2. **Visual Understanding**: Leverages vision-language models to understand layout and content
3. **Schema-Guided**: Uses Pydantic schema to guide extraction
4. **Page-Level**: Processes one page at a time

### Supported Models

- `numind/NuExtract-2.0-8B` (default, more accurate)
- `numind/NuExtract-2.0-2B` (faster, less accurate)

### Configuration

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="vlm",           # Use VLM backend
    inference="local",       # Only local supported
    output_dir="outputs"
)
```

### Advantages

✅ **Fast for Small Documents**: Efficient for single-page or few-page documents  
✅ **No Markdown Conversion**: Direct visual processing  
✅ **Good for Forms**: Excellent at extracting key-value pairs from structured layouts  
✅ **Local Inference**: No API costs or data privacy concerns  

### Limitations

⚠️ **Local Only**: No remote inference option  
⚠️ **Limited Context**: Processes pages independently  
⚠️ **Structured Documents**: Best for forms, not complex narratives  
⚠️ **GPU Required**: Needs GPU for reasonable performance  

### Ideal Use Cases

- **ID Cards**: Extracting name, DOB, ID number from identity documents
- **Invoices**: Structured invoice data with clear fields
- **Forms**: Application forms, registration documents
- **Receipts**: Simple receipt data extraction
- **Certificates**: Structured certificate information

### Example

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.id_card import IDCard

config = PipelineConfig(
    source="id_card.jpg",
    template=IDCard,
    backend="vlm",
    processing_mode="one-to-one",  # Process each page separately
    output_dir="outputs/id_cards"
)

run_pipeline(config)
```

## LLM Backend

### Overview

The LLM backend uses large language models to extract structured information from markdown/text representations of documents.

**Location**: `docling_graph/core/extractors/backends/llm_backend.py`

### How It Works

1. **Markdown Conversion**: Docling converts document to markdown first
2. **Text Processing**: LLM processes markdown content
3. **Schema-Guided**: Uses Pydantic schema with detailed field descriptions
4. **Chunking Support**: Can split large documents into manageable chunks
5. **Consolidation**: Merges extracted data from multiple chunks

### Supported Providers

#### Local Providers

- **vLLM**: High-performance local inference server
  - Requires GPU
  - Excellent for batch processing
  - Default model: `ibm-granite/granite-4.0-1b`

- **Ollama**: Easy local model management
  - CPU or GPU
  - Simple setup
  - Default model: `llama-3.1-8b`

#### Remote Providers

- **Mistral AI**: Fast, cost-effective
  - Default model: `mistral-small-latest`
  - Good balance of speed and quality

- **OpenAI**: High quality, widely used
  - Default model: `gpt-4-turbo`
  - Excellent for complex extraction

- **Google Gemini**: Competitive pricing
  - Default model: `gemini-2.5-flash`
  - Fast inference

- **IBM WatsonX**: Enterprise-grade
  - Granite, Llama, Mixtral models
  - On-premises deployment options

### Configuration

#### Local Inference (vLLM)

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="local",
    provider_override="vllm",
    model_override="ibm-granite/granite-4.0-1b",
    use_chunking=True,
    output_dir="outputs"
)
```

#### Remote Inference (OpenAI)

```python
config = PipelineConfig(
    source="document.pdf",
    template=YourTemplate,
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",
    use_chunking=True,
    llm_consolidation=True,  # Use LLM to merge chunks
    output_dir="outputs"
)
```

### Advantages

✅ **Complex Documents**: Handles narratives, research papers, reports  
✅ **Large Context**: Supports chunking for documents of any size  
✅ **Flexible Inference**: Local or remote options  
✅ **Multiple Providers**: Choose based on cost, speed, quality  
✅ **Rich Descriptions**: Leverages detailed field descriptions for better extraction  

### Limitations

⚠️ **Requires Markdown**: Needs Docling conversion first  
⚠️ **API Costs**: Remote inference incurs costs  
⚠️ **Slower**: Generally slower than VLM for small documents  
⚠️ **Context Limits**: Must chunk very large documents  

### Ideal Use Cases

- **Research Papers**: Extracting methodology, results, conclusions
- **Legal Documents**: Complex contracts with nested clauses
- **Technical Reports**: Multi-section documents with relationships
- **Insurance Policies**: Detailed coverage terms and conditions
- **Medical Records**: Narrative clinical notes

### Chunking Strategy

For large documents, the LLM backend uses a hybrid chunking approach:

```python
config = PipelineConfig(
    source="large_document.pdf",
    template=YourTemplate,
    backend="llm",
    use_chunking=True,           # Enable chunking
    llm_consolidation=False,     # Programmatic merge (faster)
    processing_mode="many-to-one",
    output_dir="outputs"
)
```

**Chunking Process**:

1. Docling segments document (sections, tables, lists)
2. Semantic chunking respects document structure
3. Each chunk processed independently
4. Results merged programmatically or via LLM

**Consolidation Options**:

- `llm_consolidation=False`: Fast programmatic merge (recommended)
- `llm_consolidation=True`: LLM-based merge (more intelligent, slower)

### Example: Research Paper

```python
from docling_graph import run_pipeline, PipelineConfig
from templates.research import ResearchPaper

config = PipelineConfig(
    source="research_paper.pdf",
    template=ResearchPaper,
    backend="llm",
    inference="remote",
    provider_override="mistral",
    model_override="mistral-medium-latest",
    processing_mode="many-to-one",
    use_chunking=True,
    llm_consolidation=False,
    output_dir="outputs/research"
)

run_pipeline(config)
```

## Choosing the Right Backend

### Decision Matrix

#### Document Type

| Document Type        | Examples                  | Recommended Backend |
|----------------------|---------------------------|---------------------|
| Structured Forms     | ID cards, invoices        | VLM                 |
| Complex Narratives   | Research, reports         | LLM                 |
| Mixed Content        | Policies, contracts       | LLM                 |

#### Document Size

| Size                 | Recommendation            |
|----------------------|---------------------------|
| Single Page          | VLM                       |
| Few Pages (2–5)      | VLM or LLM                |
| Many Pages (5+)      | LLM (with chunking)       |

#### Infrastructure Constraints

| Constraint           | Recommended Backend       |
|----------------------|---------------------------|
| GPU Available        | VLM or Local LLM          |
| No GPU               | Remote LLM                |
| Privacy Concerns     | Local VLM or LLM          |
| Cost Sensitive       | Local or Mistral          |


### Quick Selection Guide

**Use VLM when**:
- Document is 1-3 pages
- Content is structured (forms, tables)
- You have GPU available
- You need fast processing
- Privacy is critical (local only)

**Use LLM when**:
- Document is complex or narrative
- Document is large (5+ pages)
- You need flexible inference options
- Content requires deep understanding
- You want to leverage remote APIs

## Performance Comparison

### Speed

| Backend | Small Doc (1-2 pages) | Large Doc (10+ pages) |
|:--------|:---------------------|:---------------------|
| VLM     | ⚡ Fast (seconds)    | ⚠️ Slow (page-by-page) |
| LLM (Local) | 🐢 Moderate (10-30s) | 🐢 Moderate (chunked) |
| LLM (Remote) | ⚡ Fast (5-15s) | ⚡ Fast (parallel chunks) |

### Accuracy

| Backend | Structured Forms | Complex Narratives |
|:--------|:----------------|:-------------------|
| VLM     | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| LLM     | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐⭐ Excellent |

### Cost

| Backend | Setup Cost | Runtime Cost |
|:--------|:-----------|:-------------|
| VLM     | GPU required | Free |
| LLM (Local) | GPU recommended | Free |
| LLM (Remote) | None | API charges |

## Advanced Configuration

### Custom Model Selection

```python
# Override default models
config = PipelineConfig(
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4o",  # Use specific model
    output_dir="outputs"
)
```

### Provider-Specific Settings

```yaml
# config.yaml
models:
  llm:
    remote:
      providers:
        openai:
          default_model: "gpt-4-turbo"
        mistral:
          default_model: "mistral-large-latest"
```

## Troubleshooting

### VLM Issues

**Problem**: Out of memory errors  
**Solution**: Reduce batch size or use smaller model (2B instead of 8B)

**Problem**: Slow processing  
**Solution**: Ensure GPU is available and CUDA is properly configured

### LLM Issues

**Problem**: Context length exceeded  
**Solution**: Enable chunking with `use_chunking=True`

**Problem**: API rate limits  
**Solution**: Add delays between requests or use local inference

**Problem**: Poor extraction quality  
**Solution**: Improve Pydantic template descriptions and examples

## Next Steps

- Learn about [Processing Strategies](processing-strategies.md)
- Understand [Pydantic Templates](pydantic-templates.md)
- Explore [Configuration System](configuration.md)