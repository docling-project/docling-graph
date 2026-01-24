# Model Capabilities

## Overview

Docling-Graph automatically classifies models into **three capability tiers** to optimize extraction quality and performance. This intelligent system adapts prompts, consolidation strategies, and processing approaches based on model size and capabilities.

**What You'll Learn:**
- Model capability tier system
- Automatic model detection
- Adaptive behavior per tier
- Performance implications
- Supported models by tier

---

## Model Capability Tiers

### Tier Classification

| Tier | Model Size | Characteristics | Use Cases |
|------|-----------|-----------------|-----------|
| **SIMPLE** | 1B-7B params | Fast, basic understanding | Simple forms, invoices, quick extraction |
| **STANDARD** | 7B-13B params | Balanced speed/accuracy | General documents, contracts |
| **ADVANCED** | 13B+ params | High accuracy, complex reasoning | Research papers, legal documents, complex analysis |

---

## Automatic Detection

Models are automatically classified based on their name and known characteristics:

```python
from docling_graph.llm_clients.config import detect_model_capability, ModelCapability

# Automatic detection examples
capability = detect_model_capability("llama-3.1-8b")
print(capability)  # ModelCapability.STANDARD

capability = detect_model_capability("gpt-4-turbo")
print(capability)  # ModelCapability.ADVANCED

capability = detect_model_capability("granite-4.0-1b")
print(capability)  # ModelCapability.SIMPLE
```

### Detection Logic

The system uses pattern matching on model names:

1. **Size-based detection**: Extracts parameter count from model name (e.g., "8b", "13b")
2. **Known model mapping**: Uses pre-classified list of 40+ popular models
3. **Fallback heuristics**: Conservative defaults for unknown models

---

## Adaptive Behavior

The extraction system automatically adapts based on model capability:

### SIMPLE Models (1B-7B)

**Optimized for Speed**

- ✅ **Minimal Instructions**: Focused, concise prompts
- ✅ **Basic Consolidation**: Simple programmatic merging
- ✅ **Fast Processing**: Optimized for throughput
- ✅ **Lower Memory**: Efficient resource usage

**Best For:**
- Simple forms and invoices
- Structured data extraction
- High-volume processing
- Resource-constrained environments

**Example Models:**
- `ibm-granite/granite-4.0-1b`
- `meta-llama/Llama-3.2-1B`
- `numind/NuExtract-2.0-2B`

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="invoice.pdf",
    template="templates.Invoice",
    backend="llm",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b"  # SIMPLE tier
)
config.run()
```

---

### STANDARD Models (7B-13B)

**Balanced Performance**

- ✅ **Balanced Instructions**: Moderate detail level
- ✅ **Standard Consolidation**: Programmatic + optional LLM
- ✅ **Good Accuracy**: Reliable for most documents
- ✅ **Reasonable Speed**: Good throughput

**Best For:**
- General business documents
- Multi-page contracts
- Standard extraction tasks
- Production workloads

**Example Models:**
- `meta-llama/Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.1`
- `numind/NuExtract-2.0-8B`

```python
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    backend="llm",
    inference="local",
    model_override="meta-llama/Llama-3.1-8B"  # STANDARD tier
)
config.run()
```

---

### ADVANCED Models (13B+)

**Maximum Accuracy**

- ✅ **Detailed Instructions**: Comprehensive prompts
- ✅ **Chain of Density**: Multi-turn consolidation
- ✅ **High Accuracy**: Best extraction quality
- ✅ **Complex Reasoning**: Handles nuanced content

**Best For:**
- Research papers
- Legal documents
- Complex technical content
- High-accuracy requirements

**Example Models:**
- `gpt-4-turbo` (OpenAI)
- `claude-3.5-sonnet` (Anthropic)
- `gemini-2.5-flash` (Google)
- `mistral-large-latest` (Mistral)

```python
config = PipelineConfig(
    source="research_paper.pdf",
    template="templates.ResearchPaper",
    backend="llm",
    inference="remote",
    provider_override="openai",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True  # Enable Chain of Density
)
config.run()
```

---

## Chain of Density Consolidation

**ADVANCED models** automatically use Chain of Density consolidation when `llm_consolidation=True`:

### Three-Step Refinement

1. **Initial Extraction**: Extract from raw document chunks
2. **Refinement**: Merge with programmatic consolidation
3. **Final Polish**: LLM refines for consistency and completeness

```python
# Chain of Density is automatic for ADVANCED models
config = PipelineConfig(
    source="complex_document.pdf",
    template="templates.ComplexTemplate",
    backend="llm",
    inference="remote",
    model_override="gpt-4-turbo",  # ADVANCED tier
    llm_consolidation=True,  # Enables Chain of Density
    processing_mode="many-to-one"
)
config.run()
```

### When to Use Chain of Density

✅ **Use When:**
- Document has conflicting information
- Need highest accuracy
- Complex narrative content
- Using ADVANCED tier models

❌ **Skip When:**
- Simple structured data
- Speed is critical
- Using SIMPLE/STANDARD models
- Budget constraints

---

## Supported Models

### Complete Model List

See the full list of classified models in [`models.yaml`](https://github.com/IBM/docling-graph/blob/main/docling_graph/llm_clients/models.yaml).

#### SIMPLE Tier (1B-7B)

**Local Models:**
- `ibm-granite/granite-4.0-1b`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-3B`
- `numind/NuExtract-2.0-2B`
- `Qwen/Qwen2.5-1.5B`

#### STANDARD Tier (7B-13B)

**Local Models:**
- `meta-llama/Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.1`
- `numind/NuExtract-2.0-8B`
- `Qwen/Qwen2.5-7B`

**Remote APIs:**
- `mistral-small-latest` (Mistral)
- `gemini-1.5-flash` (Google)

#### ADVANCED Tier (13B+)

**Remote APIs:**
- `gpt-4-turbo` (OpenAI)
- `gpt-4o` (OpenAI)
- `claude-3.5-sonnet` (Anthropic)
- `claude-3-opus` (Anthropic)
- `gemini-2.5-flash` (Google)
- `gemini-2.5-pro` (Google)
- `mistral-large-latest` (Mistral)
- `mistral-medium-latest` (Mistral)

**IBM WatsonX:**
- `ibm/granite-13b-chat-v2`
- `meta-llama/llama-3-1-70b-instruct`

---

## Performance Implications

### Speed vs Accuracy Trade-off

| Tier | Speed | Accuracy | Memory | Cost |
|------|-------|----------|--------|------|
| **SIMPLE** | ⚡⚡⚡ Very Fast | 🟡 Moderate | 2-4 GB | $ Low |
| **STANDARD** | ⚡⚡ Fast | 🟢 Good | 8-16 GB | $$ Medium |
| **ADVANCED** | ⚡ Moderate | 💎 Excellent | 16-32 GB | $$$ High |

### Benchmark Results

**Document**: 10-page contract

| Model Tier | Time | Accuracy | Tokens Used |
|-----------|------|----------|-------------|
| SIMPLE (1B) | 15s | 85% | 2,500 |
| STANDARD (8B) | 45s | 92% | 3,200 |
| ADVANCED (GPT-4) | 90s | 97% | 4,100 |

---

## Best Practices

### 1. Match Tier to Task Complexity

```python
# ✅ Good - Simple task, simple model
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.Invoice",
    model_override="granite-4.0-1b"  # SIMPLE tier
)

# ✅ Good - Complex task, advanced model
config = PipelineConfig(
    source="research_paper.pdf",
    template="templates.ResearchPaper",
    model_override="gpt-4-turbo"  # ADVANCED tier
)

# ❌ Avoid - Overkill
config = PipelineConfig(
    source="simple_form.pdf",
    template="templates.SimpleForm",
    model_override="gpt-4-turbo"  # Unnecessary
)
```

### 2. Start Small, Scale Up

```python
# Start with SIMPLE tier
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="granite-4.0-1b"
)
result = config.run()

# If accuracy insufficient, upgrade to STANDARD
if accuracy_check(result) < 0.90:
    config.model_override = "llama-3.1-8b"
    result = config.run()

# If still insufficient, upgrade to ADVANCED
if accuracy_check(result) < 0.95:
    config.model_override = "gpt-4-turbo"
    config.llm_consolidation = True
    result = config.run()
```

### 3. Use Chain of Density Wisely

```python
# ✅ Good - Complex document with ADVANCED model
config = PipelineConfig(
    source="complex_legal.pdf",
    template="templates.LegalDocument",
    model_override="gpt-4-turbo",  # ADVANCED
    llm_consolidation=True  # Enable Chain of Density
)

# ❌ Avoid - Chain of Density with SIMPLE model
config = PipelineConfig(
    source="invoice.pdf",
    template="templates.Invoice",
    model_override="granite-4.0-1b",  # SIMPLE
    llm_consolidation=True  # Won't use Chain of Density
)
```

---

## Troubleshooting

### Issue: Model Not Detected Correctly

**Solution:**
```python
# Override automatic detection
from docling_graph.llm_clients.config import ModelCapability

# Force specific capability
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="custom-model-7b",
    # Note: Capability override not directly exposed
    # Model will be detected based on name patterns
)
```

### Issue: Poor Accuracy with SIMPLE Model

**Solution:**
```python
# Upgrade to STANDARD or ADVANCED tier
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="llama-3.1-8b"  # STANDARD tier
)
```

### Issue: Slow Processing with ADVANCED Model

**Solution:**
```python
# Try STANDARD tier first
config = PipelineConfig(
    source="document.pdf",
    template="templates.MyTemplate",
    model_override="mistral-small-latest"  # STANDARD tier
)
```

---

## Next Steps

Now that you understand model capabilities:

1. **[Extraction Backends →](extraction-backends.md)** - Learn about LLM and VLM backends
2. **[Model Configuration →](../pipeline-configuration/model-configuration.md)** - Configure model settings
3. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Optimize for your use case

---

## Related Documentation

- **[Model Configuration](../pipeline-configuration/model-configuration.md)** - Detailed model settings
- **[Extraction Backends](extraction-backends.md)** - Backend selection guide
- **[Performance Tuning](../../usage/advanced/performance-tuning.md)** - Optimization strategies
- **[Model Merging](model-merging.md)** - Consolidation strategies