# Model Configuration


## Overview

Model configuration determines which AI model processes your documents. Docling Graph supports multiple providers for both local and remote inference, giving you flexibility in choosing the right model for your needs.

**In this guide:**
- Local vs remote inference
- Supported providers and models
- Model selection strategies
- Provider-specific configuration
- Performance and cost considerations

---

## Local vs Remote Inference

### Quick Comparison

| Aspect | Local Inference | Remote Inference |
|:-------|:---------------|:-----------------|
| **Location** | Your GPU/CPU | Cloud API |
| **Setup** | Complex (GPU drivers, models) | Simple (API key) |
| **Cost** | Hardware + electricity | Pay per token |
| **Speed** | Fast (with GPU) | Variable (network dependent) |
| **Privacy** | Complete | Data sent to provider |
| **Offline** | Yes | No |
| **Models** | Limited by hardware | Latest models available |

---

## Local Inference

### Overview

Local inference runs models on your own hardware (GPU or CPU). Best for privacy, offline use, and high-volume processing.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="local",  # Local inference
    model_override="ibm-granite/granite-4.0-1b",
    provider_override="vllm"
)
```

### Supported Local Providers

#### 1. vLLM (Recommended for LLM)

**Best for:** Fast local LLM inference with GPU

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b",
    provider_override="vllm"
)
```

**Setup:**
```bash
# Install vLLM
uv add vllm

# Start vLLM server
uv run python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.0-1b \
    --port 8000
```

**Supported Models:**
- `ibm-granite/granite-4.0-1b` (default, fast)
- `ibm-granite/granite-4.0-3b` (balanced)
- `meta-llama/Llama-3.1-8B` (high quality)
- Any HuggingFace model compatible with vLLM

#### 2. Ollama

**Best for:** Easy local setup, multiple models

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    model_override="llama-3.1-8b",
    provider_override="ollama"
)
```

**Setup:**
```bash
# Install Ollama (see ollama.ai)
# Pull model
ollama pull llama3.1:8b

# Ollama runs automatically on localhost:11434
```

**Supported Models:**
- `llama3.1:8b` (recommended)
- `mistral:7b`
- `mixtral:8x7b`
- Any model in Ollama library

#### 3. Docling VLM (For VLM Backend)

**Best for:** Vision-based extraction

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="vlm",
    inference="local",
    model_override="numind/NuExtract-2.0-8B",
    provider_override="docling"
)
```

**Supported Models:**
- `numind/NuExtract-2.0-8B` (default, recommended)
- `numind/NuExtract-2.0-2B` (faster, less accurate)

### Local Inference Requirements

#### Hardware Requirements

**Minimum (CPU only):**
- 16GB RAM
- 50GB disk space
- Slow processing

**Recommended (GPU):**
- NVIDIA GPU with 8GB+ VRAM
- 32GB RAM
- 100GB disk space
- CUDA 12.1+

**Optimal (GPU):**
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100)
- 64GB RAM
- 200GB SSD
- CUDA 12.1+

#### Software Requirements

```bash
# CUDA drivers (for GPU)
nvidia-smi  # Verify CUDA installation

# Python packages
uv add vllm  # For vLLM
# or
# Install Ollama from ollama.ai
```

**See:** [Installation: GPU Setup](../installation/gpu-setup.md)

---

## Remote Inference

### Overview

Remote inference uses cloud API providers. Best for quick setup, latest models, and no hardware requirements.

### Configuration

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",  # Remote inference
    model_override="gpt-4-turbo",
    provider_override="openai"
)
```

### Supported Remote Providers

#### 1. OpenAI

**Best for:** Highest quality, latest models

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="gpt-4-turbo",
    provider_override="openai"
)
```

**Setup:**
```bash
# Set API key
export OPENAI_API_KEY="your-api-key"
```

**Supported Models:**
- `gpt-4-turbo` (recommended, best quality)
- `gpt-4` (high quality)
- `gpt-3.5-turbo` (fast, economical)

**Pricing (approximate):**
- GPT-4 Turbo: $0.01/1K input tokens, $0.03/1K output tokens
- GPT-3.5 Turbo: $0.0005/1K input tokens, $0.0015/1K output tokens

#### 2. Mistral AI

**Best for:** European provider, good balance

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="mistral-small-latest",
    provider_override="mistral"
)
```

**Setup:**
```bash
# Set API key
export MISTRAL_API_KEY="your-api-key"
```

**Supported Models:**
- `mistral-small-latest` (default, economical)
- `mistral-medium-latest` (balanced)
- `mistral-large-latest` (highest quality)

**Pricing (approximate):**
- Small: $0.001/1K tokens
- Medium: $0.0027/1K tokens
- Large: $0.008/1K tokens

#### 3. Google Gemini

**Best for:** Multimodal capabilities, competitive pricing

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="gemini-2.5-flash",
    provider_override="gemini"
)
```

**Setup:**
```bash
# Set API key
export GOOGLE_API_KEY="your-api-key"
```

**Supported Models:**
- `gemini-2.5-flash` (default, fast)
- `gemini-2.0-pro` (high quality)

**Pricing (approximate):**
- Flash: $0.00025/1K input tokens, $0.00075/1K output tokens
- Pro: $0.00125/1K input tokens, $0.005/1K output tokens

#### 4. IBM WatsonX

**Best for:** Enterprise deployments, IBM ecosystem

```python
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="ibm/granite-13b-chat-v2",
    provider_override="watsonx"
)
```

**Setup:**
```bash
# Set API key and project ID
export WATSONX_API_KEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
```

**See:** [API Keys Setup](../installation/api-keys.md) for WatsonX configuration details.

---

## Model Selection Strategies

### By Document Complexity

```python
def get_model_config(document_complexity: str):
    """Choose model based on document complexity."""
    if document_complexity == "simple":
        return {
            "inference": "local",
            "model_override": "ibm-granite/granite-4.0-1b",
            "provider_override": "vllm"
        }
    elif document_complexity == "medium":
        return {
            "inference": "local",
            "model_override": "llama3.1:8b",
            "provider_override": "ollama"
        }
    else:
        return {
            "inference": "remote",
            "model_override": "gpt-4-turbo",
            "provider_override": "openai"
        }
```

### By Volume

```python
def get_model_config(document_count: int):
    """Choose model based on processing volume."""
    if document_count < 100:
        # Low volume: use best quality
        return {
            "inference": "remote",
            "model_override": "gpt-4-turbo",
            "provider_override": "openai"
        }
    elif document_count < 1000:
        # Medium volume: balanced
        return {
            "inference": "remote",
            "model_override": "mistral-small-latest",
            "provider_override": "mistral"
        }
    else:
        # High volume: use local to avoid costs
        return {
            "inference": "local",
            "model_override": "ibm-granite/granite-4.0-1b",
            "provider_override": "vllm"
        }
```

### By Budget

```python
def get_model_config(budget: str):
    """Choose model based on budget."""
    if budget == "minimal":
        # Minimal cost: local inference
        return {
            "inference": "local",
            "model_override": "ibm-granite/granite-4.0-1b",
            "provider_override": "vllm"
        }
    elif budget == "moderate":
        # Moderate cost: economical API
        return {
            "inference": "remote",
            "model_override": "mistral-small-latest",
            "provider_override": "mistral"
        }
    else:
        # No budget constraint: best quality
        return {
            "inference": "remote",
            "model_override": "gpt-4-turbo",
            "provider_override": "openai"
        }
```


### By Quality Requirements

```python
def get_model_by_quality(quality_requirement: str):
    """Choose model based on quality requirements."""
    if quality_requirement == "acceptable":
        return {
            "inference": "local",
            "model_override": "ibm-granite/granite-4.0-1b",
            "provider_override": "vllm"
        }
    elif quality_requirement == "high":
        return {
            "inference": "local",
            "model_override": "llama3.1:8b",
            "provider_override": "ollama"
        }
    else:  # critical
        return {
            "inference": "remote",
            "model_override": "gpt-4-turbo",
            "provider_override": "openai",
        }
```

---

## Provider-Specific Configuration

### vLLM Configuration

```python
# Custom vLLM base URL
import os
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    provider_override="vllm"
)
```

### Ollama Configuration

```python
# Custom Ollama base URL
import os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    provider_override="ollama"
)
```

### API Key Configuration

```bash
# Set via environment variables (recommended)
export OPENAI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export WATSONX_API_KEY="your-key"
export WATSONX_PROJECT_ID="your-project-id"
```

Or via `.env` file:
```bash
# .env file
OPENAI_API_KEY=your-key
MISTRAL_API_KEY=your-key
GOOGLE_API_KEY=your-key
```

**See:** [Installation: API Keys](../installation/api-keys.md)

---

## Performance Comparison

### Speed Comparison

```
Document: 10-page invoice PDF

Local (vLLM, GPU):        ~30 seconds
Local (Ollama, GPU):      ~45 seconds
Remote (GPT-3.5):         ~40 seconds
Remote (GPT-4):           ~60 seconds
Remote (Mistral Small):   ~35 seconds
```

### Quality Comparison

```
Extraction Accuracy (Complex Documents):

GPT-4 Turbo:              97%
GPT-3.5 Turbo:            92%
Mistral Large:            95%
Mistral Small:            90%
Granite 4.0-1B (local):   88%
Llama 3.1-8B (local):     93%
```

### Cost Comparison

```
Processing 1000 documents (10 pages each):

Local (vLLM):             $0 (GPU amortized)
Local (Ollama):           $0 (GPU amortized)
Remote (GPT-4):           $150-300
Remote (GPT-3.5):         $10-20
Remote (Mistral Small):   $5-15
Remote (Gemini Flash):    $3-10
```

---

## Troubleshooting

### Local Inference Issues

#### 🐛 CUDA Out of Memory

```python
# Solution: Use smaller model
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b",  # Smaller model
    provider_override="vllm"
)
```

#### 🐛 vLLM Server Not Running

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Start server if needed
uv run python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.0-1b \
    --port 8000
```

### Remote Inference Issues

#### 🐛 API Key Not Found

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Set if missing
export OPENAI_API_KEY="your-key"
```

#### 🐛 Rate Limit Exceeded

```python
# Solution: Add retry logic or switch provider
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="mistral-small-latest",  # Different provider
    provider_override="mistral"
)
```

---

## Best Practices

### 👍 Start with Remote for Testing

```python
# ✅ Good - Quick setup for testing
config = PipelineConfig(
    source="test.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override="gpt-3.5-turbo"
)
```

### 👍 Use Local for Production Volume

```python
# ✅ Good - Cost-effective for high volume
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b"
)
```

### 👍 Match Model to Document Complexity

```python
# ✅ Good - Use appropriate model
if document_is_complex:
    model = "gpt-4-turbo"
else:
    model = "gpt-3.5-turbo"

config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="remote",
    model_override=model
)
```

### 👍 Monitor Costs

```python
# ✅ Good - Track API usage
import logging

logging.info(f"Processing {document_count} documents")
logging.info(f"Estimated cost: ${estimated_cost}")

run_pipeline(config)
```

---

## Model Recommendations by Use Case

### High-Volume Processing

```python
# Use SIMPLE tier for speed and cost efficiency
config = PipelineConfig(
    source="document.pdf",
    template="templates.BillingDocument",
    inference="local",
    model_override="ibm-granite/granite-4.0-1b",  # SIMPLE tier
    provider_override="vllm",
    use_chunking=True,
)
```

**Benefits:**

- 🔵 Good Accuracy
- ⚡ Fast Processing

### Critical Documents

```python
# Use ADVANCED tier for maximum accuracy
config = PipelineConfig(
    source="contract.pdf",
    template="templates.Contract",
    inference="remote",
    model_override="gpt-4-turbo",  # ADVANCED tier
    provider_override="openai",
    use_chunking=True,
)
```

**Benefits:**

- 🟢 High Accuracy
- 🌀 Multi-turn consolidation

### Balanced Approach

```python
# Use STANDARD tier for general documents
config = PipelineConfig(
    source="document.pdf",
    template="templates.Report",
    inference="local",
    model_override="llama3.1:8b",  # STANDARD tier
    provider_override="ollama",
    use_chunking=True,
)
```

**Benefits:**

- 🔵 Good Accuracy
- ⚖️ Good Balance of Speed and Quality

---

## Next Steps

Now that you understand model configuration:

1. **[Staged Extraction →](../extraction-process/staged-extraction.md)** - Multi-pass extraction
2. **[Processing Modes →](processing-modes.md)** - Choose processing strategy
3. **[Configuration Examples](configuration-examples.md)** - See complete scenarios
4. **[Extraction Process](../extraction-process/index.md)** - Understand extraction
5. **[Performance Tuning →](../../usage/advanced/performance-tuning.md)** - Optimize performance