# IBM WatsonX Integration Guide

This guide explains how to use IBM WatsonX foundation models with docling-graph for document extraction and knowledge graph generation.

## Overview

IBM WatsonX provides access to powerful foundation models including:
- **IBM Granite models**: Optimized for enterprise use cases
- **Meta Llama models**: High-performance open models
- **Mistral models**: Efficient multilingual models

## Prerequisites

### 1. IBM Cloud Account

You need an IBM Cloud account with access to WatsonX.ai:
1. Sign up at [IBM Cloud](https://cloud.ibm.com/)
2. Create a WatsonX.ai instance
3. Create a project in WatsonX.ai

### 2. API Credentials

Obtain your credentials:
- **API Key**: From IBM Cloud IAM (Identity and Access Management)
- **Project ID**: From your WatsonX.ai project settings
- **URL**: Your WatsonX.ai endpoint (optional, defaults to US South region)

### 3. Installation

Install docling-graph with WatsonX support:

```bash
# Install WatsonX support only
uv sync --extra watsonx

# Or install all remote providers
uv sync --extra remote

# Or install everything
uv sync --extra all
```

## Configuration

### Environment Variables

Set your WatsonX credentials as environment variables:

```bash
# Required
export WATSONX_API_KEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"

# Optional (defaults to US South region)
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

**Regional Endpoints:**
- US South: `https://us-south.ml.cloud.ibm.com`
- EU Germany: `https://eu-de.ml.cloud.ibm.com`
- Japan Tokyo: `https://jp-tok.ml.cloud.ibm.com`

### .env File

Alternatively, create a `.env` file in your project root:

```env
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

## Usage

### Python API

```python
from docling_graph import run_pipeline, PipelineConfig
from examples.templates.rheology_research import Research

# Configure pipeline with WatsonX
config = PipelineConfig(
    source="path/to/document.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="ibm-granite/granite-4.0-h-small",
    output_dir="outputs/watsonx_extraction"
)

# Run extraction
run_pipeline(config)
```

### CLI

```bash
uv run docling-graph convert "path/to/document.pdf" \
    --template "examples.templates.battery_research.Research" \
    --backend llm \
    --inference remote \
    --provider watsonx \
    --model "ibm-granite/granite-4.0-h-small" \
    --output-dir "outputs/watsonx_extraction"
```

## Available Models

### IBM Granite Models

**Granite 4.0 H Small** (Recommended)
- Model ID: `ibm-granite/granite-4.0-h-small`
- Context: 8K tokens
- Best for: General-purpose extraction, fast inference

**Granite 3.0 8B Instruct**
- Model ID: `ibm-granite/granite-3.0-8b-instruct`
- Context: 8K tokens
- Best for: Instruction following, structured output

**Granite 3.0 2B Instruct**
- Model ID: `ibm-granite/granite-3.0-2b-instruct`
- Context: 8K tokens
- Best for: Fast inference, cost-effective

### Meta Llama Models

**Llama 3 70B Instruct**
- Model ID: `meta-llama/llama-3-70b-instruct`
- Context: 8K tokens
- Best for: Complex reasoning, high accuracy

**Llama 3 8B Instruct**
- Model ID: `meta-llama/llama-3-8b-instruct`
- Context: 8K tokens
- Best for: Balanced performance and speed

### Mistral Models

**Mixtral 8x7B Instruct**
- Model ID: `mistralai/mixtral-8x7b-instruct-v01`
- Context: 32K tokens
- Best for: Long documents, multilingual content

## Examples

### Example 1: Scientific Paper Extraction

```python
from docling_graph import run_pipeline, PipelineConfig
from examples.templates.rheology_research import Research

config = PipelineConfig(
    source="docs/examples/data/research_paper/rheology.pdf",
    template=Research,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="ibm-granite/granite-4.0-h-small",
    output_dir="outputs/battery_research"
)

run_pipeline(config)
```

### Example 2: Invoice Processing

```python
from docling_graph import run_pipeline, PipelineConfig
from examples.templates.invoice import Invoice

config = PipelineConfig(
    source="examples/data/invoice/sample_invoice.jpg",
    template=Invoice,
    backend="llm",
    inference="remote",
    processing_mode="one-to-one",
    provider_override="watsonx",
    model_override="ibm-granite/granite-3.0-8b-instruct",
    output_dir="outputs/invoice"
)

run_pipeline(config)
```

### Example 3: Multi-page Document

```python
from docling_graph import run_pipeline, PipelineConfig
from examples.templates.insurance import InsurancePolicy

config = PipelineConfig(
    source="examples/data/insurance/insurance_terms.pdf",
    template=InsurancePolicy,
    backend="llm",
    inference="remote",
    processing_mode="many-to-one",
    provider_override="watsonx",
    model_override="meta-llama/llama-3-70b-instruct",
    output_dir="outputs/insurance"
)

run_pipeline(config)
```

## Best Practices

### Model Selection

1. **Start with Granite 4.0 H Small**: Good balance of speed and accuracy
2. **Use Llama 3 70B for complex documents**: Better reasoning for technical content
3. **Use Mixtral for long documents**: 32K context window handles large documents

### Performance Optimization

1. **Batch Processing**: Process multiple documents in sequence
2. **Processing Mode**: Use `many-to-one` for multi-page documents
3. **Temperature**: Keep at 0.1 for consistent extraction (default)

### Error Handling

```python
from docling_graph import run_pipeline, PipelineConfig

config = PipelineConfig(
    source="document.pdf",
    template=MyTemplate,
    backend="llm",
    inference="remote",
    provider_override="watsonx",
    model_override="ibm-granite/granite-4.0-h-small",
    output_dir="outputs"
)

try:
    run_pipeline(config)
    print("Extraction successful!")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Extraction failed: {e}")
```

## Troubleshooting

### Authentication Errors

**Error**: `WATSONX_API_KEY not set`
- **Solution**: Set the environment variable or add to `.env` file

**Error**: `WATSONX_PROJECT_ID not set`
- **Solution**: Get your project ID from WatsonX.ai project settings

### API Errors

**Error**: `401 Unauthorized`
- **Solution**: Verify your API key is valid and has WatsonX.ai access

**Error**: `403 Forbidden`
- **Solution**: Check your project ID and ensure you have access to the model

**Error**: `Model not found`
- **Solution**: Verify the model ID is correct and available in your region

### Connection Errors

**Error**: `Connection timeout`
- **Solution**: Check your network connection and WATSONX_URL setting

**Error**: `Invalid endpoint`
- **Solution**: Verify WATSONX_URL matches your region

## Cost Considerations

WatsonX.ai pricing is based on:
- **Token usage**: Input and output tokens
- **Model size**: Larger models cost more per token
- **Region**: Prices may vary by region

**Tips to reduce costs:**
1. Use smaller models (Granite 2B/3B) for simple extraction
2. Optimize prompts to reduce token usage
3. Use `many-to-one` mode to process documents in a single call
4. Monitor usage in IBM Cloud dashboard

## Additional Resources

- [IBM WatsonX.ai Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)
- [WatsonX Python SDK](https://ibm.github.io/watsonx-ai-python-sdk/)
- [IBM Cloud API Keys](https://cloud.ibm.com/docs/account?topic=account-userapikey)

## Support

For issues specific to:
- **WatsonX.ai**: Contact IBM Cloud Support
- **docling-graph**: Open an issue on [GitHub](https://github.com/ayoub-ibm/docling-graph/issues)