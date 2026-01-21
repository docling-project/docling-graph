# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Installation Options

Choose the installation option that matches your use case:

### Minimal Installation

Includes core VLM features (Docling), **no** LLM inference:

```bash
pip install docling-graph
```

Or with uv:

```bash
uv pip install docling-graph
```

### Full Installation

Includes **all** features, VLM, and all local/remote LLM providers:

```bash
pip install docling-graph[all]
```

### Specific Features

Install only the features you need:

#### Local LLM Support

Adds support for vLLM and Ollama (requires GPU for vLLM):

```bash
pip install docling-graph[local]
```

#### Remote API Support

Adds support for Mistral, OpenAI, Gemini, and IBM WatsonX APIs:

```bash
pip install docling-graph[remote]
```

#### Individual Providers

Install specific LLM providers:

```bash
# OpenAI
pip install docling-graph[openai]

# Mistral
pip install docling-graph[mistral]

# Google Gemini
pip install docling-graph[gemini]

# IBM WatsonX
pip install docling-graph[watsonx]

# Ollama (local)
pip install docling-graph[ollama]

# vLLM (local)
pip install docling-graph[vllm]
```

## GPU Support (Optional)

For local inference with GPU acceleration, follow the [GPU Setup Guide](../guides/setup_with_gpu_support.md).

## API Key Setup

If you're using remote/cloud inference, set your API keys:

### Linux/macOS

```bash
export OPENAI_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
export WATSONX_API_KEY="your-key-here"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional
```

### Windows (Command Prompt)

```cmd
set OPENAI_API_KEY=your-key-here
set MISTRAL_API_KEY=your-key-here
set GEMINI_API_KEY=your-key-here
set WATSONX_API_KEY=your-key-here
set WATSONX_PROJECT_ID=your-project-id
```

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY="your-key-here"
$env:MISTRAL_API_KEY="your-key-here"
$env:GEMINI_API_KEY="your-key-here"
$env:WATSONX_API_KEY="your-key-here"
$env:WATSONX_PROJECT_ID="your-project-id"
```

### Using .env File

Alternatively, create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-key-here
MISTRAL_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
WATSONX_API_KEY=your-key-here
WATSONX_PROJECT_ID=your-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

## Verify Installation

```bash
# Check version
python -c "import docling_graph; print(docling_graph.__version__)"

# Test CLI
docling-graph --help
```

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph.git
cd docling-graph

# Install with development dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you've installed the required extras:

```bash
pip install docling-graph[all]
```

### GPU Issues

For GPU-related issues, see the [GPU Setup Guide](../guides/setup_with_gpu_support.md).

### API Connection Issues

Verify your API keys are set correctly:

```python
import os
print(os.getenv('OPENAI_API_KEY'))  # Should print your key
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first conversion
- [Configuration](configuration.md) - Learn about configuration options
- [Examples](../examples/README.md) - Explore example use cases