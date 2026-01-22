# Installation

**Pipeline Stage**: 2 - Installation

**Prerequisites**: 
- [Introduction](../01-introduction/index.md)

This section guides you through setting up Docling Graph on your system.

## Overview

Docling Graph uses **uv** as the package manager for fast, reliable dependency management. All installation and execution commands use `uv` exclusively.

### What You'll Install

1. **Core Package**: Docling Graph with VLM support
2. **Optional Features**: LLM providers (local and/or remote)
3. **GPU Support** (optional): PyTorch with CUDA for local inference
4. **API Keys** (optional): For remote LLM providers

## Quick Start

### Minimal Installation

For basic VLM functionality:

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install core dependencies
uv sync
```

This installs:
- ✅ Docling (document conversion)
- ✅ VLM backend (NuExtract models)
- ✅ Core graph functionality
- ❌ LLM providers (not included)

### Full Installation

For all features (VLM + all LLM providers):

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install all dependencies
uv sync --extra all
```

This installs:
- ✅ Everything from minimal
- ✅ vLLM (local LLM inference)
- ✅ Ollama client
- ✅ Mistral AI client
- ✅ OpenAI client
- ✅ Google Gemini client
- ✅ IBM WatsonX client

## Installation Options

### By Feature Set

Choose the installation that matches your needs:

#### Local LLM Support

For local inference with vLLM and Ollama:

```bash
uv sync --extra local
```

**Includes**:
- vLLM (requires GPU)
- Ollama client

**Use when**:
- You have GPU available
- Privacy is critical
- No API costs desired

#### Remote API Support

For cloud-based LLM providers:

```bash
uv sync --extra remote
```

**Includes**:
- Mistral AI client
- OpenAI client
- Google Gemini client
- IBM WatsonX client

**Use when**:
- No GPU available
- Need high-quality models
- Willing to pay API costs

#### Individual Providers

Install specific providers only:

```bash
# OpenAI only
uv sync --extra openai

# Mistral only
uv sync --extra mistral

# Google Gemini only
uv sync --extra gemini

# IBM WatsonX only
uv sync --extra watsonx

# Ollama only (local)
uv sync --extra ollama

# vLLM only (local, requires GPU)
uv sync --extra vllm
```

### Combining Features

You can combine multiple extras:

```bash
# Local + Remote
uv sync --extra local --extra remote

# Specific providers
uv sync --extra ollama --extra openai --extra mistral
```

## System Requirements

### Minimum Requirements

- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 8 GB minimum
- **Disk**: 5 GB free space
- **OS**: Linux, macOS, or Windows (with WSL recommended)

### Recommended for Local Inference

- **GPU**: NVIDIA GPU with 8+ GB VRAM
- **CUDA**: 11.8 or 12.1
- **RAM**: 16 GB or more
- **Disk**: 20 GB free space (for models)

### For VLM Only

- **GPU**: NVIDIA GPU with 4+ GB VRAM (for NuExtract-2B)
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for NuExtract-8B)

### For Remote API Only

- **No GPU required**
- **Internet connection** required
- **API keys** required

## Verification

### Check Installation

```bash
# Check version
uv run docling-graph --version

# Check Python version
uv run python --version

# Test CLI
uv run docling-graph --help
```

Expected output:
```
Docling Graph v0.3.0
Python 3.10+ 
Usage: docling-graph [OPTIONS] COMMAND [ARGS]...
```

### Test Import

```bash
uv run python -c "import docling_graph; print(docling_graph.__version__)"
```

Expected output:
```
0.3.0
```

## Next Steps

After installation, you need to:

1. **[Set Up Requirements](requirements.md)** - Verify system requirements
2. **[Configure GPU](gpu-setup.md)** (optional) - Set up CUDA for local inference
3. **[Set Up API Keys](api-keys.md)** (optional) - Configure remote providers
4. **[Define Schema](../03-schema-definition/index.md)** - Create your first Pydantic template

## Common Issues

### Issue: `uv` not found

**Solution**: Install uv first:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Issue: Python version mismatch

**Solution**: Specify Python version:

```bash
uv python install 3.10
uv sync
```

### Issue: Import errors after installation

**Solution**: Ensure you're using `uv run`:

```bash
# Wrong
python script.py

# Correct
uv run python script.py
```

### Issue: GPU not detected

**Solution**: See [GPU Setup Guide](gpu-setup.md)

## Performance Notes

**New in v0.3.0**: Significant CLI performance improvements:

- **Init command**: 75-85% faster with intelligent dependency caching
  - First run: ~1-1.5s (checks dependencies)
  - Subsequent runs: ~0.5-1s (uses cache)
- **Dependency validation**: 90-95% faster (2-3s → 0.1-0.2s)
- **Lazy loading**: Configuration constants loaded on-demand

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install with development dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```

## Updating

To update to the latest version:

```bash
# Update from git
git pull origin main

# Sync dependencies
uv sync --extra all
```

## Uninstalling

To remove Docling Graph:

```bash
# Remove virtual environment
rm -rf .venv

# Remove cloned repository
cd ..
rm -rf docling-graph
```

## Related Documentation

- **[Requirements](requirements.md)**: Detailed system requirements
- **[GPU Setup](gpu-setup.md)**: Configure CUDA for local inference
- **[API Keys](api-keys.md)**: Set up remote providers
- **[Quick Start](../09-examples/quickstart.md)**: Your first extraction

---

**Installation complete?** Move on to [schema definition](../03-schema-definition/index.md) to create your first template!