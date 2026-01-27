# Installation

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

### Installation

LiteLLM is included by default; no extra installs are required for LLM providers.

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install dependencies
uv sync
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
Docling Graph v1.2.0
Python 3.10+ 
Usage: docling-graph [OPTIONS] COMMAND [ARGS]...
```

### Test Import

```bash
uv run python -c "import docling_graph; print(docling_graph.__version__)"
```

Expected output:
```
v1.2.0
```

## Next Steps

After installation, you need to:

1. **[Set Up Requirements](requirements.md)** - Verify system requirements
2. **[Configure GPU](gpu-setup.md)** (optional) - Set up CUDA for local inference
3. **[Set Up API Keys](api-keys.md)** (optional) - Configure remote providers
4. **[Define Schema](../schema-definition/index.md)** - Create your first Pydantic template

## Common Issues

### 🐛 `uv` not found

**Solution**: Install uv first:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 🐛 Python version mismatch

**Solution**: Specify Python version:

```bash
uv python install 3.10
uv sync
```

### 🐛 Import errors after installation

**Solution**: Ensure you're using `uv run`:

```bash
# Wrong
python script.py

# Correct
uv run python script.py
```

### 🐛 GPU not detected

**Solution**: See [GPU Setup Guide](gpu-setup.md)

## Performance Notes

**New in v1.2.0**: Significant CLI performance improvements:

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
uv sync
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