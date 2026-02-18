# Installation

## Overview

Docling Graph is available on **PyPI**. Install with pip for the recommended experience, or clone the repository and use **uv** for development.

### What You'll Install

1. **Core Package**: Docling Graph with VLM support
2. **Optional Features**: LLM providers (local and/or remote) via LiteLLM (included by default)
3. **GPU Support** (optional): PyTorch with CUDA for local inference
4. **API Keys** (optional): For remote LLM providers

## Quick Start

### Install from PyPI (Recommended)

```bash
pip install docling-graph
```

This installs:

- ✅ Docling (document conversion)
- ✅ VLM backend (NuExtract models)
- ✅ Core graph functionality
- ✅ LiteLLM (for LLM providers; no extra install needed)

Run the CLI with:

```bash
docling-graph --version
docling-graph --help
```

### Install from Source (Development)

To contribute or use the latest development version:

```bash
git clone https://github.com/docling-project/docling-graph
cd docling-graph
uv sync
```

Use `uv run docling-graph` when running the CLI from a source checkout.

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
# Check version (use docling-graph if installed via pip; uv run docling-graph if from source)
docling-graph --version

# Test CLI
docling-graph --help
```

Expected output:
```
Docling Graph v1.2.0
Usage: docling-graph [OPTIONS] COMMAND [ARGS]...
```

### Test Import

```bash
python -c "import docling_graph; print(docling_graph.__version__)"
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

### 🐛 `uv` not found (source install only)

If you install from source, you need [uv](https://docs.astral.sh/uv/). Install it with:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 🐛 Python version mismatch (source install)

When using uv from source, specify Python version if needed:

```bash
uv python install 3.10
uv sync
```

### 🐛 Import errors after installation

**Solution**: If you installed from source with uv, run scripts and the CLI via `uv run`:

```bash
uv run python script.py
uv run docling-graph --help
```

If you installed with pip, use `python` and `docling-graph` directly.

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
git clone https://github.com/docling-project/docling-graph
cd docling-graph

# Install with development dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```

## Updating

**If you installed from PyPI:**

```bash
pip install -U docling-graph
```

**If you installed from source:**

```bash
git pull origin main
uv sync
```

## Uninstalling

**If you installed from PyPI:**

```bash
pip uninstall docling-graph
```

**If you installed from source:**

```bash
rm -rf .venv
cd ..
rm -rf docling-graph
```