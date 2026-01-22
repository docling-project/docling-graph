# Basic Setup

**Pipeline Stage**: 2 - Installation

**Prerequisites**: 
- [Installation Overview](index.md)
- [System Requirements](requirements.md)

This page provides step-by-step instructions for installing Docling Graph.

## Installation Methods

### Method 1: From Source (Recommended)

This is the recommended method for most users.

#### Step 1: Install uv

First, install the `uv` package manager:

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip)**:
```bash
pip install uv
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/IBM/docling-graph
cd docling-graph
```

#### Step 3: Choose Installation Type

**Minimal (VLM only)**:
```bash
uv sync
```

**Full (all features)**:
```bash
uv sync --extra all
```

**Local LLM only**:
```bash
uv sync --extra local
```

**Remote API only**:
```bash
uv sync --extra remote
```

**Specific providers**:
```bash
# OpenAI
uv sync --extra openai

# Mistral
uv sync --extra mistral

# Gemini
uv sync --extra gemini

# WatsonX
uv sync --extra watsonx

# Ollama (local)
uv sync --extra ollama

# vLLM (local, requires GPU)
uv sync --extra vllm
```

#### Step 4: Verify Installation

```bash
# Check version
uv run docling-graph --version

# Test CLI
uv run docling-graph --help

# Test Python import
uv run python -c "import docling_graph; print(docling_graph.__version__)"
```

Expected output:
```
Docling Graph v0.3.0
Usage: docling-graph [OPTIONS] COMMAND [ARGS]...
0.3.0
```

### Method 2: From PyPI (Coming Soon)

**Note**: PyPI installation will be available in a future release.

```bash
# Future release
pip install docling-graph[all]
```

## Installation Scenarios

### Scenario 1: Quick Start (Remote LLM)

For users who want to get started quickly without GPU:

```bash
# Install
git clone https://github.com/IBM/docling-graph
cd docling-graph
uv sync --extra remote

# Set API key
export OPENAI_API_KEY="your-key-here"

# Test
uv run docling-graph --version
```

**Time**: ~2-3 minutes  
**Requirements**: Internet connection, API key  
**GPU**: Not required

### Scenario 2: Local VLM (GPU Required)

For users with GPU who want local inference:

```bash
# Install
git clone https://github.com/IBM/docling-graph
cd docling-graph
uv sync

# Verify GPU
nvidia-smi

# Test
uv run docling-graph --version
```

**Time**: ~5-10 minutes  
**Requirements**: NVIDIA GPU with 4+ GB VRAM  
**GPU**: Required

### Scenario 3: Full Local Setup (GPU Required)

For users who want all local capabilities:

```bash
# Install
git clone https://github.com/IBM/docling-graph
cd docling-graph
uv sync --extra local

# Verify GPU
nvidia-smi

# Test
uv run docling-graph --version
```

**Time**: ~10-15 minutes  
**Requirements**: NVIDIA GPU with 8+ GB VRAM  
**GPU**: Required

### Scenario 4: Hybrid (Local + Remote)

For maximum flexibility:

```bash
# Install
git clone https://github.com/IBM/docling-graph
cd docling-graph
uv sync --extra all

# Set API keys (optional)
export OPENAI_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"

# Test
uv run docling-graph --version
```

**Time**: ~10-15 minutes  
**Requirements**: GPU recommended, API keys optional  
**GPU**: Optional

## Post-Installation Configuration

### Initialize Configuration

Run the interactive configuration wizard:

```bash
uv run docling-graph init
```

This creates a `config.yaml` file with your preferences.

**New in v0.3.0**: Init command is 75-85% faster with intelligent caching!

### Verify Installation

Run a simple test:

```bash
# Check all commands work
uv run docling-graph --help
uv run docling-graph init --help
uv run docling-graph convert --help
uv run docling-graph inspect --help
```

### Test with Example

```bash
# Run a simple example (requires API key or GPU)
uv run python docs/examples/scripts/03_llm_remote_api.py
```

## Directory Structure

After installation, your directory should look like:

```
docling-graph/
├── .venv/                  # Virtual environment (created by uv)
├── docs/                   # Documentation
├── docling_graph/          # Source code
├── examples/               # Example scripts and templates
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
└── README.md               # Project readme
```

## Environment Variables

### Optional Configuration

Set these environment variables for customization:

```bash
# Logging level
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Temporary directory
export TEMP_DIR="/tmp/docling"
```

### API Keys (if using remote providers)

See [API Keys Setup](api-keys.md) for detailed instructions.

## Updating

### Update to Latest Version

```bash
# Navigate to repository
cd docling-graph

# Pull latest changes
git pull origin main

# Update dependencies
uv sync --extra all
```

### Update Specific Components

```bash
# Update only remote providers
uv sync --extra remote

# Update only local providers
uv sync --extra local
```

## Troubleshooting

### Issue: `uv` command not found

**Cause**: uv not in PATH

**Solution**:
```bash
# Add to PATH (Linux/macOS)
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: Permission denied

**Cause**: Insufficient permissions

**Solution**:
```bash
# Don't use sudo with uv
# If you used sudo, remove and reinstall:
rm -rf .venv
uv sync --extra all
```

### Issue: Import errors

**Cause**: Not using `uv run`

**Solution**:
```bash
# Wrong
python script.py

# Correct
uv run python script.py
```

### Issue: Slow installation

**Cause**: Network or disk speed

**Solution**:
```bash
# Use verbose mode to see progress
uv sync --extra all --verbose

# Or install in stages
uv sync                    # Core first
uv sync --extra remote     # Then remote
uv sync --extra local      # Then local
```

### Issue: CUDA not found (for GPU users)

**Cause**: CUDA not installed or not in PATH

**Solution**: See [GPU Setup Guide](gpu-setup.md)

### Issue: Out of disk space

**Cause**: Insufficient disk space

**Solution**:
```bash
# Check disk space
df -h

# Clean up if needed
uv cache clean

# Or install minimal version
uv sync  # No extras
```

## Verification Checklist

After installation, verify:

- [ ] `uv run docling-graph --version` works
- [ ] `uv run docling-graph --help` shows commands
- [ ] `uv run python -c "import docling_graph"` succeeds
- [ ] GPU detected (if using local inference): `nvidia-smi`
- [ ] API keys set (if using remote): `echo $OPENAI_API_KEY`
- [ ] Config initialized: `uv run docling-graph init`

## Performance Notes

### Installation Speed

**New in v0.3.0**:
- First install: ~2-5 minutes (depending on extras)
- Subsequent updates: ~30-60 seconds
- Dependency caching: 90-95% faster validation

### Disk Usage

```
Minimal install:     ~2.5 GB
Full install:        ~5 GB
With models:         ~20 GB (varies by model)
```

### Memory Usage

```
Installation:        ~1 GB RAM
Runtime (minimal):   ~2 GB RAM
Runtime (with GPU):  ~8-16 GB RAM
```

## Development Setup

For contributors:

```bash
# Clone repository
git clone https://github.com/IBM/docling-graph
cd docling-graph

# Install with dev dependencies
uv sync --all-extras --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy docling_graph
```

## Uninstalling

To completely remove Docling Graph:

```bash
# Remove virtual environment
cd docling-graph
rm -rf .venv

# Remove repository (optional)
cd ..
rm -rf docling-graph

# Remove cache (optional)
rm -rf ~/.cache/docling-graph
```

## Next Steps

Installation complete! Now:

1. **[GPU Setup](gpu-setup.md)** (if using local inference) - Configure CUDA
2. **[API Keys](api-keys.md)** (if using remote) - Set up API keys
3. **[Schema Definition](../03-schema-definition/index.md)** - Create your first template
4. **[Quick Start](../09-examples/quickstart.md)** - Run your first extraction

## Related Documentation

- **[Requirements](requirements.md)**: System requirements
- **[GPU Setup](gpu-setup.md)**: CUDA configuration
- **[API Keys](api-keys.md)**: Remote provider setup
- **[Installation Overview](index.md)**: Installation options

---

**Installation successful?** Continue to [GPU setup](gpu-setup.md) or [API keys](api-keys.md) depending on your use case!