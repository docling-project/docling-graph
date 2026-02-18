# Basic Setup

## Installation Methods

### Method 1: From PyPI (Recommended)

Install the latest release from PyPI:

```bash
pip install docling-graph
```

This installs the core package with VLM support and LiteLLM (for LLM providers). No extra steps are required for remote or local LLM backends.

**Verify installation:**

```bash
docling-graph --version
docling-graph --help
python -c "import docling_graph; print(docling_graph.__version__)"
```

Expected output:
```
Docling Graph v1.2.0
Usage: docling-graph [OPTIONS] COMMAND [ARGS]...
v1.2.0
```

### Method 2: From Source (Development)

Use this method to contribute or run the latest development version.

#### Step 1: Install uv

Install the [uv](https://docs.astral.sh/uv/) package manager:

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
git clone https://github.com/docling-project/docling-graph
cd docling-graph
```

#### Step 3: Install Dependencies

```bash
uv sync
```

This installs the same core package as PyPI (VLM + LiteLLM). Use `uv sync --extra dev` for development tools.

#### Step 4: Verify Installation

When installed from source, run the CLI with `uv run`:

```bash
uv run docling-graph --version
uv run docling-graph --help
uv run python -c "import docling_graph; print(docling_graph.__version__)"
```

## Installation Scenarios

### Scenario 1: Quick Start (Remote LLM)

For users who want to get started quickly without GPU:

```bash
# Install from PyPI
pip install docling-graph

# Set API key
export OPENAI_API_KEY="your-key-here"

# Test
docling-graph --version
```

**Time**: ~1-2 minutes  
**Requirements**: Internet connection, API key  
**GPU**: Not required

### Scenario 2: Local VLM (GPU Required)

For users with GPU who want local inference:

```bash
# Install from PyPI
pip install docling-graph

# Verify GPU
nvidia-smi

# Test
docling-graph --version
```

**Time**: ~2-5 minutes  
**Requirements**: NVIDIA GPU with 4+ GB VRAM  
**GPU**: Required

### Scenario 3: Full Local Setup (GPU Required)

For users who want all local capabilities:

```bash
# Install from PyPI
pip install docling-graph

# Verify GPU
nvidia-smi

# Test
docling-graph --version
```

**Time**: ~5-10 minutes  
**Requirements**: NVIDIA GPU with 8+ GB VRAM  
**GPU**: Required

### Scenario 4: Hybrid (Local + Remote)

For maximum flexibility:

```bash
# Install from PyPI
pip install docling-graph

# Set API keys (optional)
export OPENAI_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"

# Test
docling-graph --version
```

**Time**: ~2-5 minutes  
**Requirements**: GPU recommended, API keys optional  
**GPU**: Optional

## Post-Installation Configuration

### Initialize Configuration

Run the interactive configuration wizard:

```bash
docling-graph init
```

(Use `uv run docling-graph init` if you installed from source.)

This creates a `config.yaml` file with your preferences.

**New in v1.2.0**: Init command is 75-85% faster with intelligent caching!

### Verify Installation

Run a simple test:

```bash
# Check all commands work (use uv run ... if from source)
docling-graph --help
docling-graph init --help
docling-graph convert --help
docling-graph inspect --help
```

### Test with Example

```bash
# Run a simple example (requires API key or GPU; from repo only)
python docs/examples/scripts/02_quickstart_llm_pdf.py
# Or from source: uv run python docs/examples/scripts/02_quickstart_llm_pdf.py
```

## Directory Structure (Source Install Only)

When you install from source, your directory should look like:

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
uv sync
```

### Update Specific Components

```bash
# Update only remote providers
uv sync

# Update only local providers
uv sync
```

## Troubleshooting

### 🐛 `uv` command not found

**Cause**: uv not in PATH

**Solution**:
```bash
# Add to PATH (Linux/macOS)
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🐛 Permission denied

**Cause**: Insufficient permissions

**Solution**:
```bash
# Don't use sudo with uv
# If you used sudo, remove and reinstall:
rm -rf .venv
uv sync
```

### 🐛 Import errors (source install)

**Cause**: When installed from source, scripts must be run with `uv run` so they use the project environment.

**Solution**:
```bash
# From source: use uv run
uv run python script.py
uv run docling-graph --help
```
If you installed with pip, use `python` and `docling-graph` directly.

### 🐛 Slow installation

**Cause**: Network or disk speed

**Solution**:
```bash
# Use verbose mode to see progress
uv sync --verbose

# Or install in stages
uv sync                    # Core first
uv sync     # Then remote
uv sync     # Then local
```

### 🐛 CUDA not found (for GPU users)

**Cause**: CUDA not installed or not in PATH

**Solution**: See [GPU Setup Guide](gpu-setup.md)

### 🐛 Out of disk space

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

- [ ] `docling-graph --version` works (or `uv run docling-graph --version` if from source)
- [ ] `docling-graph --help` shows commands
- [ ] `python -c "import docling_graph"` succeeds
- [ ] GPU detected (if using local inference): `nvidia-smi`
- [ ] API keys set (if using remote): `echo $OPENAI_API_KEY`
- [ ] Config initialized: `docling-graph init`

## Performance Notes

### Installation Speed

**New in v1.2.0**:
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
git clone https://github.com/docling-project/docling-graph
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

**If you installed from PyPI:**
```bash
pip uninstall docling-graph
```

**If you installed from source:**
```bash
cd docling-graph
rm -rf .venv
cd ..
rm -rf docling-graph
# Optional: remove cache
rm -rf ~/.cache/docling-graph
```

## Next Steps

Installation complete! Now:

1. **[GPU Setup](gpu-setup.md)** (if using local inference) - Configure CUDA
2. **[API Keys](api-keys.md)** (if using remote) - Set up API keys
3. **[Schema Definition](../schema-definition/index.md)** - Create your first template
4. **[Quick Start](../../introduction/quickstart.md)** - Run your first extraction