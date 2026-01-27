# init Command


## Overview

The `init` command creates a `config.yaml` file in your current directory through an **interactive setup process**.

**Purpose:**
- Generate configuration files
- Validate dependencies
- Guide API key setup
- Provide next steps

---

## Basic Usage

```bash
uv run docling-graph init
```

This launches an interactive wizard that guides you through:
1. Backend selection (LLM/VLM)
2. Inference mode (local/remote)
3. Provider selection
4. Model selection
5. Processing mode
6. Export format

---

## Interactive Setup

### Step 1: Backend Selection

```
Choose your backend:
1. LLM (Language Model) - Best for text-heavy documents
2. VLM (Vision-Language Model) - Best for forms and structured layouts

Your choice [1-2]:
```

### Step 2: Inference Mode

```
Choose inference mode:
1. Local - Run models on your machine
2. Remote - Use API providers (requires API key)

Your choice [1-2]:
```

### Step 3: Provider Selection

**For Local LLM:**
```
Choose local LLM provider:
1. vLLM (recommended for GPU)
2. Ollama (recommended for CPU)

Your choice [1-2]:
```

**For Remote LLM:**
```
Choose remote provider:
1. Mistral AI
2. OpenAI
3. Google Gemini
4. IBM watsonx

Your choice [1-4]:
```

### Step 4: Model Selection

```
Available models for Mistral AI:
1. mistral-small-latest (fast, cost-effective)
2. mistral-large-latest (most capable)

Your choice [1-2]:
```

### Step 5: Processing Mode

```
Choose processing mode:
1. many-to-one - Merge all pages into single graph (recommended)
2. one-to-one - Create separate graph per page

Your choice [1-2]:
```

### Step 6: Export Format

```
Choose export format:
1. CSV (for Neo4j import)
2. Cypher (for direct Neo4j execution)

Your choice [1-2]:
```

---

## Generated Configuration

### Example: Remote LLM (Mistral)

```yaml
# config.yaml
defaults:
  processing_mode: many-to-one
  backend: llm
  inference: remote
  export_format: csv

docling:
  pipeline: ocr
  export:
    docling_json: true
    markdown: true
    per_page_markdown: false

models:
  llm:
    local:
      model: ibm-granite/granite-4.0-1b
      provider: vllm
    remote:
      model: mistral-small-latest
      provider: mistral
  vlm:
    local:
      model: numind/NuExtract-2.0-8B
      provider: docling

output:
  directory: outputs
```

### Example: Local LLM (Ollama)

```yaml
defaults:
  processing_mode: many-to-one
  backend: llm
  inference: local
  export_format: csv

models:
  llm:
    local:
      model: llama3:8b
      provider: ollama
    remote:
      model: mistral-small-latest
      provider: mistral
  vlm:
    local:
      model: numind/NuExtract-2.0-8B
      provider: docling

output:
  directory: outputs
```

### Example: VLM (Local)

```yaml
defaults:
  processing_mode: one-to-one
  backend: vlm
  inference: local
  export_format: csv

docling:
  pipeline: vision

models:
  llm:
    local:
      model: ibm-granite/granite-4.0-1b
      provider: vllm
    remote:
      model: mistral-small-latest
      provider: mistral
  vlm:
    local:
      model: numind/NuExtract-2.0-8B
      provider: docling

output:
  directory: outputs
```

---

## Dependency Validation

After configuration, `init` validates required dependencies:

### All Dependencies Installed

```
✅ All required dependencies are installed
```

### Missing Dependencies

```
⚠ Missing dependencies for remote inference
Run: uv sync
```

**Dependencies:**
- `uv sync` installs LiteLLM and all core runtime dependencies

---

## Next Steps Guidance

### Remote Provider Setup

```
Next steps:
1. Install dependencies:
   uv sync

2. Set your API key:
   export MISTRAL_API_KEY="your-api-key-here"

3. Run your first conversion:
   uv run docling-graph convert document.pdf \
       --template "templates.BillingDocument"
```

### Local Provider Setup

```
Next steps:
1. Install dependencies:
   uv sync

2. Start Ollama server:
   ollama serve

3. Pull the model:
   ollama pull llama3:8b

4. Run your first conversion:
   uv run docling-graph convert document.pdf \
       --template "templates.BillingDocument"
```

---

## Overwriting Configuration

If `config.yaml` already exists:

```
A configuration file: 'config.yaml' already exists.
Overwrite it? [y/N]:
```

- **y** - Replace existing configuration
- **N** - Cancel and keep existing file

---

## Non-Interactive Mode

If interactive mode is unavailable (e.g., in CI/CD):

```bash
uv run docling-graph init
# Falls back to default configuration
```

Default configuration uses:
- Backend: `llm`
- Inference: `local`
- Provider: `vllm`
- Processing: `many-to-one`
- Export: `csv`

---

## Complete Examples

### 📍 First-Time Setup

```bash
# Initialize configuration
uv run docling-graph init

# Follow prompts:
# 1. Choose LLM backend
# 2. Choose remote inference
# 3. Choose provider (LiteLLM ID)
# 4. Choose model (LiteLLM identifier)
# 5. Choose many-to-one processing
# 6. Choose CSV export

# Install dependencies
uv sync

# Set API key
export MISTRAL_API_KEY="your-key"

# Test conversion
uv run docling-graph convert test.pdf \
    --template "templates.BillingDocument"
```

### 📍 Local Development Setup

```bash
# Initialize for local development
uv run docling-graph init

# Follow prompts:
# 1. Choose LLM backend
# 2. Choose local inference
# 3. Choose provider (LiteLLM ID)
# 4. Choose model (LiteLLM identifier)
# 5. Choose many-to-one processing
# 6. Choose CSV export

# Install dependencies
uv sync

# Start Ollama
ollama serve

# Pull model
ollama pull llama3:8b

# Test conversion
uv run docling-graph convert test.pdf \
    --template "templates.BillingDocument"
```

### 📍 VLM Setup

```bash
# Initialize for VLM
uv run docling-graph init

# Follow prompts:
# 1. Choose VLM backend
# 2. Choose local inference (only option for VLM)
# 3. Choose one-to-one processing
# 4. Choose CSV export

# Install dependencies
uv sync

# Test conversion
uv run docling-graph convert form.jpg \
    --template "templates.IDCard"
```

---

## Configuration File Location

The `config.yaml` file is created in your **current working directory**:

```bash
# Create config in project root
cd /path/to/project
uv run docling-graph init

# Creates: /path/to/project/config.yaml
```

**Best Practice:** Run `init` from your project root directory.

---

## Manual Configuration

You can also create `config.yaml` manually:

```yaml
# Minimal configuration
defaults:
  backend: llm
  inference: remote

models:
  llm:
    remote:
      model: mistral-small-latest
      provider: mistral
```

Or use the template:

```bash
# Copy template
cp docling_graph/config_template.yaml config.yaml

# Edit as needed
nano config.yaml
```

---

## Troubleshooting

### 🐛 Interactive Mode Not Available

**Error:**
```
Interactive mode not available. Using default configuration.
```

**Solution:**
- Running in non-interactive environment (CI/CD)
- Default configuration will be used
- Manually edit `config.yaml` if needed

### 🐛 Permission Denied

**Error:**
```
Error saving config: Permission denied
```

**Solution:**
```bash
# Check directory permissions
ls -la

# Run from writable directory
cd ~/projects/my-project
uv run docling-graph init
```

### 🐛 Invalid Configuration

**Error:**
```
Error creating config: Invalid backend type
```

**Solution:**
- Restart `init` command
- Choose valid options (llm/vlm)
- Check for typos in manual edits

---

## Best Practices

### 👍 Initialize Per Project

```bash
# ✅ Good - One config per project
cd project1/
uv run docling-graph init

cd project2/
uv run docling-graph init

# ❌ Avoid - Shared config across projects
cd ~/
uv run docling-graph init
```

### 👍 Version Control

```bash
# ✅ Good - Track configuration
git add config.yaml
git commit -m "Add docling-graph configuration"

# Add to .gitignore if it contains secrets
echo "config.yaml" >> .gitignore
```

### 👍 Environment-Specific Configs

```bash
# Development
cp config.yaml config.dev.yaml

# Production
cp config.yaml config.prod.yaml

# Use specific config
cp config.prod.yaml config.yaml
uv run docling-graph convert document.pdf -t "templates.BillingDocument"
```

---

## Next Steps

Now that you have a configuration:

1. **[convert Command →](convert-command.md)** - Convert documents
2. **[CLI Recipes →](cli-recipes.md)** - Common patterns
3. **[Configuration Guide →](../../fundamentals/pipeline-configuration/index.md)** - Advanced config