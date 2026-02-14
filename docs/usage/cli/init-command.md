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

1. **Processing mode** (one-to-one / many-to-one)
2. **Extraction contract** (direct / staged / delta)
3. **Delta Extraction Tuning** (only when delta is selected): resolvers (enable + mode) and optional quality gate tweaks
4. **Backend type** (LLM / VLM)
5. **Inference location** (local / remote; skipped for VLM)
6. **Docling pipeline** and export options
7. **Provider and model** selection (by backend/inference)
8. **Export format**
9. **Output directory**
10. (If remote LLM with custom provider) Use custom endpoint (URL and API key via environment variables)

When custom endpoint is enabled, the wizard expects the fixed env var names `CUSTOM_LLM_BASE_URL` and `CUSTOM_LLM_API_KEY`.

---

## Interactive Setup

### Step 1: Processing Mode

```
1. Processing Mode
 How should documents be processed?
  • one-to-one: Creates a separate Pydantic instance for each page
  • many-to-one: Combines the entire document into a single Pydantic instance
Select processing mode (one-to-one, many-to-one) [many-to-one]:
```

### Step 2: Extraction Contract

```
2. Extraction Contract
 How should LLM extraction prompts/execution be orchestrated?
  • direct: Single-pass best-effort extraction (fastest)
  • staged: Multi-pass staged extraction for improved small-model quality
  • delta: Chunk-based graph extraction for long documents (batches → merge → projection)
Select extraction contract (direct, staged, delta) [direct]:
```

### Step 3: Delta Extraction Tuning (only when delta is selected)

If you selected **delta**, the wizard shows a **Delta Extraction Tuning** section:

- **Delta Resolvers** — Enable post-merge duplicate resolution and choose mode (`off`, `fuzzy`, `semantic`, `chain`). Resolvers merge near-duplicate entities after the graph merge.
- **Delta Quality Gates** — Optionally customize:
- Max parent lookup misses before failing the quality gate
- Adaptive parent lookup tolerance
- Require at least one relationship in the projected graph
- Require structural attachments (avoid root-only outputs)

These values are written into `defaults` in your `config.yaml`. You can override them later in the config file or via the [convert](convert-command.md) command (where applicable). See [Delta Extraction](../../fundamentals/extraction-process/delta-extraction.md) for full options.

### Step 4: Backend Type

```
Backend Type
 Which AI backend should be used?
  • llm: Language Model (text-based)
  • vlm: Vision-Language Model (image-based)
Select backend type [llm]:
```

### Step 5: Inference Location (LLM only)

```
Inference Location
 How should models be executed?
  • local: Run on your machine
  • remote: Use cloud APIs
Select inference location [remote]:
```

(VLM backend skips this step and uses local inference.)

### Step 6: Docling Pipeline and Export Options

You choose the Docling pipeline (ocr / vision) and whether to export Docling JSON, markdown, and per-page markdown.

### Step 7: Provider and Model Selection

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

### Step 8: Model Selection

```
Select model for <provider> [default]:
```

### Step 9: Export Format

```
Export Format
 Output format for results
  • csv: CSV files (nodes.csv, edges.csv)
  • cypher: Cypher script for Neo4j
Select export format [csv]:
```

### Step 10: Output Directory

The wizard then prompts for the output directory (default: `outputs`).

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

### Example: Delta extraction with resolvers

When you select **delta** and enable resolvers (and optionally customize quality), your `defaults` will include delta options:

```yaml
defaults:
  processing_mode: many-to-one
  extraction_contract: delta
  delta_resolvers_enabled: true
  delta_resolvers_mode: semantic
  delta_quality_max_parent_lookup_miss: 4
  delta_quality_adaptive_parent_lookup: true
  delta_quality_require_relationships: false
  delta_quality_require_structural_attachments: false
  backend: llm
  inference: remote
  export_format: csv
# ... docling, models, output
```

You can then run `docling-graph convert ... --extraction-contract delta` and the config will be used. See [Delta Extraction](../../fundamentals/extraction-process/delta-extraction.md) and [convert Command — Delta Extraction Tuning](convert-command.md#delta-extraction-tuning) for details.

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

   (If you chose custom endpoint, set instead:)
   export CUSTOM_LLM_BASE_URL="https://your-llm.example.com/v1"
   export CUSTOM_LLM_API_KEY="your-key"

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
- Processing: `many-to-one`
- Extraction contract: `direct`
- Backend: `llm`
- Inference: `local`
- Provider: `vllm`
- Export: `csv`

---

## Complete Examples

### 📍 First-Time Setup

```bash
# Initialize configuration
uv run docling-graph init

# Follow prompts:
# 1. Processing mode (e.g. many-to-one)
# 2. Extraction contract (direct / staged / delta)
# 3. (If delta) Resolvers and quality tuning
# 4. Backend (e.g. LLM), inference (e.g. remote)
# 5. Docling pipeline and export options
# 6. Provider and model
# 7. Export format, output directory

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
# 1. Processing mode (e.g. many-to-one)
# 2. Extraction contract (e.g. direct)
# 3. Backend (LLM), inference (local)
# 4. Docling pipeline and export options
# 5. Provider and model
# 6. Export format, output directory

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
# 1. Processing mode (e.g. one-to-one)
# 2. Extraction contract (e.g. direct)
# 3. Backend (VLM) — inference is local only
# 4. Docling pipeline and export options
# 5. Model, export format, output directory

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