# CLI Reference


## Overview

The **docling-graph CLI** provides command-line tools for document-to-graph conversion, configuration management, and graph visualization.

**Available Commands:**
- `init` - Create configuration files
- `convert` - Convert documents to graphs
- `inspect` - Visualize graphs in browser

---

## Quick Start

### Installation

```bash
# Install with all features
uv sync

# Verify installation
uv run docling-graph --version
```

### Basic Usage

```bash
# 1. Initialize configuration
uv run docling-graph init

# 2. Convert a document
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument"

# 3. Visualize the graph
uv run docling-graph inspect outputs/
```

---

## Global Options

Available with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Enable detailed logging |
| `--version` | | Show version and exit |
| `--help` | `-h` | Show help message |

### Examples

```bash
# Show version
uv run docling-graph --version

# Enable verbose logging
uv run docling-graph --verbose convert document.pdf -t "templates.BillingDocument"

# Show help
uv run docling-graph --help
uv run docling-graph convert --help
```

---

## Command Overview

### init

Create a configuration file with interactive prompts.

```bash
uv run docling-graph init
```

**Features:**
- Interactive configuration builder
- Dependency validation
- Provider/model identifiers use LiteLLM routing
- API key guidance

**Learn more:** [init Command →](init-command.md)

---

### convert

Convert documents to knowledge graphs.

```bash
uv run docling-graph convert SOURCE --template TEMPLATE [OPTIONS]
```

**Features:**
- Multiple backend support (LLM/VLM)
- Flexible processing modes
- Configurable chunking
- Multiple export formats

**Learn more:** [convert Command →](convert-command.md)

---

### inspect

Visualize graphs in your browser.

```bash
uv run docling-graph inspect PATH [OPTIONS]
```

**Features:**
- Interactive HTML visualization
- CSV and JSON import
- Node/edge exploration
- Self-contained output

**Learn more:** [inspect Command →](inspect-command.md)

---

## Common Workflows

### Workflow 1: First-Time Setup

```bash
# 1. Initialize configuration
uv run docling-graph init

# 2. Install dependencies (if prompted)
uv sync

# 3. Set API key (if using remote)
export MISTRAL_API_KEY="your-key"

# 4. Convert first document
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument"
```

### Workflow 2: Batch Processing

```bash
# Process multiple documents
for pdf in documents/*.pdf; do
    uv run docling-graph convert "$pdf" \
        --template "templates.BillingDocument" \
        --output-dir "outputs/$(basename $pdf .pdf)"
done

# Visualize results
for dir in outputs/*/; do
    uv run docling-graph inspect "$dir" \
        --output "${dir}/visualization.html" \
        --no-open
done
```

### Workflow 3: Development Iteration

```bash
# 1. Convert with verbose logging
uv run docling-graph --verbose convert document.pdf \
    --template "templates.BillingDocument" \
    --output-dir "test_output"

# 2. Inspect results
uv run docling-graph inspect test_output/

# 3. Iterate on template
# Edit templates/billing_document.py

# 4. Re-run conversion
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --output-dir "test_output"
```

---

## Configuration Priority

The CLI uses the following priority order (highest to lowest):

1. **Command-line arguments** (e.g., `--backend llm`)
2. **config.yaml** (created by `init`)
3. **Built-in defaults** (from PipelineConfig)

### Example

```yaml
# config.yaml
defaults:
  backend: llm
  inference: local
```

```bash
# This uses remote inference (CLI overrides config)
uv run docling-graph convert doc.pdf \
    --template "templates.BillingDocument" \
    --inference remote
```

---

## Environment Variables

### API Keys

```bash
# Remote providers
export MISTRAL_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export WATSONX_API_KEY="your-key"
```

### Local Providers

```bash
# vLLM base URL (default: http://localhost:8000/v1)
export VLLM_BASE_URL="http://custom-host:8000/v1"

# Ollama base URL (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://custom-host:11434"
```

---

## Output Structure

Default output directory structure:

```
outputs/
├── metadata.json          # Pipeline metadata
├── docling/               # Docling conversion output
│   ├── document.json      # Docling format
│   └── document.md        # Markdown export
└── docling_graph/         # Graph outputs
    ├── graph.json         # Complete graph
    ├── nodes.csv          # Node data
    ├── edges.csv          # Edge data
    ├── graph.html         # Interactive visualization
    └── report.md          # Summary report
```

---

## Error Handling

### Common Errors

**Configuration Error:**
```bash
[red]Configuration Error:[/red] Invalid backend type: 'invalid'
```
**Solution:** Use `llm` or `vlm`

**Extraction Error:**
```bash
[red]Extraction Error:[/red] Template not found: 'templates.Missing'
```
**Solution:** Check template path and ensure it's importable

**Pipeline Error:**
```bash
[red]Pipeline Error:[/red] API key not found for provider: mistral
```
**Solution:** Set `MISTRAL_API_KEY` environment variable

### Verbose Mode

Enable verbose logging for debugging:

```bash
uv run docling-graph --verbose convert document.pdf \
    --template "templates.BillingDocument"
```

---

## Best Practices

### 👍 Use Configuration Files

```bash
# ✅ Good - Reusable configuration
uv run docling-graph init
uv run docling-graph convert document.pdf -t "templates.BillingDocument"

# ❌ Avoid - Repeating options
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument" \
    --backend llm \
    --inference remote \
    --provider mistral \
    --model mistral-large-latest
```

### 👍 Organize Output

```bash
# ✅ Good - Organized by document
uv run docling-graph convert invoice_001.pdf \
    --template "templates.BillingDocument" \
    --output-dir "outputs/invoice_001"

# ❌ Avoid - Overwriting outputs
uv run docling-graph convert invoice_001.pdf \
    --template "templates.BillingDocument"
```

### 👍 Use Verbose for Development

```bash
# ✅ Good - Debug during development
uv run docling-graph --verbose convert document.pdf \
    --template "templates.BillingDocument"

# ✅ Good - Silent in production
uv run docling-graph convert document.pdf \
    --template "templates.BillingDocument"
```

---

## Next Steps

Explore each command in detail:

1. **[init Command →](init-command.md)** - Configuration setup
2. **[convert Command →](convert-command.md)** - Document conversion
3. **[inspect Command →](inspect-command.md)** - Graph visualization
4. **[CLI Recipes →](cli-recipes.md)** - Common patterns

Or continue to:
- **[Python API →](../api/index.md)** - Programmatic usage
- **[Examples →](../examples/index.md)** - Real-world examples