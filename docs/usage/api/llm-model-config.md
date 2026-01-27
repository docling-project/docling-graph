# LLM Model Configuration

This guide explains how to define models, override settings, and inspect the
resolved (effective) LLM configuration at runtime.

## Select a Model and Provider

Model context windows and output limits are resolved dynamically via LiteLLM.
To use a new model, simply specify the provider and model name in your config
or via CLI overrides.

## Override via Python (API)

You can override generation, reliability, connection settings, and model limits at runtime:

```python
from docling_graph import PipelineConfig

config = PipelineConfig(
    source="doc.pdf",
    template="templates.BillingDocument",
    backend="llm",
    inference="remote",
    model_override="gpt-4o",
    provider_override="openai",
    llm_overrides={
        "generation": {"temperature": 0.2, "max_tokens": 2048},
        "reliability": {"timeout_s": 120, "max_retries": 1},
        "context_limit": 128000,           # Override context window size
        "max_output_tokens": 4096,        # Override max output tokens
    },
)
```

## Override via Config File

In `config.yaml`, use the same `llm_overrides` shape:

```yaml
models:
  llm:
    remote:
      provider: openai
      model: gpt-4o

llm_overrides:
  generation:
    temperature: 0.2
    max_tokens: 2048
  reliability:
    timeout_s: 120
  context_limit: 128000        # Override context window size
  max_output_tokens: 4096      # Override max output tokens
```

## Override via CLI

Common overrides:

```bash
docling-graph convert doc.pdf --template templates.BillingDocument \
  --provider openai --model gpt-4o \
  --llm-temperature 0.2 \
  --llm-max-tokens 2048 \
  --llm-timeout 120 \
  --llm-context-limit 128000 \
  --llm-max-output-tokens 4096
```

### Available CLI Overrides

- `--llm-temperature`: Generation temperature (0.0-2.0)
- `--llm-max-tokens`: Maximum tokens in response
- `--llm-top-p`: Top-p sampling parameter
- `--llm-timeout`: Request timeout in seconds
- `--llm-retries`: Maximum retry attempts
- `--llm-base-url`: Custom API base URL
- `--llm-context-limit`: Total context window size in tokens
- `--llm-max-output-tokens`: Maximum tokens the model can generate

## Model Limits and Defaults

### Context Limit and Max Output Tokens

By default, these values are resolved from LiteLLM metadata. If LiteLLM doesn't have information about your model, the system falls back to defaults:

- **Default context limit**: 8,192 tokens
- **Default max output tokens**: 2,048 tokens

**Important**: If you see warnings about falling back to defaults, provide explicit values via CLI flags or `llm_overrides` to optimize extraction performance.

### Merge Threshold

The `merge_threshold` controls when chunks are merged into batches (default: **95%**). This is provider-specific and can be overridden programmatically:

```python
from docling_graph.core.extractors import ChunkBatcher

batcher = ChunkBatcher(
    context_limit=128000,
    schema_json='{"title": "Schema"}',
    tokenizer=tokenizer,
    merge_threshold=0.90,  # Override default 95%
    provider="openai",
)
```

**Note**: Merge threshold is not currently available as a CLI option. Use the Python API for advanced control.

## View the Resolved Config

CLI:

```bash
docling-graph convert doc.pdf --template templates.BillingDocument --show-llm-config
```

Python:

```python
from docling_graph.llm_clients.config import resolve_effective_model_config

effective = resolve_effective_model_config("openai", "gpt-4o")
print(effective.model_dump())
```
