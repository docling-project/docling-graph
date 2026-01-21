# Configuration API

Configuration classes for the Docling Graph pipeline.

## PipelineConfig

::: docling_graph.config.PipelineConfig
    options:
      show_source: true
      heading_level: 3

## ModelConfig

::: docling_graph.llm_clients.config.ModelConfig
    options:
      show_source: true
      heading_level: 3

## ProviderConfig

::: docling_graph.llm_clients.config.ProviderConfig
    options:
      show_source: true
      heading_level: 3

## Configuration Functions

### get_provider_config

::: docling_graph.llm_clients.config.get_provider_config
    options:
      show_source: true
      heading_level: 4

### get_model_config

::: docling_graph.llm_clients.config.get_model_config
    options:
      show_source: true
      heading_level: 4

### get_context_limit

::: docling_graph.llm_clients.config.get_context_limit
    options:
      show_source: true
      heading_level: 4

### get_recommended_chunk_size

::: docling_graph.llm_clients.config.get_recommended_chunk_size
    options:
      show_source: true
      heading_level: 4

## See Also

- [Pipeline API](pipeline.md) - Main pipeline interface
- [Configuration Guide](../getting-started/configuration.md) - Detailed configuration guide
- [LLM Clients](llm-clients.md) - LLM client implementations