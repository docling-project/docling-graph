"""
CLI constants for validation enums and provider defaults.

NOTE: Default values are centralized here and in PipelineConfig.
This module contains validation enums and model defaults for the CLI.
"""

from typing import Any, Final

# Configuration file name
CONFIG_FILE_NAME: Final[str] = "config.yaml"

# Processing modes (enum for validation)
PROCESSING_MODES: Final[list[str]] = ["one-to-one", "many-to-one"]

# Extraction contracts (prompt/execution behavior for LLM backend).
# "auto" resolves to direct or dense per document once its size is known.
EXTRACTION_CONTRACTS: Final[list[str]] = ["direct", "dense", "auto"]

# LLM input serializations. "auto" pairs the format to the resolved contract
# (direct -> doclang-geo, dense -> doclang, raw text -> markdown).
LLM_INPUT_FORMATS: Final[list[str]] = ["markdown", "doclang", "doclang-geo", "auto"]

# Backend types (enum for validation)
BACKENDS: Final[list[str]] = ["llm", "vlm"]

# Inference locations (enum for validation)
INFERENCE_LOCATIONS: Final[list[str]] = ["local", "remote"]

# Export formats (enum for validation)
EXPORT_FORMATS: Final[list[str]] = ["csv", "cypher"]

# Merge duplicate-group fold orders (enum for validation)
MERGE_PRECEDENCE: Final[list[str]] = ["input-order", "richest"]

# Merge scalar conflict policies (enum for validation)
MERGE_CONFLICTS: Final[list[str]] = ["keep-first", "keep-all"]

# Docling pipeline configurations (enum for validation)
DOCLING_PIPELINES: Final[list[str]] = ["ocr", "vision"]

# Ontology formats accepted by `template from-ontology` (enum for validation).
# "auto" sniffs the format from the file suffix/content.
TEMPLATE_FORMATS: Final[list[str]] = ["owl", "linkml", "jsonschema", "auto"]

# Docling export formats (enum for validation)
DOCLING_EXPORT_FORMATS: Final[list[str]] = ["markdown", "json", "document"]

# Providers (enum for validation)
LOCAL_PROVIDERS: Final[list[str]] = ["vllm", "ollama", "lmstudio", "custom"]
API_PROVIDERS: Final[list[str]] = ["mistral", "openai", "gemini", "watsonx", "bedrock", "custom"]

# Provider-specific default models (for CLI prompts)
PROVIDER_DEFAULT_MODELS: Final[dict[str, str]] = {
    "mistral": "mistral-small-latest",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "watsonx": "ibm/granite-4-h-small",
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}

# Local provider default models
LOCAL_PROVIDER_DEFAULTS: Final[dict[str, str]] = {
    "vllm": "ibm-granite/granite-4.0-1b",
    "ollama": "llama-3.1-8b",
    "lmstudio": "local-model",
}

# VLM default model
VLM_DEFAULT_MODEL: Final[str] = "numind/NuExtract-2.0-2B"
