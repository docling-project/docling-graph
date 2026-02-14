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

# Extraction contracts (prompt/execution behavior for LLM backend)
EXTRACTION_CONTRACTS: Final[list[str]] = ["direct", "staged", "delta"]
DELTA_RESOLVER_MODES: Final[list[str]] = ["off", "fuzzy", "semantic", "chain"]

# Backend types (enum for validation)
BACKENDS: Final[list[str]] = ["llm", "vlm"]

# Inference locations (enum for validation)
INFERENCE_LOCATIONS: Final[list[str]] = ["local", "remote"]

# Export formats (enum for validation)
EXPORT_FORMATS: Final[list[str]] = ["csv", "cypher"]

# Docling pipeline configurations (enum for validation)
DOCLING_PIPELINES: Final[list[str]] = ["ocr", "vision"]

# Docling export formats (enum for validation)
DOCLING_EXPORT_FORMATS: Final[list[str]] = ["markdown", "json", "document"]

# Providers (enum for validation)
LOCAL_PROVIDERS: Final[list[str]] = ["vllm", "ollama", "custom"]
API_PROVIDERS: Final[list[str]] = ["mistral", "openai", "gemini", "watsonx", "custom"]

# Provider-specific default models (for CLI prompts)
PROVIDER_DEFAULT_MODELS: Final[dict[str, str]] = {
    "mistral": "mistral-small-latest",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "watsonx": "ibm/granite-4-h-small",
}

# Local provider default models
LOCAL_PROVIDER_DEFAULTS: Final[dict[str, str]] = {
    "vllm": "ibm-granite/granite-4.0-1b",
    "ollama": "llama-3.1-8b",
}

# VLM default model
VLM_DEFAULT_MODEL: Final[str] = "numind/NuExtract-2.0-2B"
