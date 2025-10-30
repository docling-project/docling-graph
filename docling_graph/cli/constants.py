"""
CLI constants and configuration values.
"""

from typing import Final


# Configuration
CONFIG_FILE_NAME: Final[str] = "config.yaml"

# Processing modes
PROCESSING_MODES: Final[list[str]] = ["one-to-one", "many-to-one"]

# Backend types
BACKEND_TYPES: Final[list[str]] = ["llm", "vlm"]

# Inference locations
INFERENCE_LOCATIONS: Final[list[str]] = ["local", "remote"]

# Export formats
EXPORT_FORMATS: Final[list[str]] = ["csv", "cypher", "json"]

# Docling pipeline configurations
DOCLING_PIPELINES: Final[list[str]] = ["ocr", "vision"]

# Docling export formats
DOCLING_EXPORT_FORMATS: Final[list[str]] = ["markdown", "json", "document"]

# Default Docling settings
DEFAULT_DOCLING_CONFIG = {
    "pipeline": "ocr",
    "export": {
        "docling_json": True,
        "markdown": True,
        "per_page_markdown": False
    }
}

# Default model configurations
DEFAULT_MODELS = {
    "vlm": "numind/NuExtract-2.0-8B",
    "llm_local": "llama-3.1-8b",
    "llm_remote": {
        "mistral": "mistral-small-latest",
        "openai": "gpt-4-turbo",
        "gemini": "gemini-2.5-flash"
    }
}

# Local Providers
LOCAL_PROVIDERS: Final[list[str]] = ["vllm", "ollama"]

# API Providers
API_PROVIDERS: Final[list[str]] = ["mistral", "openai", "gemini"]
