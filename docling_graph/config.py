"""
Pipeline configuration class for type-safe config creation.

This module provides a PipelineConfig class that makes it easy to create
configurations for the docling-graph pipeline programmatically.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from typing_extensions import Self  # <-- use typing_extensions for Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    default_model: str = Field(..., description="The model name/path to use")
    provider: str = Field(..., description="The provider for this model")


class LLMConfig(BaseModel):
    """LLM model configurations for local and remote inference."""
    local: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            default_model="ibm-granite/granite-4.0-1b",
            provider="vllm",
        )
    )
    remote: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            default_model="mistral-small-latest",
            provider="mistral",
        )
    )


class VLMConfig(BaseModel):
    """VLM model configuration."""
    local: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            default_model="numind/NuExtract-2.0-8B",
            provider="docling",
        )
    )


class ModelsConfig(BaseModel):
    """Complete models configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)


class PipelineConfig(BaseModel):
    """
    Type-safe configuration for the docling-graph pipeline.
    This is the SINGLE SOURCE OF TRUTH for all defaults.
    All other modules should reference these defaults via PipelineConfig, not duplicate them.
    """

    # Optional fields (empty by default, filled in at runtime)
    source: Union[str, Path] = Field(default="", description="Path to the source document")
    template: Union[str, type[BaseModel]] = Field(
        default="", description="Pydantic template class or dotted path string"
    )

    # Core processing settings (with defaults)
    backend: Literal["llm", "vlm"] = Field(default="llm")
    inference: Literal["local", "remote"] = Field(default="local")
    processing_mode: Literal["one-to-one", "many-to-one"] = Field(default="many-to-one")

    # Docling settings (with defaults)
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")

    # Model overrides
    model_override: Optional[str] = None
    provider_override: Optional[str] = None

    # Models configuration (flat only, with defaults)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # Export settings (with defaults)
    export_format: Literal["csv", "cypher"] = Field(default="csv")
    export_docling: bool = Field(default=True)
    export_docling_json: bool = Field(default=True)
    export_markdown: bool = Field(default=True)
    export_per_page_markdown: bool = Field(default=False)

    # Graph settings (with defaults)
    reverse_edges: bool = Field(default=False)

    # Output settings (with defaults)
    output_dir: Union[str, Path] = Field(default="outputs")

    @field_validator("source", "output_dir")
    @classmethod
    def _path_to_str(cls, v: Union[str, Path]) -> str:
        return str(v)

    @model_validator(mode="after")
    def _validate_vlm_constraints(self) -> Self:
        if self.backend == "vlm" and self.inference == "remote":
            raise ValueError(
                "VLM backend currently only supports local inference. Use inference='local' or backend='llm'."
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format expected by run_pipeline."""
        return {
            "source": self.source,
            "template": self.template,
            "backend": self.backend,
            "inference": self.inference,
            "processing_mode": self.processing_mode,
            "docling_config": self.docling_config,
            "model_override": self.model_override,
            "provider_override": self.provider_override,
            "export_format": self.export_format,
            "export_docling": self.export_docling,
            "export_docling_json": self.export_docling_json,
            "export_markdown": self.export_markdown,
            "export_per_page_markdown": self.export_per_page_markdown,
            "reverse_edges": self.reverse_edges,
            "output_dir": self.output_dir,
            "models": self.models.model_dump(),
        }

    def run(self) -> None:
        """Convenience method to run the pipeline with this configuration."""
        from docling_graph.pipeline import run_pipeline
        run_pipeline(self.to_dict())

    @classmethod
    def generate_yaml_dict(cls) -> Dict[str, Any]:
        """
        Generate a YAML-compatible config dict with all defaults.
        This is used by init.py to create config_template.yaml and config.yaml
        without hardcoding defaults in multiple places.
        """
        default_config = cls()
        return {
            "defaults": {
                "backend": default_config.backend,
                "inference": default_config.inference,
                "processing_mode": default_config.processing_mode,
                "export_format": default_config.export_format,
            },
            "docling": {
                "pipeline": default_config.docling_config,
                "export": {
                    "docling_json": default_config.export_docling_json,
                    "markdown": default_config.export_markdown,
                    "per_page_markdown": default_config.export_per_page_markdown,
                },
            },
            "models": default_config.models.model_dump(),
            "output": {
                "directory": str(default_config.output_dir),
            },
        }
