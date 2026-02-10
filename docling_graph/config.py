"""
Pipeline configuration class for type-safe config creation.

This module provides a PipelineConfig class that makes it easy to create
configurations for the docling-graph pipeline programmatically.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .llm_clients.config import LlmRuntimeOverrides


class BackendConfig(BaseModel):
    """Configuration for an extraction backend."""

    provider: str = Field(..., description="Backend provider (e.g., 'ollama', 'mistral', 'vlm')")
    model: str = Field(..., description="Model name or path")
    api_key: str | None = Field(None, description="API key, if required")
    base_url: str | None = Field(None, description="Base URL for API, if required")


class ExtractorConfig(BaseModel):
    """Configuration for the extraction strategy."""

    strategy: Literal["many-to-one", "one-to-one"] = Field(default="many-to-one")
    extraction_contract: Literal["direct", "staged"] = Field(default="direct")
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")
    use_chunking: bool = Field(default=True)
    llm_consolidation: bool = Field(default=False)
    chunker_config: Dict[str, Any] | None = Field(default=None)


class ModelConfig(BaseModel):
    """Model selection for a backend."""

    model: str = Field(..., description="The model name/path to use")
    provider: str = Field(..., description="The provider for this model")


class LLMConfig(BaseModel):
    """LLM model configurations for local and remote inference."""

    local: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="ibm-granite/granite-4.0-1b",
            provider="vllm",
        )
    )
    remote: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="mistral-small-latest",
            provider="mistral",
        )
    )


class VLMConfig(BaseModel):
    """VLM model configuration."""

    local: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="numind/NuExtract-2.0-8B",
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Optional fields (empty by default, filled in at runtime)
    source: Union[str, Path] = Field(default="", description="Path to the source document")
    template: Union[str, type[BaseModel]] = Field(
        default="", description="Pydantic template class or dotted path string"
    )

    # Core processing settings (with defaults)
    backend: Literal["llm", "vlm"] = Field(default="llm")
    inference: Literal["local", "remote"] = Field(default="local")
    processing_mode: Literal["one-to-one", "many-to-one"] = Field(default="many-to-one")
    extraction_contract: Literal["direct", "staged"] = Field(default="direct")

    # Docling settings (with defaults)
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")

    # Model overrides
    model_override: str | None = None
    provider_override: str | None = None

    # Optional custom LLM client (implements LLMClientProtocol)
    llm_client: Any | None = Field(
        default=None,
        description="Custom LLM client instance to use for LLM backend.",
        exclude=True,
    )

    # Models configuration (flat only, with defaults)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    llm_overrides: LlmRuntimeOverrides = Field(
        default_factory=LlmRuntimeOverrides, description="Runtime overrides for LLM settings."
    )

    # Extract settings (with defaults)
    use_chunking: bool = Field(default=True, description="Enable chunking for document processing")
    chunk_max_tokens: int | None = Field(
        default=None,
        description="Max tokens per chunk when chunking is used (default: 512).",
    )
    debug: bool = Field(
        default=False, description="Enable debug artifacts (controlled by --debug flag)"
    )
    max_batch_size: int = 1
    staged_max_fields_per_group: int = Field(
        default=6, description="Max number of scalar fields per staged extraction group."
    )
    staged_max_skeleton_fields: int = Field(
        default=10, description="Max number of root fields in staged skeleton pass."
    )
    staged_max_repair_rounds: int = Field(
        default=2, description="Maximum targeted repair rounds for staged extraction."
    )
    staged_max_pass_retries: int = Field(
        default=1, description="Retries per staged pass before giving up."
    )
    staged_quality_depth: int = Field(
        default=3, description="Recursive depth for staged quality analysis."
    )
    staged_include_prior_context: bool = Field(
        default=True, description="Include prior pass output in staged prompts."
    )
    llm_consolidation: bool = Field(
        default=False,
        description="Use LLM for conflict resolution when heuristic staged reconciliation is insufficient.",
    )
    staged_merge_similarity_fallback: bool = Field(
        default=True,
        description="When True (default), merge entity list items by similarity (e.g. overlapping children) when ID/identity match fails; logs a warning when used.",
    )

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

    # File export control (with auto-detection)
    dump_to_disk: bool | None = Field(
        default=None,
        description=(
            "Control file exports to disk. "
            "None (default) = auto-detect: CLI mode exports, API mode doesn't. "
            "True = force exports. False = disable exports."
        ),
    )

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
            "extraction_contract": self.extraction_contract,
            "docling_config": self.docling_config,
            "use_chunking": self.use_chunking,
            "chunk_max_tokens": self.chunk_max_tokens,
            "debug": self.debug,
            "model_override": self.model_override,
            "provider_override": self.provider_override,
            "staged_max_fields_per_group": self.staged_max_fields_per_group,
            "staged_max_skeleton_fields": self.staged_max_skeleton_fields,
            "staged_max_repair_rounds": self.staged_max_repair_rounds,
            "staged_max_pass_retries": self.staged_max_pass_retries,
            "staged_quality_depth": self.staged_quality_depth,
            "staged_include_prior_context": self.staged_include_prior_context,
            "llm_consolidation": self.llm_consolidation,
            "staged_merge_similarity_fallback": self.staged_merge_similarity_fallback,
            "export_format": self.export_format,
            "export_docling": self.export_docling,
            "export_docling_json": self.export_docling_json,
            "export_markdown": self.export_markdown,
            "export_per_page_markdown": self.export_per_page_markdown,
            "reverse_edges": self.reverse_edges,
            "output_dir": self.output_dir,
            "dump_to_disk": self.dump_to_disk,
            "models": self.models.model_dump(),
            "llm_overrides": self.llm_overrides.model_dump(),
            "llm_client": self.llm_client,
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
                "extraction_contract": default_config.extraction_contract,
                "export_format": default_config.export_format,
                "chunk_max_tokens": default_config.chunk_max_tokens,
                "staged_max_fields_per_group": default_config.staged_max_fields_per_group,
                "staged_max_skeleton_fields": default_config.staged_max_skeleton_fields,
                "staged_max_repair_rounds": default_config.staged_max_repair_rounds,
                "staged_max_pass_retries": default_config.staged_max_pass_retries,
                "staged_quality_depth": default_config.staged_quality_depth,
                "staged_include_prior_context": default_config.staged_include_prior_context,
                "llm_consolidation": default_config.llm_consolidation,
                "staged_merge_similarity_fallback": default_config.staged_merge_similarity_fallback,
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
            "llm_overrides": default_config.llm_overrides.model_dump(),
            "output": {
                "directory": str(default_config.output_dir),
            },
        }
