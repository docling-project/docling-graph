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
    extraction_contract: Literal["direct", "dense"] = Field(default="direct")
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")
    use_chunking: bool = Field(default=True)
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
    extraction_contract: Literal["direct", "dense"] = Field(default="direct")

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

    # LLM output mode (default ON): API schema-enforced output via response_format json_schema.
    # Set False to use legacy prompt-embedded schema mode.
    structured_output: bool = Field(
        default=True,
        description="Enable schema-enforced structured output for LLM extraction.",
    )
    structured_sparse_check: bool = Field(
        default=True,
        description="Enable sparse structured-output quality check with automatic legacy fallback.",
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

    # Parallel workers for extraction
    parallel_workers: int | None = Field(
        default=None,
        description="Parallel workers for extraction.",
    )

    # Gleaning settings (for direct contract)
    gleaning_enabled: bool = Field(
        default=True,
        description="Run optional second-pass extraction (what did you miss?) for direct contract.",
    )
    gleaning_max_passes: int = Field(
        default=1,
        description="Max gleaning passes (1 = one extra pass). Used when gleaning_enabled is True.",
    )
    # Dense contract: optional post-merge fuzzy/semantic resolvers
    dense_resolvers_enabled: bool = Field(
        default=False,
        description="Enable optional post-merge dense duplicate resolvers (fuzzy/semantic merge).",
    )
    dense_resolvers_mode: Literal["off", "fuzzy", "semantic", "chain"] = Field(
        default="off",
        description="Dense resolver mode: off | fuzzy | semantic | chain.",
    )
    dense_resolvers_fuzzy_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for dense fuzzy post-merge dedup.",
    )
    dense_resolvers_semantic_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for dense semantic post-merge dedup.",
    )
    dense_resolvers_allow_merge_different_ids: bool = Field(
        default=False,
        description="If True, allow dense resolver to merge nodes with different non-empty ids.",
    )
    dense_prune_barren_branches: bool = Field(
        default=False,
        description="If True, remove dense skeleton nodes that have no filled children and no scalar data (barren branches).",
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

    def to_metadata_config_dict(
        self,
        *,
        resolved_model: str | None = None,
        resolved_provider: str | None = None,
    ) -> Dict[str, Any]:
        """
        Return full effective config as a JSON-serializable dict for metadata.json.
        Includes all options with their effective values (including defaults not overridden).
        """
        # Full dump, JSON-serializable (Path -> str, etc.), exclude non-serializable
        data = self.model_dump(mode="json", exclude={"llm_client", "template"})
        # template can be a Pydantic model class; serialize as dotted path string
        if isinstance(self.template, str):
            data["template"] = self.template
        else:
            data["template"] = f"{self.template.__module__}.{self.template.__qualname__}"
        if resolved_model is not None:
            data["resolved_model"] = resolved_model
        if resolved_provider is not None:
            data["resolved_provider"] = resolved_provider
        return data

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
            "structured_output": self.structured_output,
            "structured_sparse_check": self.structured_sparse_check,
            "use_chunking": self.use_chunking,
            "chunk_max_tokens": self.chunk_max_tokens,
            "debug": self.debug,
            "model_override": self.model_override,
            "provider_override": self.provider_override,
            "parallel_workers": self.parallel_workers,
            "gleaning_enabled": self.gleaning_enabled,
            "gleaning_max_passes": self.gleaning_max_passes,
            "dense_resolvers_enabled": self.dense_resolvers_enabled,
            "dense_resolvers_mode": self.dense_resolvers_mode,
            "dense_resolvers_fuzzy_threshold": self.dense_resolvers_fuzzy_threshold,
            "dense_resolvers_semantic_threshold": self.dense_resolvers_semantic_threshold,
            "dense_resolvers_allow_merge_different_ids": self.dense_resolvers_allow_merge_different_ids,
            "dense_prune_barren_branches": self.dense_prune_barren_branches,
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
                "structured_output": default_config.structured_output,
                "structured_sparse_check": default_config.structured_sparse_check,
                "parallel_workers": default_config.parallel_workers,
                "dense_resolvers_enabled": default_config.dense_resolvers_enabled,
                "dense_resolvers_mode": default_config.dense_resolvers_mode,
                "dense_resolvers_fuzzy_threshold": default_config.dense_resolvers_fuzzy_threshold,
                "dense_resolvers_semantic_threshold": default_config.dense_resolvers_semantic_threshold,
                "dense_resolvers_allow_merge_different_ids": default_config.dense_resolvers_allow_merge_different_ids,
                "dense_prune_barren_branches": default_config.dense_prune_barren_branches,
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
