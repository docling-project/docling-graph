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
    extraction_contract: Literal["direct", "staged", "delta"] = Field(default="direct")
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


# Staged (3-pass) preset defaults: (pass_retries, workers, nodes_fill_cap, id_shard_size).
_STAGED_PRESETS: Dict[str, tuple[int, int, int, int]] = {
    "standard": (2, 1, 5, 0),
    "advanced": (1, 1, 15, 0),
}


def get_effective_staged_tuning(
    preset: str,
    pass_retries: int | None,
    workers: int | None,
    nodes_fill_cap: int | None,
    id_shard_size: int | None = None,
) -> tuple[int, int, int, int]:
    """Return (pass_retries, workers, nodes_fill_cap, id_shard_size) with preset defaults applied."""
    r, w, n, s = _STAGED_PRESETS.get(preset, _STAGED_PRESETS["standard"])
    return (
        pass_retries if pass_retries is not None else r,
        workers if workers is not None else w,
        nodes_fill_cap if nodes_fill_cap is not None else n,
        id_shard_size if id_shard_size is not None else s,
    )


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
    extraction_contract: Literal["direct", "staged", "delta"] = Field(default="direct")

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
    llm_batch_token_size: int = Field(
        default=1024,
        description="Max total input tokens per LLM batch in delta extraction.",
    )
    debug: bool = Field(
        default=False, description="Enable debug artifacts (controlled by --debug flag)"
    )
    max_batch_size: int = 1

    # Staged (3-pass) tuning: ID pass → fill pass (bottom-up) → merge. Preset sets defaults; overrides apply when not None.
    staged_tuning_preset: Literal["standard", "advanced"] = Field(
        default="standard",
        description="Preset: 'standard' for typical LLMs; 'advanced' for larger context (more paths per ID-pass call, larger fill batches).",
    )
    staged_pass_retries: int | None = Field(
        default=None,
        description="Retries per staged pass when LLM returns invalid JSON (None = use preset).",
    )
    parallel_workers: int | None = Field(
        default=None,
        description="Parallel workers for extraction (staged fill pass and delta batch calls). None = use preset for staged.",
    )
    delta_normalizer_validate_paths: bool = Field(
        default=True, description="Drop/repair delta IR nodes with unknown catalog paths."
    )
    delta_normalizer_canonicalize_ids: bool = Field(
        default=True, description="Canonicalize delta IR identifier values before merge."
    )
    delta_normalizer_strip_nested_properties: bool = Field(
        default=True, description="Drop nested properties from delta IR nodes/relationships."
    )
    delta_normalizer_attach_provenance: bool = Field(
        default=True, description="Attach batch/chunk provenance to normalized delta IR."
    )
    delta_resolvers_enabled: bool = Field(
        default=True, description="Enable optional post-merge delta duplicate resolvers."
    )
    delta_resolvers_mode: Literal["off", "fuzzy", "semantic", "chain"] = Field(
        default="semantic", description="Resolver mode for delta post-merge dedup."
    )
    delta_resolver_fuzzy_threshold: float = Field(
        default=0.8, description="Similarity threshold for fuzzy post-merge dedup."
    )
    delta_resolver_semantic_threshold: float = Field(
        default=0.8, description="Similarity threshold for semantic post-merge dedup."
    )
    delta_resolver_properties: list[str] | None = Field(
        default=None,
        description="Optional list of property names used for resolver matching.",
    )
    delta_resolver_paths: list[str] | None = Field(
        default=None, description="Optional path allowlist for resolver matching."
    )
    delta_resolver_allow_merge_different_ids: bool = Field(
        default=False,
        description="If True, allow resolver to merge nodes with different non-empty ids (content similarity decides). If False, do not merge when both have distinct identity strings.",
    )
    quality_max_unknown_path_drops: int = Field(
        default=-1,
        description="Maximum allowed unknown-path drops before delta quality gate fails (-1 disables).",
    )
    quality_max_id_mismatch: int = Field(
        default=-1,
        description="Maximum allowed ID key mismatches before delta quality gate fails (-1 disables).",
    )
    quality_max_nested_property_drops: int = Field(
        default=-1,
        description="Maximum allowed nested property drops before delta quality gate fails (-1 disables).",
    )
    delta_quality_require_root: bool = Field(
        default=True, description="Require root instance in delta quality gate."
    )
    delta_quality_min_instances: int = Field(
        default=20,
        description="Minimum attached nodes required by delta quality gate; below this, fall back to direct extraction.",
    )
    delta_quality_max_parent_lookup_miss: int = Field(
        default=4,
        description=(
            "Maximum allowed parent lookup misses before delta quality gate fails "
            "(higher default for weak local models)."
        ),
    )
    delta_quality_adaptive_parent_lookup: bool = Field(
        default=True,
        description="Enable adaptive parent_lookup_miss tolerance for delta quality gate.",
    )
    delta_quality_min_non_empty_properties: int = Field(
        default=-1,
        description=(
            "Minimum non-empty merged node properties required by delta quality gate (-1 disables)."
        ),
    )
    delta_quality_min_root_non_empty_fields: int = Field(
        default=-1,
        description=(
            "Minimum non-empty scalar root fields required by delta quality gate (-1 disables)."
        ),
    )
    delta_quality_min_non_empty_by_path: dict[str, int] | None = Field(
        default=None,
        description=(
            "Optional minimum non-empty property counts per catalog path for delta quality gate "
            "(e.g. {'line_items[]': 3})."
        ),
    )
    delta_quality_max_orphan_ratio: float = Field(
        default=-1.0,
        description="Maximum allowed orphan ratio before delta quality gate fails (-1 disables).",
    )
    delta_quality_max_canonical_duplicates: int = Field(
        default=-1,
        description=(
            "Maximum allowed duplicate canonical identities across paths before delta quality gate fails "
            "(-1 disables)."
        ),
    )
    delta_batch_split_max_retries: int = Field(
        default=1,
        description="Maximum split-retry rounds for failed delta batches.",
    )
    delta_identity_filter_enabled: bool = Field(
        default=True,
        description="Drop entity nodes whose identity value is not in schema allowlist or looks like a section title.",
    )
    delta_identity_filter_strict: bool = Field(
        default=False,
        description="If True, drop any entity node whose identity is not in allowlist; if False, also drop when section-title heuristic matches.",
    )
    staged_nodes_fill_cap: int | None = Field(
        default=None, description="Max nodes per LLM call in fill pass (None = use preset)."
    )
    staged_id_shard_size: int | None = Field(
        default=None, description="Paths per ID-pass call; 0 = single call (None = use preset)."
    )

    staged_id_identity_only: bool = Field(
        default=True, description="Discover only identity-bearing paths in ID pass."
    )
    staged_id_compact_prompt: bool = Field(
        default=True, description="Use compact ID-pass prompt to reduce token pressure."
    )
    staged_id_auto_shard_threshold: int = Field(
        default=10, description="Auto-shard ID pass when catalog paths exceed this threshold."
    )
    staged_id_shard_min_size: int = Field(
        default=2, description="Minimum split size when retrying failed ID shards."
    )
    staged_quality_require_root: bool = Field(
        default=True, description="Require root descriptor in staged quality gate."
    )
    staged_quality_min_instances: int = Field(
        default=1, description="Minimum ID instances required by staged quality gate."
    )
    staged_quality_max_parent_lookup_miss: int = Field(
        default=0, description="Maximum allowed parent lookup misses before fallback."
    )
    staged_id_max_tokens: int | None = Field(
        default=None, description="Optional max_tokens override for staged ID calls."
    )
    staged_fill_max_tokens: int | None = Field(
        default=None, description="Optional max_tokens override for staged fill calls."
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
        (
            effective_retries,
            effective_workers,
            effective_nodes_fill_cap,
            effective_id_shard_size,
        ) = get_effective_staged_tuning(
            self.staged_tuning_preset,
            self.staged_pass_retries,
            self.parallel_workers,
            self.staged_nodes_fill_cap,
            self.staged_id_shard_size,
        )
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
            "llm_batch_token_size": self.llm_batch_token_size,
            "debug": self.debug,
            "model_override": self.model_override,
            "provider_override": self.provider_override,
            "staged_tuning_preset": self.staged_tuning_preset,
            "staged_pass_retries": effective_retries,
            "parallel_workers": effective_workers,
            "delta_normalizer_validate_paths": self.delta_normalizer_validate_paths,
            "delta_normalizer_canonicalize_ids": self.delta_normalizer_canonicalize_ids,
            "delta_normalizer_strip_nested_properties": self.delta_normalizer_strip_nested_properties,
            "delta_normalizer_attach_provenance": self.delta_normalizer_attach_provenance,
            "delta_resolvers_enabled": self.delta_resolvers_enabled,
            "delta_resolvers_mode": self.delta_resolvers_mode,
            "delta_resolver_fuzzy_threshold": self.delta_resolver_fuzzy_threshold,
            "delta_resolver_semantic_threshold": self.delta_resolver_semantic_threshold,
            "delta_resolver_properties": self.delta_resolver_properties,
            "delta_resolver_paths": self.delta_resolver_paths,
            "delta_resolver_allow_merge_different_ids": self.delta_resolver_allow_merge_different_ids,
            "quality_max_unknown_path_drops": self.quality_max_unknown_path_drops,
            "quality_max_id_mismatch": self.quality_max_id_mismatch,
            "quality_max_nested_property_drops": self.quality_max_nested_property_drops,
            "delta_quality_require_root": self.delta_quality_require_root,
            "delta_quality_min_instances": self.delta_quality_min_instances,
            "delta_quality_max_parent_lookup_miss": self.delta_quality_max_parent_lookup_miss,
            "delta_quality_adaptive_parent_lookup": self.delta_quality_adaptive_parent_lookup,
            "delta_quality_min_non_empty_properties": self.delta_quality_min_non_empty_properties,
            "delta_quality_min_root_non_empty_fields": self.delta_quality_min_root_non_empty_fields,
            "delta_quality_min_non_empty_by_path": self.delta_quality_min_non_empty_by_path,
            "delta_quality_max_orphan_ratio": self.delta_quality_max_orphan_ratio,
            "delta_quality_max_canonical_duplicates": self.delta_quality_max_canonical_duplicates,
            "delta_batch_split_max_retries": self.delta_batch_split_max_retries,
            "delta_identity_filter_enabled": self.delta_identity_filter_enabled,
            "delta_identity_filter_strict": self.delta_identity_filter_strict,
            "staged_nodes_fill_cap": effective_nodes_fill_cap,
            "staged_id_shard_size": effective_id_shard_size,
            "staged_id_identity_only": self.staged_id_identity_only,
            "staged_id_compact_prompt": self.staged_id_compact_prompt,
            "staged_id_auto_shard_threshold": self.staged_id_auto_shard_threshold,
            "staged_id_shard_min_size": self.staged_id_shard_min_size,
            "staged_quality_require_root": self.staged_quality_require_root,
            "staged_quality_min_instances": self.staged_quality_min_instances,
            "staged_quality_max_parent_lookup_miss": self.staged_quality_max_parent_lookup_miss,
            "staged_id_max_tokens": self.staged_id_max_tokens,
            "staged_fill_max_tokens": self.staged_fill_max_tokens,
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
                "llm_batch_token_size": default_config.llm_batch_token_size,
                "structured_output": default_config.structured_output,
                "structured_sparse_check": default_config.structured_sparse_check,
                "staged_tuning_preset": default_config.staged_tuning_preset,
                "staged_pass_retries": default_config.staged_pass_retries,
                "parallel_workers": default_config.parallel_workers,
                "delta_normalizer_validate_paths": default_config.delta_normalizer_validate_paths,
                "delta_normalizer_canonicalize_ids": default_config.delta_normalizer_canonicalize_ids,
                "delta_normalizer_strip_nested_properties": default_config.delta_normalizer_strip_nested_properties,
                "delta_normalizer_attach_provenance": default_config.delta_normalizer_attach_provenance,
                "delta_resolvers_enabled": default_config.delta_resolvers_enabled,
                "delta_resolvers_mode": default_config.delta_resolvers_mode,
                "delta_resolver_fuzzy_threshold": default_config.delta_resolver_fuzzy_threshold,
                "delta_resolver_semantic_threshold": default_config.delta_resolver_semantic_threshold,
                "delta_resolver_properties": default_config.delta_resolver_properties,
                "delta_resolver_paths": default_config.delta_resolver_paths,
                "delta_resolver_allow_merge_different_ids": default_config.delta_resolver_allow_merge_different_ids,
                "quality_max_unknown_path_drops": default_config.quality_max_unknown_path_drops,
                "quality_max_id_mismatch": default_config.quality_max_id_mismatch,
                "quality_max_nested_property_drops": default_config.quality_max_nested_property_drops,
                "delta_quality_require_root": default_config.delta_quality_require_root,
                "delta_quality_min_instances": default_config.delta_quality_min_instances,
                "delta_quality_max_parent_lookup_miss": default_config.delta_quality_max_parent_lookup_miss,
                "delta_quality_adaptive_parent_lookup": default_config.delta_quality_adaptive_parent_lookup,
                "delta_quality_min_non_empty_properties": default_config.delta_quality_min_non_empty_properties,
                "delta_quality_min_root_non_empty_fields": default_config.delta_quality_min_root_non_empty_fields,
                "delta_quality_min_non_empty_by_path": default_config.delta_quality_min_non_empty_by_path,
                "delta_quality_max_orphan_ratio": default_config.delta_quality_max_orphan_ratio,
                "delta_quality_max_canonical_duplicates": default_config.delta_quality_max_canonical_duplicates,
                "delta_batch_split_max_retries": default_config.delta_batch_split_max_retries,
                "delta_identity_filter_enabled": default_config.delta_identity_filter_enabled,
                "delta_identity_filter_strict": default_config.delta_identity_filter_strict,
                "staged_nodes_fill_cap": default_config.staged_nodes_fill_cap,
                "staged_id_shard_size": default_config.staged_id_shard_size,
                "staged_id_identity_only": default_config.staged_id_identity_only,
                "staged_id_compact_prompt": default_config.staged_id_compact_prompt,
                "staged_id_auto_shard_threshold": default_config.staged_id_auto_shard_threshold,
                "staged_id_shard_min_size": default_config.staged_id_shard_min_size,
                "staged_quality_require_root": default_config.staged_quality_require_root,
                "staged_quality_min_instances": default_config.staged_quality_min_instances,
                "staged_quality_max_parent_lookup_miss": default_config.staged_quality_max_parent_lookup_miss,
                "staged_id_max_tokens": default_config.staged_id_max_tokens,
                "staged_fill_max_tokens": default_config.staged_fill_max_tokens,
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
