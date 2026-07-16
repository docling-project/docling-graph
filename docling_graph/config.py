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
    extraction_contract: Literal["direct", "dense", "auto"] = Field(default="auto")
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
    # Default "auto": running direct on a document that dwarfs the model's
    # output budget silently self-rations (verified: 36k-char paper direct
    # scored node-F1 0.04 vs 0.14 dense), and on huge documents it burns
    # minutes of doomed calls. Auto costs nothing on small documents (resolves
    # to direct) and the decision is logged per document.
    extraction_contract: Literal["direct", "dense", "auto"] = Field(
        default="auto",
        description=(
            "Extraction contract: 'auto' (default — per document, picks direct when a "
            "single call fits the model's context window and output budget, dense "
            "otherwise), 'direct' (always one full-document call), or 'dense' "
            "(always skeleton-then-fill over chunks)."
        ),
    )

    # Docling settings (with defaults)
    docling_config: Literal["ocr", "vision"] = Field(default="ocr")

    # Remote conversion via docling-serve. When a URL is set (directly or via
    # the DOCLING_SERVE_URL env var), document conversion is delegated to that
    # instance and no local conversion models are loaded. docling_config still
    # selects the server-side pipeline ('ocr' -> standard, 'vision' -> vlm).
    docling_serve_url: str | None = Field(
        default=None,
        description=(
            "Base URL of a docling-serve instance (e.g. 'http://localhost:5001'). "
            "When set, document conversion runs remotely on that instance instead "
            "of locally. Falls back to the DOCLING_SERVE_URL environment variable."
        ),
    )
    docling_serve_api_key: str | None = Field(
        default=None,
        description=(
            "API key for the docling-serve instance, sent as the X-Api-Key header. "
            "Falls back to the DOCLING_SERVE_API_KEY environment variable."
        ),
    )
    docling_serve_timeout: int = Field(
        default=300,
        description=(
            "Approximate deadline in seconds for one document's remote "
            "conversion, from submission to terminal status. Conversion runs "
            "asynchronously on the server; the client polls until this "
            "deadline, and server queue time counts toward it. Connect/read "
            "timeouts and transient-error retries are bounded separately. On "
            "timeout, the job may still be running server-side."
        ),
    )
    docling_serve_headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra HTTP headers sent on every docling-serve request, e.g. "
            '{"Authorization": "Bearer <token>"} for deployments that do not '
            "use X-Api-Key auth. Falls back to the DOCLING_SERVE_HEADERS "
            "environment variable (a JSON object string). Treated as a "
            "secret: never written to metadata.json."
        ),
    )

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

    # LLM input serialization: how the document text is rendered for the LLM.
    # 'auto' (default) pairs the format to the resolved extraction contract per
    # document (direct -> doclang-geo, dense -> doclang; raw text inputs ->
    # markdown) — the benchmark-validated pairing. Fixed values pin one
    # serialization: 'markdown' is the cheapest baseline; 'doclang'/'doclang-geo'
    # render DocLang XML (structure + optional geometry) at a higher token cost.
    llm_input_format: Literal["markdown", "doclang", "doclang-geo", "auto"] = Field(
        default="auto",
        description=(
            "Serialization of document text sent to the LLM: 'auto' (default; pairs "
            "the format to the resolved extraction contract: direct->doclang-geo, "
            "dense->doclang, raw text->markdown), 'markdown' (cheapest baseline), "
            "'doclang' (DocLang XML, structure only), or 'doclang-geo' (DocLang XML "
            "with page-coordinate geometry). DocLang costs more tokens; benchmark "
            "before pinning a fixed format."
        ),
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

    # Parallel workers for extraction
    parallel_workers: int | None = Field(
        default=None,
        description="Parallel workers for extraction.",
    )

    # Gleaning (direct contract): one optional "what did you miss?" pass.
    gleaning_enabled: bool = Field(
        default=True,
        description="Run a second-pass extraction (what did you miss?) for the direct contract.",
    )

    # Dense contract tuning: sizing knobs plus one intent-driven dedupe mode.
    # Mandatory cleanup (root singleton, barren-branch pruning, quality gate)
    # are pipeline invariants and intentionally not configurable.
    dense_skeleton_batch_tokens: int = Field(
        default=2048,
        description="Max tokens per dense Phase 1 (skeleton) chunk batch.",
    )
    dense_fill_nodes_cap: int = Field(
        default=5,
        description="Max node instances per dense Phase 2 (fill) LLM call.",
    )
    dense_fill_context: Literal["scoped", "full"] = Field(
        default="scoped",
        description=(
            "Document context sent with each dense fill call: 'scoped' sends only the "
            "skeleton batches where the node was observed (plus document head); "
            "'full' always sends the whole document."
        ),
    )
    dense_dedupe: Literal["off", "standard", "aggressive"] = Field(
        default="standard",
        description=(
            "Skeleton dedupe intensity. 'off': exact canonical-id dedup only. "
            "'standard': adds one id-space LLM reconciliation call that collapses "
            "same-entity aliases found at different granularities. 'aggressive': "
            "also merges near-identical same-path identifier strings (OCR noise); "
            "similarity thresholds are handled internally."
        ),
    )

    # Data grounding / provenance: deterministic node-to-source mapping.
    # Fully out-of-band (no prompt or LLM-output changes at any level).
    provenance: Literal["off", "standard", "detailed"] = Field(
        default="standard",
        description=(
            "Deterministic provenance ledger mapping graph nodes to source "
            "chunks/pages. 'off': disabled. 'standard': node annotations + "
            "provenance.json, with an exact verbatim-identifier locator "
            "(precise chunk/page) and an approximate observed fallback. "
            "'detailed': also embeds character spans in the node annotation."
        ),
    )
    # Export settings (with defaults)
    export_format: Literal["csv", "cypher"] = Field(default="csv")
    export_docling: bool = Field(default=True)
    export_docling_json: bool = Field(default=True)
    export_markdown: bool = Field(default=True)
    export_doclang: bool = Field(
        default=True,
        description="Export the Docling document as DocLang (.dclg) — content+geometry interchange.",
    )
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

    @field_validator("docling_serve_url")
    @classmethod
    def _normalize_serve_url(cls, v: str | None) -> str | None:
        return cls._clean_serve_url(v)

    @staticmethod
    def _clean_serve_url(v: str | None) -> str | None:
        """Normalize a docling-serve URL; empty -> None, validate scheme."""
        if v is None:
            return None
        v = v.strip().rstrip("/")
        if not v:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"docling_serve_url must start with http:// or https://, got: {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_vlm_constraints(self) -> Self:
        if self.backend == "vlm" and self.inference == "remote":
            raise ValueError(
                "VLM backend currently only supports local inference. Use inference='local' or backend='llm'."
            )
        return self

    @model_validator(mode="after")
    def _resolve_docling_serve_env(self) -> Self:
        """Fill docling-serve settings from environment variables.

        Explicit values (config file, CLI flag, constructor) win; the env vars
        DOCLING_SERVE_URL / DOCLING_SERVE_API_KEY / DOCLING_SERVE_HEADERS are
        fallbacks so cluster deployments can enable remote conversion without
        touching configs.
        """
        import json
        import os

        if not self.docling_serve_url:
            self.docling_serve_url = self._clean_serve_url(os.environ.get("DOCLING_SERVE_URL"))
        if self.docling_serve_url and not self.docling_serve_api_key:
            self.docling_serve_api_key = os.environ.get("DOCLING_SERVE_API_KEY") or None
        if self.docling_serve_url and not self.docling_serve_headers:
            raw_headers = os.environ.get("DOCLING_SERVE_HEADERS")
            if raw_headers:
                # Auth material: a malformed value must fail loudly, not be
                # silently dropped (requests would go out unauthenticated).
                try:
                    parsed = json.loads(raw_headers)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"DOCLING_SERVE_HEADERS must be a JSON object string: {e}"
                    ) from e
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"DOCLING_SERVE_HEADERS must be a JSON object, got {type(parsed).__name__}"
                    )
                self.docling_serve_headers = {str(k): str(v) for k, v in parsed.items()}
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
        # and secrets (docling-serve API key and custom auth headers must
        # never land in metadata.json)
        data = self.model_dump(
            mode="json",
            exclude={"llm_client", "template", "docling_serve_api_key", "docling_serve_headers"},
        )
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
            "docling_serve_url": self.docling_serve_url,
            "docling_serve_api_key": self.docling_serve_api_key,
            "docling_serve_timeout": self.docling_serve_timeout,
            "docling_serve_headers": self.docling_serve_headers,
            "structured_output": self.structured_output,
            "structured_sparse_check": self.structured_sparse_check,
            "llm_input_format": self.llm_input_format,
            "use_chunking": self.use_chunking,
            "chunk_max_tokens": self.chunk_max_tokens,
            "debug": self.debug,
            "model_override": self.model_override,
            "provider_override": self.provider_override,
            "parallel_workers": self.parallel_workers,
            "gleaning_enabled": self.gleaning_enabled,
            "dense_skeleton_batch_tokens": self.dense_skeleton_batch_tokens,
            "dense_fill_nodes_cap": self.dense_fill_nodes_cap,
            "dense_fill_context": self.dense_fill_context,
            "dense_dedupe": self.dense_dedupe,
            "provenance": self.provenance,
            "export_format": self.export_format,
            "export_docling": self.export_docling,
            "export_docling_json": self.export_docling_json,
            "export_markdown": self.export_markdown,
            "export_doclang": self.export_doclang,
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
                "llm_input_format": default_config.llm_input_format,
                "chunk_max_tokens": default_config.chunk_max_tokens,
                "structured_output": default_config.structured_output,
                "structured_sparse_check": default_config.structured_sparse_check,
                "parallel_workers": default_config.parallel_workers,
                "gleaning_enabled": default_config.gleaning_enabled,
                "dense_skeleton_batch_tokens": default_config.dense_skeleton_batch_tokens,
                "dense_fill_nodes_cap": default_config.dense_fill_nodes_cap,
                "dense_fill_context": default_config.dense_fill_context,
                "dense_dedupe": default_config.dense_dedupe,
                "provenance": default_config.provenance,
            },
            "docling": {
                "pipeline": default_config.docling_config,
                # Remote conversion via docling-serve; keep url null for local
                # conversion. The api key is read from DOCLING_SERVE_API_KEY;
                # extra auth headers (e.g. Authorization: Bearer) come from
                # DOCLING_SERVE_HEADERS (JSON object string).
                "serve": {
                    "url": default_config.docling_serve_url,
                    "timeout": default_config.docling_serve_timeout,
                },
                "export": {
                    "docling_json": default_config.export_docling_json,
                    "markdown": default_config.export_markdown,
                    "doclang": default_config.export_doclang,
                    "per_page_markdown": default_config.export_per_page_markdown,
                },
            },
            "models": default_config.models.model_dump(),
            "llm_overrides": default_config.llm_overrides.model_dump(),
            "output": {
                "directory": str(default_config.output_dir),
            },
        }
