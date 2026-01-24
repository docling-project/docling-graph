"""
Adaptive chunk batching for efficient LLM extraction.

Groups multiple chunks into batches that fit within context window,
reducing API calls while preserving semantic boundaries.

Integrates with real tokenizers from DocumentChunker for accurate token counting.
Supports provider-specific configurations from centralized registry.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List

from rich import print as rich_print

from docling_graph.llm_clients.config import ProviderConfig, get_provider_config

logger = logging.getLogger(__name__)


@dataclass
class ChunkBatch:
    """A batch of chunks to send to LLM in a single call."""

    batch_id: int
    """Batch sequence number."""

    chunks: List[str]
    """List of chunk texts in this batch."""

    total_tokens: int
    """Estimated total tokens in batch (including prompt/overhead)."""

    chunk_indices: List[int]
    """Original chunk indices from document."""

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def combined_text(self) -> str:
        """Combine all chunks with separators for LLM."""
        separator = "\n\n---CHUNK BOUNDARY---\n\n"
        return separator.join(
            [f"[Chunk {i + 1}/{len(self.chunks)}]\n{chunk}" for i, chunk in enumerate(self.chunks)]
        )


class ChunkBatcher:
    """
    Intelligently batch chunks with real tokenizer integration.

    Supports provider-specific configurations from centralized registry.
    """

    # Conservative fallback ratios (only used if tokenizer fails)
    TOKENIZER_RATIOS = {
        "llama": 3.5,  # Llama/Mistral family
        "gpt": 4.0,  # GPT family
        "small_model": 2.5,  # 1B-5B models (more verbose tokenization)
        "multilingual": 3.0,  # Multilingual models
        "default": 3.0,  # Conservative default
    }

    SAFETY_MARGIN = 1.2  # 20% buffer for all estimates
    CHUNK_OVERHEAD_TOKENS = 50  # Overhead per chunk when batched

    def __init__(
        self,
        context_limit: int,
        system_prompt_tokens: int = 500,
        response_buffer_tokens: int = 500,
        merge_threshold: float | None = None,
        tokenizer_type: str = "default",
        provider: str | None = None,
    ) -> None:
        """
        Initialize batcher with context constraints and provider configuration.

        Args:
            context_limit: Total context window in tokens
            system_prompt_tokens: Tokens for system prompt (default: 500)
            response_buffer_tokens: Tokens reserved for response (default: 500)
            merge_threshold: Merge chunks if batch is <this% of available context
                (default: provider-specific or 0.85)
            tokenizer_type: Fallback tokenizer family (llama, gpt, small_model, etc.)
            provider: LLM provider name (openai, anthropic, google, etc.)
                     Used to apply provider-specific optimizations
        """
        self.context_limit = context_limit
        self.system_prompt_tokens = system_prompt_tokens
        self.response_buffer_tokens = response_buffer_tokens

        # Get provider configuration from centralized registry
        self.provider_name = provider or "unknown"
        self.provider_config = self._get_provider_config(self.provider_name)

        # Use provider-specific merge threshold if not explicitly set
        self.merge_threshold = (
            merge_threshold if merge_threshold is not None else self.provider_config.merge_threshold
        )

        # Fallback heuristic configuration
        self.char_per_token = self.TOKENIZER_RATIOS.get(
            tokenizer_type, self.TOKENIZER_RATIOS["default"]
        )

        # Available tokens for content
        self.available_tokens = context_limit - system_prompt_tokens - response_buffer_tokens

        rich_print(
            f"[blue][ChunkBatcher][/blue] Initialized with:\n"
            f" • Provider: [cyan]{self.provider_config.provider_id}[/cyan]\n"
            f" • Context limit: [yellow]{context_limit:,}[/yellow] tokens\n"
            f" • Available for content: [cyan]{self.available_tokens:,}[/cyan] tokens\n"
            f" • Merge threshold: {self.merge_threshold * 100:.0f}% "
            f"({'provider default' if merge_threshold is None else 'custom'})\n"
            f" • Fallback tokenizer: {tokenizer_type} ({self.char_per_token} chars/token)"
        )

    def _get_provider_config(self, provider: str) -> ProviderConfig:
        """
        Get provider configuration from centralized registry.

        Args:
            provider: Provider name (case-insensitive)

        Returns:
            ProviderConfig from registry, or default config if not found
        """
        if not provider:
            provider = "unknown"

        provider_lower = provider.lower()

        # Try direct lookup first
        config = get_provider_config(provider_lower)
        if config:
            return config

        # Try common name mappings
        provider_mappings = {
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "google",
            "watson": "watsonx",
        }

        for key, mapped_provider in provider_mappings.items():
            if key in provider_lower:
                config = get_provider_config(mapped_provider)
                if config:
                    return config

        # Return default config if provider not found
        logger.warning(f"Provider '{provider}' not found in registry, using default configuration")
        # Create a minimal default config
        from docling_graph.llm_clients.config import ProviderConfig as LLMProviderConfig

        return LLMProviderConfig(
            provider_id="unknown",
            models={},
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            content_ratio=0.8,
            merge_threshold=0.85,
            rate_limit_rpm=None,
            supports_batching=True,
        )

    def _estimate_tokens(
        self,
        text: str,
        tokenizer_fn: Callable[[str], int] | None = None,
    ) -> int:
        """
        Estimate tokens using real tokenizer or conservative fallback.

        Priority:
        1. Real tokenizer (from DocumentChunker)
        2. Conservative heuristic with safety margin

        Args:
            text: Text to estimate
            tokenizer_fn: Optional real tokenizer function

        Returns:
            Estimated token count with safety margin applied
        """
        # Try real tokenizer first
        if tokenizer_fn:
            try:
                tokens = tokenizer_fn(text)
                # Apply safety margin to real tokenizer too
                return int(tokens * self.SAFETY_MARGIN)
            except Exception as e:
                logger.warning(
                    f"Tokenizer function failed: {e}. Falling back to heuristic estimation."
                )

        # Fallback: conservative heuristic with safety margin
        estimated = int(len(text) / self.char_per_token * self.SAFETY_MARGIN)
        logger.debug(
            f"Using heuristic token estimation: {len(text)} chars → {estimated} tokens "
            f"(ratio: {self.char_per_token}, margin: {self.SAFETY_MARGIN})"
        )
        return estimated

    def batch_chunks(
        self,
        chunks: List[str],
        tokenizer_fn: Callable[[str], int] | None = None,
    ) -> List[ChunkBatch]:
        """
        Batch chunks to fit context window efficiently.

        Strategy:
        1. Group chunks that fit together in context
        2. Merge undersized batches if below merge_threshold
        3. Minimize total number of API calls

        Args:
            chunks: List of chunk texts
            tokenizer_fn: Optional real tokenizer function from DocumentChunker
                         (e.g., self.doc_processor.chunker.tokenizer.count_tokens)

        Returns:
            List of ChunkBatch objects ready for LLM extraction
        """
        if not chunks:
            return []

        # Log tokenizer source
        if tokenizer_fn:
            rich_print("[blue][ChunkBatcher][/blue] Using real tokenizer from DocumentChunker")
        else:
            rich_print(
                f"[yellow][ChunkBatcher][/yellow] No tokenizer provided, "
                f"using conservative heuristic ({self.char_per_token} chars/token)"
            )

        # Phase 1: Create candidate batches (greedy packing)
        batches: List[ChunkBatch] = []
        current_batch_chunks: List[str] = []
        current_batch_indices: List[int] = []
        current_tokens = 0

        for chunk_idx, chunk_text in enumerate(chunks):
            # Estimate tokens for this chunk (with overhead)
            chunk_tokens = (
                self._estimate_tokens(chunk_text, tokenizer_fn) + self.CHUNK_OVERHEAD_TOKENS
            )

            # Check if adding this chunk exceeds available context
            potential_total = current_tokens + chunk_tokens

            if current_batch_chunks and potential_total > self.available_tokens:
                # Start new batch
                batches.append(
                    ChunkBatch(
                        batch_id=len(batches),
                        chunks=current_batch_chunks.copy(),
                        total_tokens=current_tokens,
                        chunk_indices=current_batch_indices.copy(),
                    )
                )
                current_batch_chunks = [chunk_text]
                current_batch_indices = [chunk_idx]
                current_tokens = chunk_tokens
            else:
                # Add to current batch
                current_batch_chunks.append(chunk_text)
                current_batch_indices.append(chunk_idx)
                current_tokens = potential_total

        # Add final batch
        if current_batch_chunks:
            batches.append(
                ChunkBatch(
                    batch_id=len(batches),
                    chunks=current_batch_chunks,
                    total_tokens=current_tokens,
                    chunk_indices=current_batch_indices,
                )
            )

        # Phase 2: Merge undersized batches (if below threshold)
        merged_batches = self._merge_undersized_batches(batches)

        # Log summary
        self._log_batching_summary(
            total_chunks=len(chunks),
            batches=merged_batches,
            total_tokens=sum(b.total_tokens for b in merged_batches),
        )

        return merged_batches

    def _merge_undersized_batches(
        self,
        batches: List[ChunkBatch],
    ) -> List[ChunkBatch]:
        """
        Merge batches that are below merge_threshold of available context.

        This prevents many small API calls and improves context utilization.

        Args:
            batches: List of batches to potentially merge

        Returns:
            List of optimally merged batches
        """
        if len(batches) <= 1:
            return batches

        threshold_tokens = int(self.available_tokens * self.merge_threshold)
        merged: List[ChunkBatch] = []

        i = 0
        while i < len(batches):
            current = batches[i]

            # If batch is already large enough, keep it
            if current.total_tokens >= threshold_tokens:
                merged.append(current)
                i += 1
                continue

            # Try to merge with next batch(es)
            combined_chunks = current.chunks.copy()
            combined_indices = current.chunk_indices.copy()
            combined_tokens = current.total_tokens

            j = i + 1
            while j < len(batches):
                next_batch = batches[j]
                potential_total = combined_tokens + next_batch.total_tokens

                # Stop if merging would exceed context
                if potential_total > self.available_tokens:
                    break

                # Merge this batch
                combined_chunks.extend(next_batch.chunks)
                combined_indices.extend(next_batch.chunk_indices)
                combined_tokens = potential_total
                j += 1

            # Create merged batch
            merged.append(
                ChunkBatch(
                    batch_id=len(merged),
                    chunks=combined_chunks,
                    total_tokens=combined_tokens,
                    chunk_indices=combined_indices,
                )
            )

            i = j

        return merged

    def _log_batching_summary(
        self,
        total_chunks: int,
        batches: List[ChunkBatch],
        total_tokens: int,
    ) -> None:
        """Log batching statistics with cost information."""
        reduction = (total_chunks - len(batches)) / max(1, total_chunks) * 100
        avg_batch_size = sum(b.chunk_count for b in batches) / max(1, len(batches))
        avg_utilization = (
            total_tokens / (len(batches) * self.available_tokens) * 100 if batches else 0
        )

        summary = (
            f"[blue][ChunkBatcher][/blue] Batching summary:\n"
            f"  • Total chunks: [cyan]{total_chunks}[/cyan]\n"
            f"  • Batches created: [yellow]{len(batches)}[/yellow] ([green]-{reduction:.0f}%[/green] API calls)\n"
            f"  • Avg chunks/batch: {avg_batch_size:.1f}\n"
            f"  • Context utilization: {avg_utilization:.1f}%"
        )

        rich_print(summary)

        # Log per-batch details
        for batch in batches:
            utilization = batch.total_tokens / self.available_tokens * 100
            batch_info = (
                f"  └─ Batch {batch.batch_id}: "
                f"{batch.chunk_count} chunks "
                f"({batch.total_tokens:,} tokens, {utilization:.0f}% utilized)"
            )

            rich_print(batch_info)
