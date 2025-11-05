"""
Adaptive chunk batching for efficient LLM extraction.

Groups multiple chunks into batches that fit within context window,
reducing API calls while preserving semantic boundaries.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from rich import print as rich_print


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
    """Intelligently batch chunks to optimize context window usage."""

    # Overhead per chunk when batched (metadata, separators, etc.)
    CHUNK_OVERHEAD_TOKENS = 50

    def __init__(
        self,
        context_limit: int,
        system_prompt_tokens: int = 500,
        response_buffer_tokens: int = 500,
        merge_threshold: float = 0.85,
    ) -> None:
        """
        Initialize batcher with context constraints.

        Args:
            context_limit: Total context window (e.g., 3500 for granite-4.0-1b)
            system_prompt_tokens: Tokens for system prompt (default: 500)
            response_buffer_tokens: Tokens reserved for response (default: 500)
            merge_threshold: Merge chunks if batch is <this% of available context
                (default: 0.85 = 85%, prevents many tiny batches)
        """
        self.context_limit = context_limit
        self.system_prompt_tokens = system_prompt_tokens
        self.response_buffer_tokens = response_buffer_tokens
        self.merge_threshold = merge_threshold

        # Available tokens for content
        self.available_tokens = context_limit - system_prompt_tokens - response_buffer_tokens

        rich_print(
            f"[blue][ChunkBatcher][/blue] Initialized with:\n"
            f" • Context limit: [yellow]{context_limit:,}[/yellow] tokens\n"
            f" • Available for content: [cyan]{self.available_tokens:,}[/cyan] tokens\n"
            f" • Merge threshold: {merge_threshold * 100:.0f}%"
        )

    def batch_chunks(
        self,
        chunks: List[str],
        tokenizer_fn: Optional[Callable[[str], int]] = None,
    ) -> List[ChunkBatch]:
        """
        Batch chunks to fit context window efficiently.

        Strategy:
        1. Group chunks that fit together in context
        2. Merge undersized batches if below merge_threshold
        3. Minimize total number of API calls

        Args:
            chunks: List of chunk texts
            tokenizer_fn: Optional function to count tokens accurately.
                If None, uses rough heuristic (tokens ≈ chars / 4)

        Returns:
            List of ChunkBatch objects ready for LLM extraction
        """
        if not chunks:
            return []

        # Helper: estimate tokens for a string
        def count_tokens(text: str) -> int:
            if tokenizer_fn:
                try:
                    return tokenizer_fn(text)
                except Exception:
                    pass
            # Fallback: rough heuristic
            return len(text) // 4

        # Phase 1: Create candidate batches (greedy packing)
        batches: List[ChunkBatch] = []
        current_batch_chunks: List[str] = []
        current_batch_indices: List[int] = []
        current_tokens = 0

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_tokens = count_tokens(chunk_text) + self.CHUNK_OVERHEAD_TOKENS

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
        """Log batching statistics."""
        reduction = (total_chunks - len(batches)) / max(1, total_chunks) * 100
        avg_batch_size = sum(b.chunk_count for b in batches) / max(1, len(batches))
        avg_utilization = (
            total_tokens / (len(batches) * self.available_tokens) * 100 if batches else 0
        )

        rich_print(
            f"[blue][ChunkBatcher][/blue] Batching summary:\n"
            f"  • Total chunks: [cyan]{total_chunks}[/cyan]\n"
            f"  • Batches created: [yellow]{len(batches)}[/yellow] ([green]-{reduction:.0f}%[/green] API calls)\n"
            f"  • Avg chunks/batch: {avg_batch_size:.1f}\n"
            f"  • Context utilization: {avg_utilization:.1f}%"
        )

        # Log per-batch details
        for batch in batches:
            utilization = batch.total_tokens / self.available_tokens * 100
            rich_print(
                f"  └─ Batch {batch.batch_id}: "
                f"{batch.chunk_count} chunks "
                f"({batch.total_tokens:,} tokens, {utilization:.0f}% utilized)"
            )
