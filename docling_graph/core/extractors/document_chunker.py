"""
Structure-preserving document chunker using Docling's HybridChunker.

Preserves:
- Tables (not split across chunks)
- Lists (kept intact)
- Hierarchical structure (sections with headers)
- Semantic boundaries

Configurable per LLM provider tokenizer.
"""

from typing import List, Optional, Union

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument
from rich import print as rich_print
from transformers import AutoTokenizer

from ...llm_clients.config import (
    get_recommended_chunk_size,
    get_tokenizer_for_provider,
)


class DocumentChunker:
    """Structure-preserving document chunker using Docling's HybridChunker."""

    def __init__(
        self,
        tokenizer_name: str | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
        merge_peers: bool = True,
        schema_size: int = 0,
    ) -> None:
        """
        Initialize the chunker with smart defaults based on provider or custom tokenizer.

        Now uses centralized llm_config.py registry with dynamic adjustment based on schema complexity.

        Args:
            tokenizer_name: Name of the tokenizer to use
            max_tokens: Maximum tokens per chunk (if None, calculated from provider)
            provider: LLM provider name (e.g., "watsonx", "openai")
            merge_peers: Whether to merge peer sections in chunking
            schema_size: Size of Pydantic schema JSON for dynamic chunk sizing
        """
        self.tokenizer: Union[HuggingFaceTokenizer, OpenAITokenizer]

        # Step 1: Determine tokenizer name
        if tokenizer_name is None and provider is not None:
            tokenizer_name = get_tokenizer_for_provider(provider)

        elif tokenizer_name is None:
            tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Step 2: Determine max_tokens (using centralized lookup with schema awareness)
        if max_tokens is None:
            if provider is not None:
                max_tokens = get_recommended_chunk_size(provider, "", schema_size)
            else:
                max_tokens = 5120

        # Step 3: Initialize tokenizer and chunker
        if tokenizer_name != "tiktoken":
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=max_tokens,
            )
        else:
            # Special handling for OpenAI tiktoken
            try:
                import tiktoken

                tt_tokenizer = tiktoken.encoding_for_model("gpt-4o")
                self.tokenizer = OpenAITokenizer(
                    tokenizer=tt_tokenizer,
                    max_tokens=max_tokens,
                )
            except ImportError:
                rich_print(
                    "[yellow][DocumentChunker][/yellow] tiktoken not installed, "
                    "falling back to HuggingFace tokenizer"
                )
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.tokenizer = HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=max_tokens,
                )

        # Step 4: Create HybridChunker instance
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=merge_peers,
        )

        self.max_tokens = max_tokens
        self.original_max_tokens = max_tokens  # Store original for schema adjustments
        self.tokenizer_name = tokenizer_name
        self.merge_peers = merge_peers

        rich_print(
            f"[blue][DocumentChunker][/blue] Initialized with:\n"
            f" • Tokenizer: [cyan]{tokenizer_name}[/cyan]\n"
            f" • Max tokens/chunk: [yellow]{max_tokens}[/yellow]\n"
            f" • Merge peers: {merge_peers}"
        )

    def update_schema_config(self, schema_size: int) -> None:
        """
        Update chunker configuration based on schema size.

        Adjusts max_tokens to reserve space for schema in context window,
        preventing context overflow when schema is large.

        Args:
            schema_size: Size of the JSON schema in bytes
        """
        import logging

        logger = logging.getLogger(__name__)

        if not self.tokenizer:
            logger.warning("No tokenizer available for schema config update")
            return

        # Estimate schema tokens (conservative: 3.5 chars per token)
        schema_tokens = int(schema_size / 3.5)

        # Adjust max_tokens to reserve space for schema
        # Keep at least 50% of original capacity
        min_tokens = int(self.original_max_tokens * 0.5)
        adjusted_max = self.original_max_tokens - schema_tokens

        if adjusted_max < min_tokens:
            logger.warning(
                f"Schema is very large ({schema_tokens} tokens, {schema_size} bytes). "
                f"Reducing chunk size from {self.original_max_tokens} to minimum {min_tokens}"
            )
            self.max_tokens = min_tokens
        elif adjusted_max < self.original_max_tokens:
            logger.info(
                f"Adjusted chunk size from {self.original_max_tokens} to {adjusted_max} "
                f"to accommodate schema ({schema_tokens} tokens)"
            )
            self.max_tokens = adjusted_max
        else:
            # Schema is small, no adjustment needed
            self.max_tokens = self.original_max_tokens

        # Update the tokenizer's max_tokens
        if hasattr(self.tokenizer, "max_tokens"):
            self.tokenizer.max_tokens = self.max_tokens

        # Update the chunker's tokenizer max_tokens as well
        if hasattr(self.chunker, "tokenizer") and hasattr(self.chunker.tokenizer, "max_tokens"):
            self.chunker.tokenizer.max_tokens = self.max_tokens

        # Note: chunker.max_tokens is a read-only property derived from tokenizer.max_tokens
        # so we don't need to (and can't) set it directly

        rich_print(
            f"[blue][DocumentChunker][/blue] Schema config updated:\n"
            f" • Schema size: {schema_size} bytes (~{schema_tokens} tokens)\n"
            f" • Adjusted max_tokens: {self.max_tokens} (was {self.original_max_tokens})"
        )

    @staticmethod
    def calculate_recommended_max_tokens(
        context_limit: int,
        system_prompt_tokens: int = 500,
        response_buffer_tokens: int = 500,
    ) -> int:
        """
        Calculate recommended max_tokens for a given context window.

        Formula:
        available = context_limit - system_prompt - response_buffer
        max_tokens = available * 0.8  # Reserve 20% for metadata enrichment

        Args:
            context_limit: Total context window (e.g., 8000 for Mistral-Large)
            system_prompt_tokens: Estimated tokens for system prompt (default: 500)
            response_buffer_tokens: Space reserved for LLM output (default: 500)

        Returns:
            Recommended max_tokens value for chunker
        """
        available = context_limit - system_prompt_tokens - response_buffer_tokens
        recommended = int(available * 0.8)
        return max(512, recommended)  # Minimum 512 tokens

    def chunk_document(self, document: DoclingDocument) -> List[str]:
        """
        Chunk a DoclingDocument into structure-aware text chunks.

        Args:
            document: Parsed DoclingDocument from DocumentConverter

        Returns:
            List of contextualized text chunks, ready for LLM consumption
        """
        chunks = []

        # Chunk the document using HybridChunker
        chunk_iter = self.chunker.chunk(dl_doc=document)

        for chunk in chunk_iter:
            # Use contextualized text (includes metadata like headers, section captions)
            # This is essential for LLM extraction to understand chunk context
            enriched_text = self.chunker.contextualize(chunk=chunk)
            chunks.append(enriched_text)

        return chunks

    def chunk_document_with_stats(self, document: DoclingDocument) -> tuple[List[str], dict]:
        """
        Chunk document and return tokenization statistics.
        Useful for debugging/optimization to understand chunk distribution.

        Args:
            document: Parsed DoclingDocument

        Returns:
            Tuple of (chunks, stats) where stats contains:
            - total_chunks: number of chunks
            - chunk_tokens: list of token counts per chunk
            - avg_tokens: average tokens per chunk
            - max_tokens_in_chunk: maximum tokens in any chunk
            - total_tokens: sum of all chunk tokens
        """
        chunks = []
        chunk_tokens = []

        chunk_iter = self.chunker.chunk(dl_doc=document)

        for chunk in chunk_iter:
            enriched_text = self.chunker.contextualize(chunk=chunk)
            chunks.append(enriched_text)

            # Count tokens for this chunk
            num_tokens = self.tokenizer.count_tokens(enriched_text)
            chunk_tokens.append(num_tokens)

        stats = {
            "total_chunks": len(chunks),
            "chunk_tokens": chunk_tokens,
            "avg_tokens": sum(chunk_tokens) / len(chunk_tokens) if chunk_tokens else 0,
            "max_tokens_in_chunk": max(chunk_tokens) if chunk_tokens else 0,
            "total_tokens": sum(chunk_tokens),
        }

        return chunks, stats

    def chunk_text_fallback(self, text: str) -> List[str]:
        """
        Fallback chunker for raw text when DoclingDocument unavailable.

        This is a simple token-based splitter that respects sentence boundaries.
        For best results, always use chunk_document() with a DoclingDocument.

        Args:
            text: Raw text string (e.g., plain Markdown)

        Returns:
            List of text chunks
        """
        # Rough heuristic: 1 token ≈ 4 characters for most tokenizers
        max_chars = self.max_tokens * 4

        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + max_chars, len(text))

            # Try to break at sentence/semantic boundary
            if end_pos < len(text):
                # Priority order for breaking points
                for delimiter in [". ", "! ", "? ", "\n\n", "\n"]:
                    last_break = text.rfind(delimiter, current_pos, end_pos)
                    if last_break != -1:
                        end_pos = last_break + len(delimiter)
                        break

            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos

        return chunks

    def get_config_summary(self) -> dict:
        """Get current chunker configuration as dictionary."""
        return {
            "tokenizer_name": self.tokenizer_name,
            "max_tokens": self.max_tokens,
            "merge_peers": self.merge_peers,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }
