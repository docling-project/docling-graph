"""
Structure-preserving document chunker using Docling's HybridChunker.

- Single sizing knob: chunk_max_tokens (with sensible default)
- Always initializes tokenizer and chunker (no lazy defaults)

Preserves:
- Tables (not split across chunks)
- Lists (kept intact)
- Hierarchical structure (sections with headers)
- Semantic boundaries
"""

import logging
import re
from typing import List, Union

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument
from rich import print as rich_print
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Structure-preserving document chunker using Docling's HybridChunker.

    - Single sizing parameter: chunk_max_tokens (defaults to 512)
    - No coupling to model context limits or output budgets
    - Always initializes tokenizer and chunker
    """

    def __init__(
        self,
        tokenizer_name: str | None = None,
        chunk_max_tokens: int = 512,
        merge_peers: bool = True,
    ) -> None:
        """
        Initialize the chunker with explicit parameters.

        Args:
            tokenizer_name: Name of the tokenizer to use (default: sentence-transformers/all-MiniLM-L6-v2)
            chunk_max_tokens: Maximum tokens per chunk (default: 512)
            merge_peers: Whether to merge peer sections in chunking (default: True)
        """
        if tokenizer_name is None:
            tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.tokenizer_name = tokenizer_name
        self.chunk_max_tokens = chunk_max_tokens
        self.merge_peers = merge_peers

        # Initialize tokenizer (library API uses max_tokens)
        if tokenizer_name == "tiktoken":
            try:
                import tiktoken

                tt_tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenizer: Union[HuggingFaceTokenizer, OpenAITokenizer] = OpenAITokenizer(
                    tokenizer=tt_tokenizer,
                    max_tokens=chunk_max_tokens,
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
                    max_tokens=chunk_max_tokens,
                )
        else:
            # HuggingFace tokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=chunk_max_tokens,
            )

        # Initialize HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=merge_peers,
        )

        rich_print(
            f"[blue][DocumentChunker][/blue] Initialized with:\n"
            f"  • Tokenizer: [cyan]{tokenizer_name}[/cyan]\n"
            f"  • Chunk Max Tokens: [yellow]{chunk_max_tokens}[/yellow]\n"
            f"  • Merge Peers: {merge_peers}"
        )

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
        if self.tokenizer.count_tokens(text) <= self.chunk_max_tokens:
            return [text]

        # Split on sentence boundaries and newlines
        segments = [seg for seg in re.split(r"(?<=[.!?])\s+|\n\n|\n", text) if seg]
        chunks: list[str] = []
        current_segments: list[str] = []

        for segment in segments:
            candidate_segments = [*current_segments, segment]
            candidate_text = " ".join(candidate_segments).strip()
            if not candidate_text:
                continue
            candidate_tokens = self.tokenizer.count_tokens(candidate_text)

            if candidate_tokens <= self.chunk_max_tokens or not current_segments:
                current_segments = candidate_segments
                continue

            chunks.append(" ".join(current_segments).strip())
            current_segments = [segment]

        if current_segments:
            chunks.append(" ".join(current_segments).strip())

        return chunks

    def get_config_summary(self) -> dict:
        """
        Get current chunker configuration as dictionary.

        Returns:
            Dictionary with tokenizer_name, chunk_max_tokens, merge_peers, tokenizer_class
        """
        return {
            "tokenizer_name": self.tokenizer_name,
            "chunk_max_tokens": self.chunk_max_tokens,
            "merge_peers": self.merge_peers,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }
