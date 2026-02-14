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
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Large tokenizer max length used only for counting/splitting operations.
# This avoids HF warnings when we inspect oversized text before re-splitting it.
_TOKENIZER_COUNTING_MAX_LENGTH = 1_000_000


def _raise_tokenizer_max_length(
    hf_tokenizer: PreTrainedTokenizerBase, chunk_max_tokens: int
) -> None:
    """Raise the tokenizer's model_max_length so encoding long text for counting doesn't warn.

    We use the tokenizer only for token counting and chunk splitting, not for the model.
    HybridChunker can produce chunks slightly over chunk_max_tokens; encoding them would
    otherwise trigger: "Token indices sequence length is longer than the specified
    maximum sequence length (e.g. 685 > 512)".
    """
    current = getattr(hf_tokenizer, "model_max_length", None)
    if not isinstance(current, int):
        return
    new_max = max(chunk_max_tokens, _TOKENIZER_COUNTING_MAX_LENGTH)
    if current < new_max:
        hf_tokenizer.model_max_length = new_max


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
                _raise_tokenizer_max_length(hf_tokenizer, chunk_max_tokens)
                self.tokenizer = HuggingFaceTokenizer(
                    tokenizer=hf_tokenizer,
                    max_tokens=chunk_max_tokens,
                )
        else:
            # HuggingFace tokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            _raise_tokenizer_max_length(hf_tokenizer, chunk_max_tokens)
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

        # Chunk the document using HybridChunker (structure-based; may exceed chunk_max_tokens)
        chunk_iter = self.chunker.chunk(dl_doc=document)

        for chunk in chunk_iter:
            enriched_text = self.chunker.contextualize(chunk=chunk)
            token_count = self.tokenizer.count_tokens(enriched_text)
            if token_count <= self.chunk_max_tokens:
                chunks.append(enriched_text)
            else:
                # Hard cap: re-split any oversized chunk so we never exceed chunk_max_tokens
                chunks.extend(self.chunk_text_fallback(enriched_text))

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
            token_count = self.tokenizer.count_tokens(enriched_text)
            if token_count <= self.chunk_max_tokens:
                chunks.append(enriched_text)
                chunk_tokens.append(token_count)
            else:
                # Re-split oversized chunk and count tokens for each sub-chunk
                sub_chunks = self.chunk_text_fallback(enriched_text)
                chunks.extend(sub_chunks)
                for sub in sub_chunks:
                    chunk_tokens.append(self.tokenizer.count_tokens(sub))

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

        # Split on sentence boundaries and newlines first.
        segments = [
            seg.strip() for seg in re.split(r"(?<=[.!?])\s+|\n\n|\n", text) if seg and seg.strip()
        ]
        if not segments:
            return self._split_oversized_by_chars(text)
        chunks: list[str] = []
        current_segments: list[str] = []

        for segment in segments:
            segment_tokens = self.tokenizer.count_tokens(segment)
            if segment_tokens > self.chunk_max_tokens:
                if current_segments:
                    chunks.append(" ".join(current_segments).strip())
                    current_segments = []
                chunks.extend(self._split_oversized_segment(segment))
                continue

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

        # Final safety pass: never return chunks over the hard token cap.
        safe_chunks: list[str] = []
        for chunk in chunks:
            if self.tokenizer.count_tokens(chunk) <= self.chunk_max_tokens:
                safe_chunks.append(chunk)
            else:
                safe_chunks.extend(self._split_oversized_segment(chunk))

        return safe_chunks

    def _split_oversized_segment(self, segment: str) -> list[str]:
        """Split one oversized segment into <= chunk_max_tokens chunks."""
        segment = segment.strip()
        if not segment:
            return []
        if self.tokenizer.count_tokens(segment) <= self.chunk_max_tokens:
            return [segment]

        words = segment.split()
        if len(words) <= 1:
            return self._split_oversized_by_chars(segment)

        result: list[str] = []
        current_words: list[str] = []

        for word in words:
            if not current_words:
                if self.tokenizer.count_tokens(word) <= self.chunk_max_tokens:
                    current_words = [word]
                else:
                    result.extend(self._split_oversized_by_chars(word))
                continue

            candidate = " ".join([*current_words, word]).strip()
            if self.tokenizer.count_tokens(candidate) <= self.chunk_max_tokens:
                current_words.append(word)
                continue

            result.append(" ".join(current_words).strip())
            if self.tokenizer.count_tokens(word) <= self.chunk_max_tokens:
                current_words = [word]
            else:
                result.extend(self._split_oversized_by_chars(word))
                current_words = []

        if current_words:
            result.append(" ".join(current_words).strip())

        return result

    def _split_oversized_by_chars(self, text: str) -> list[str]:
        """Character-level fallback splitter when token/word boundaries are insufficient."""
        remaining = (text or "").strip()
        if not remaining:
            return []

        parts: list[str] = []
        while remaining:
            if self.tokenizer.count_tokens(remaining) <= self.chunk_max_tokens:
                parts.append(remaining)
                break

            lo = 1
            hi = len(remaining)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = remaining[:mid].strip()
                if not candidate:
                    lo = mid + 1
                    continue
                if self.tokenizer.count_tokens(candidate) <= self.chunk_max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            piece = remaining[:best].strip()
            if not piece:
                piece = remaining[:1]
            parts.append(piece)
            remaining = remaining[len(piece) :].strip()

        return parts

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
