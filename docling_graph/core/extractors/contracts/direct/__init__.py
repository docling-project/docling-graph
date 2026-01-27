"""
Direct extraction contracts package.

This package contains prompts and utilities for direct (full-document) extraction mode.
Direct mode performs best-effort extraction in a single LLM call without chunking or
map-reduce orchestration.
"""

from .prompts import PromptDict, get_extraction_prompt

__all__ = [
    "PromptDict",
    "get_extraction_prompt",
]
