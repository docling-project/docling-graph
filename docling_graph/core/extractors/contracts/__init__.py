"""
Extraction contracts package.

This package provides the direct extraction mode: best-effort full-document
extraction in a single LLM call.

Import from: `from .contracts import direct`
"""

from . import direct

__all__ = [
    "direct",
]
