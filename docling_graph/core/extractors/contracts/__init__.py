"""
Extraction contracts package.

This package provides extraction contracts:
- direct: best-effort full-document extraction in a single LLM call.
- dense: two-phase skeleton-then-fill extraction (autonomous).

Import from: `from .contracts import direct, dense`
"""

from . import dense, direct

__all__ = [
    "dense",
    "direct",
]
