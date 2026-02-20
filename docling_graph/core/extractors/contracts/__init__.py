"""
Extraction contracts package.

This package provides extraction contracts:
- direct: best-effort full-document extraction in a single LLM call.
- staged: multi-pass focused extraction with deterministic reconciliation.
- delta: chunk-based graph IR extraction with merge and projection.
- dense: two-phase skeleton-then-fill extraction (autonomous).

Import from: `from .contracts import direct, staged, delta, dense`
"""

from . import delta, dense, direct, staged

__all__ = [
    "delta",
    "dense",
    "direct",
    "staged",
]
