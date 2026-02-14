"""
Extraction contracts package.

This package provides extraction contracts:
- direct: best-effort full-document extraction in a single LLM call.
- staged: multi-pass focused extraction with deterministic reconciliation.

Import from: `from .contracts import direct, staged, delta`
"""

from . import delta, direct, staged

__all__ = [
    "delta",
    "direct",
    "staged",
]
