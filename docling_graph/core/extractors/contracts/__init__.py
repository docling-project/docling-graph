"""
Extraction contracts package.

This package provides extraction contracts:
- direct: best-effort full-document extraction in a single LLM call.
- staged: multi-pass focused extraction with deterministic reconciliation.

Import from: `from .contracts import direct, staged`
"""

from . import direct
from . import staged

__all__ = [
    "direct",
    "staged",
]
