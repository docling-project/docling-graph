"""Auto extraction-contract resolution.

"auto" is not a third execution path: it is a per-document decision between
the two real contracts, made by the strategy once the document's actual size
is known (after Docling conversion). It exists so users don't have to guess —
running the direct contract on a document that dwarfs the model's context
window or output budget fails after minutes of doomed calls.

The decision is deterministic and uses the same arithmetic the backend uses
to detect doomed direct calls:

- direct needs the whole document plus the response inside one context
  window, and the response itself must be able to represent the document's
  content within the output-token budget;
- dense (skeleton-then-fill over chunks) has no such coupling to document
  size, at the cost of more LLM calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Rough chars-per-token for sizing decisions. Deliberately conservative
# (real prose runs ~4.5-5 chars/token), so estimates overshoot: a document
# flagged as too large for direct is either truly infeasible or close enough
# to the limit that a single-call extraction would be uselessly lossy.
CHARS_PER_TOKEN = 4
# Direct is viable while the input stays within this multiple of the output
# character capacity; beyond it a single response cannot represent the
# document without silently self-rationing. 1.0 is deliberate (R=1.0): with a
# verbatim-heavy template the output scales with the document, and at ratio
# 2.0 a 62k-char contract slipped through direct and lost 57% of its verbatim
# clause texts with no truncation warning. When output pressure is detected,
# route to dense early.
DIRECT_OVERFLOW_RATIO = 1.0


@dataclass(frozen=True)
class ContractDecision:
    """Resolved contract plus the numbers that drove the choice."""

    contract: Literal["direct", "dense"]
    reason: str
    estimated_input_tokens: int
    output_budget_tokens: int
    context_limit_tokens: int | None

    def describe(self) -> str:
        """One-line human-readable summary for logs."""
        context_part = (
            f"context limit {self.context_limit_tokens} tokens"
            if self.context_limit_tokens
            else "context limit unknown"
        )
        return (
            f"contract={self.contract} ({self.reason}; "
            f"~{self.estimated_input_tokens} estimated input tokens, "
            f"output budget {self.output_budget_tokens} tokens, {context_part})"
        )


def resolve_auto_contract(
    *,
    markdown_chars: int,
    output_budget_tokens: int,
    context_limit_tokens: int | None,
    chunking_available: bool,
) -> ContractDecision:
    """Pick direct or dense for one document.

    ``markdown_chars`` must be the document's CONTENT character count — for
    DocLang serializations, strip the markup first (see
    ``doclang_format.content_chars``). The same document must resolve to the
    same contract regardless of the chosen LLM serialization; markup overhead
    is not information the response has to represent.

    Direct is chosen only when BOTH hold:
    1. the estimated input plus the output budget fits the context window
       (always true when the window is unknown — the guard cannot lie), and
    2. the document is small enough that a single response can plausibly
       represent it (input within DIRECT_OVERFLOW_RATIO x the output
       character capacity).

    Dense is chosen otherwise — unless chunking is disabled, in which case
    dense cannot run and direct is used regardless (the backend still refuses
    calls that arithmetically cannot fit).
    """
    estimated_input_tokens = max(1, markdown_chars // CHARS_PER_TOKEN)

    if not chunking_available:
        return ContractDecision(
            contract="direct",
            reason="chunking disabled, dense unavailable",
            estimated_input_tokens=estimated_input_tokens,
            output_budget_tokens=output_budget_tokens,
            context_limit_tokens=context_limit_tokens,
        )

    fits_context = (
        context_limit_tokens is None
        or estimated_input_tokens + output_budget_tokens <= context_limit_tokens
    )
    output_char_capacity = output_budget_tokens * CHARS_PER_TOKEN
    fits_output_budget = markdown_chars <= output_char_capacity * DIRECT_OVERFLOW_RATIO

    if fits_context and fits_output_budget:
        return ContractDecision(
            contract="direct",
            reason="document fits a single call",
            estimated_input_tokens=estimated_input_tokens,
            output_budget_tokens=output_budget_tokens,
            context_limit_tokens=context_limit_tokens,
        )

    if not fits_context:
        reason = "input would exceed the model's context window"
    else:
        reason = (
            f"document ({markdown_chars} chars) exceeds what a single "
            f"~{output_budget_tokens}-token response can represent"
        )
    return ContractDecision(
        contract="dense",
        reason=reason,
        estimated_input_tokens=estimated_input_tokens,
        output_budget_tokens=output_budget_tokens,
        context_limit_tokens=context_limit_tokens,
    )
