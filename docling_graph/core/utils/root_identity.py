"""Root-identity repair: class-name-echo clearing and source-stem fallback.

The root node is the anchor of every extracted graph: all root-level edges key
on its identity, so a root whose identity fields are empty (the model found no
document reference) or echo the template class name (a schema echo like
``reference_document="AssuranceMRH"``) silently breaks every root-anchored
edge downstream — in graph dedup across batches and in any ground-truth
comparison.

Two deterministic, root-only repairs:

1. **Class-name-echo clearing**: an identity value that canonicalizes to the
   template class name is a schema echo, never document data (a document whose
   real reference equals its extraction template's Python class name does not
   occur in practice). Cleared so a later repair or fill can supply the real
   value instead of locking the echo in.
2. **Source-stem fallback**: when every root identity field is empty after
   extraction (and echo clearing), the first identity field is set to the
   source document's stem (e.g. ``insurance_terms``). The root is a singleton,
   so a synthetic identity is safe *for the root only* — it restores a stable,
   human-readable anchor at zero model cost. Non-root nodes must never receive
   synthetic identities (that is the merge rescue ladder's gated job).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_ALNUM = re.compile(r"[^a-z0-9]+")

# A stem longer than this is not a usable identifier (defensive: DoclingDocument
# names normally derive from file stems).
_MAX_STEM_CHARS = 80


def _canonical(text: str) -> str:
    """Casefold and strip to [a-z0-9] for order-insensitive echo comparison."""
    return _ALNUM.sub("", text.casefold())


def is_class_name_echo(value: Any, class_name: str) -> bool:
    """True when a string identity value is just the template class name echoed back."""
    if not isinstance(value, str) or not value.strip():
        return False
    canon = _canonical(value)
    return bool(canon) and canon == _canonical(class_name)


def _root_id_fields(model: BaseModel) -> list[str]:
    config = getattr(model, "model_config", None) or {}
    raw = config.get("graph_id_fields", []) if hasattr(config, "get") else []
    return [f for f in raw if isinstance(f, str) and f in type(model).model_fields]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def repair_root_identity(
    model: BaseModel,
    *,
    document_stem: str | None = None,
) -> BaseModel:
    """Clear class-name echoes on the root's identity fields, then apply the stem fallback.

    Mutates ``model`` in place (assignment is not validated, matching pipeline
    salvage behavior) and returns it. No-op for templates without
    ``graph_id_fields``. The fallback only fires when EVERY identity field is
    empty and a usable ``document_stem`` is available; a partially-filled
    identity is document data and is never touched.
    """
    id_fields = _root_id_fields(model)
    if not id_fields:
        return model
    class_name = type(model).__name__

    try:
        for field in id_fields:
            if is_class_name_echo(getattr(model, field, None), class_name):
                object.__setattr__(model, field, "")
                logger.info(
                    "Root identity: cleared %s.%s (echoed the template class name)",
                    class_name,
                    field,
                )

        if all(_is_empty(getattr(model, field, None)) for field in id_fields):
            stem = (document_stem or "").strip()
            if stem and len(stem) <= _MAX_STEM_CHARS:
                object.__setattr__(model, id_fields[0], stem)
                logger.info(
                    "Root identity: %s.%s was empty after extraction; "
                    "falling back to source stem %r so root-anchored edges stay matchable",
                    class_name,
                    id_fields[0],
                    stem,
                )
    except Exception as e:  # repair must never break extraction
        logger.warning("Root identity repair skipped: %s", e)
    return model
