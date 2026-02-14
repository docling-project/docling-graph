"""Delta contract strategy-level operations."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel


def extract_delta_from_text(
    *,
    backend: Any,
    doc_processor: Any,
    text: str,
    template: type[BaseModel],
) -> tuple[BaseModel | None, float]:
    """Run delta extraction for raw text by chunking first."""
    chunks, chunk_metadata = doc_processor.chunk_text(text)
    start_time = time.time()
    model = backend.extract_from_chunk_batches(
        chunks=chunks,
        chunk_metadata=chunk_metadata,
        template=template,
        context="text input",
    )
    return model, time.time() - start_time


def extract_delta_from_document(
    *,
    backend: Any,
    doc_processor: Any,
    document: Any,
    template: type[BaseModel],
    trace_data: Any = None,
) -> tuple[BaseModel | None, float]:
    """Run delta extraction for converted document using chunk metadata."""
    chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(document)
    if trace_data is not None:
        for cmeta, chunk_text in zip(chunk_metadata, chunks, strict=False):
            trace_data.emit(
                "chunk_created",
                "extraction",
                {
                    "chunk_id": cmeta.get("chunk_id"),
                    "token_count": cmeta.get("token_count"),
                    "page_numbers": cmeta.get("page_numbers"),
                    "text_content": chunk_text,
                },
            )
    start_time = time.time()
    model = backend.extract_from_chunk_batches(
        chunks=chunks,
        chunk_metadata=chunk_metadata,
        template=template,
        context="full document",
    )
    return model, time.time() - start_time
