"""Shared schema primitives used across documents, analysis, and chat."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A source citation derived from RETRIEVED CHUNK METADATA, never from
    model-generated text. This is the core anti-hallucination mechanism:
    the API layer builds citations from what was actually retrieved, and
    the LLM is only asked to indicate *which* retrieved chunks it used.
    """

    document_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    chunk_id: str


class RetrievedChunk(BaseModel):
    """A single chunk returned by the hybrid retriever, with full
    provenance metadata preserved end-to-end."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(..., ge=1)
    text: str
    score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
