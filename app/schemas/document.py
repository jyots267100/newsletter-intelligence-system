"""Schemas for document upload, metadata, and management."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Persistent record for an ingested document. Stored in the document
    registry (see `app/repositories/document_repository.py`)."""

    document_id: str
    filename: str
    content_hash: str
    page_count: int = 0
    chunk_count: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str | None = None
    uploaded_at: datetime
    file_size_bytes: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: ProcessingStatus
    page_count: int
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str
