"""PDF ingestion pipeline: validate -> extract -> chunk -> embed -> index.

This module is intentionally pure pipeline logic (no FastAPI/Streamlit
concerns) so it is independently testable and reusable from the
`scripts/ingest_documents.py` CLI.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.exceptions import DuplicateDocumentError, InvalidFileError, PDFExtractionError
from app.core.logging import get_logger
from app.repositories.document_repository import DocumentRepository
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.document import DocumentMetadata, ProcessingStatus
from app.utils.chunking import chunk_pages
from app.utils.pdf_loader import compute_content_hash, extract_pages

logger = get_logger(__name__)

ALLOWED_EXTENSION = ".pdf"
PDF_MAGIC_BYTES = b"%PDF-"


def validate_upload(filename: str, file_bytes: bytes, max_size_bytes: int) -> None:
    if not filename.lower().endswith(ALLOWED_EXTENSION):
        raise InvalidFileError(f"Only PDF files are supported, got: {filename}")

    if not file_bytes:
        raise InvalidFileError("Uploaded file is empty.")

    if len(file_bytes) > max_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        raise InvalidFileError(
            f"File size {size_mb:.1f}MB exceeds the {limit_mb:.0f}MB limit."
        )

    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise InvalidFileError("File content does not look like a valid PDF.")


class IngestionService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_size_bytes: int,
    ):
        self._repo = document_repository
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_upload_size_bytes = max_upload_size_bytes

    def ingest(self, filename: str, file_bytes: bytes) -> DocumentMetadata:
        validate_upload(filename, file_bytes, self._max_upload_size_bytes)

        content_hash = compute_content_hash(file_bytes)
        existing = self._repo.get_by_content_hash(content_hash)
        if existing is not None:
            raise DuplicateDocumentError(
                f"An identical document was already uploaded as '{existing.filename}'.",
                existing_document_id=existing.document_id,
            )

        document_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            status=ProcessingStatus.PROCESSING,
            uploaded_at=datetime.now(timezone.utc),
            file_size_bytes=len(file_bytes),
        )
        self._repo.save(metadata)

        try:
            pages = extract_pages(file_bytes)
            chunks = chunk_pages(
                pages,
                document_id=document_id,
                filename=filename,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            if not chunks:
                raise PDFExtractionError(
                    "No extractable text found in this PDF. Scanned/image-only "
                    "PDFs are not supported (OCR is not implemented)."
                )

            self._vector.add_chunks(chunks)
            self._keyword.add_chunks(chunks)

            metadata.page_count = len(pages)
            metadata.chunk_count = len(chunks)
            metadata.status = ProcessingStatus.READY
            self._repo.save(metadata)
            logger.info(
                "Ingested document %s (%s): %d pages, %d chunks",
                document_id, filename, len(pages), len(chunks),
            )
        except Exception as exc:
            metadata.status = ProcessingStatus.FAILED
            metadata.error_message = str(exc)
            self._repo.save(metadata)
            logger.exception("Ingestion failed for %s", filename)
            raise

        return metadata

    def delete(self, document_id: str) -> bool:
        deleted = self._repo.delete(document_id)
        if deleted:
            self._vector.delete_document(document_id)
            self._keyword.delete_document(document_id)
        return deleted
