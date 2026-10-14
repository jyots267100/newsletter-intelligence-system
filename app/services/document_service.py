"""Document management facade used by API routes.

Routes stay thin: they call into this service rather than touching the
repository or retrieval indexes directly.
"""
from __future__ import annotations

from app.core.exceptions import DocumentNotFoundError
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentMetadata
from app.services.ingestion_service import IngestionService


class DocumentService:
    def __init__(self, repository: DocumentRepository, ingestion_service: IngestionService):
        self._repo = repository
        self._ingestion = ingestion_service

    def upload(self, filename: str, file_bytes: bytes) -> DocumentMetadata:
        return self._ingestion.ingest(filename, file_bytes)

    def list_documents(self) -> list[DocumentMetadata]:
        return sorted(self._repo.list_all(), key=lambda d: d.uploaded_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentMetadata:
        document = self._repo.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return document

    def delete_document(self, document_id: str) -> None:
        if not self._repo.get(document_id):
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        self._ingestion.delete(document_id)
