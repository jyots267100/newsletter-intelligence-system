"""Ingestion service unit tests use stub vector/keyword stores (recording
add_chunks/delete_document calls) so these tests don't depend on
downloading an embedding model or running real ChromaDB - that coverage
lives in tests/integration/test_ingestion_pipeline.py.
"""
from pathlib import Path

import pytest

from app.core.exceptions import DuplicateDocumentError, InvalidFileError, PDFExtractionError
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import ProcessingStatus
from app.services.ingestion_service import IngestionService, validate_upload


class _StubIndex:
    def __init__(self):
        self.added: list = []
        self.deleted: list[str] = []

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def delete_document(self, document_id):
        self.deleted.append(document_id)


@pytest.fixture
def service(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    vector = _StubIndex()
    keyword = _StubIndex()
    svc = IngestionService(
        repo, vector, keyword, chunk_size=1000, chunk_overlap=100, max_upload_size_bytes=20 * 1024 * 1024
    )
    return svc, repo, vector, keyword


def test_validate_upload_rejects_non_pdf_extension():
    with pytest.raises(InvalidFileError):
        validate_upload("file.txt", b"%PDF-1.4 ...", max_size_bytes=10_000_000)


def test_validate_upload_rejects_empty_file():
    with pytest.raises(InvalidFileError):
        validate_upload("file.pdf", b"", max_size_bytes=10_000_000)


def test_validate_upload_rejects_oversized_file():
    with pytest.raises(InvalidFileError):
        validate_upload("file.pdf", b"%PDF-" + b"x" * 100, max_size_bytes=50)


def test_validate_upload_rejects_non_pdf_content():
    with pytest.raises(InvalidFileError):
        validate_upload("file.pdf", b"not actually a pdf", max_size_bytes=10_000_000)


def test_validate_upload_accepts_valid_pdf_header():
    validate_upload("file.pdf", b"%PDF-1.4 rest of file", max_size_bytes=10_000_000)


def test_ingest_success_indexes_chunks_and_saves_metadata(service, sample_pdf_bytes):
    svc, repo, vector, keyword = service
    metadata = svc.ingest("newsletter.pdf", sample_pdf_bytes)

    assert metadata.status == ProcessingStatus.READY
    assert metadata.page_count == 4
    assert metadata.chunk_count > 0
    assert len(vector.added) == metadata.chunk_count
    assert len(keyword.added) == metadata.chunk_count
    assert repo.get(metadata.document_id) is not None


def test_ingest_duplicate_raises_and_does_not_reindex(service, sample_pdf_bytes):
    svc, repo, vector, keyword = service
    first = svc.ingest("newsletter.pdf", sample_pdf_bytes)
    vector.added.clear()
    keyword.added.clear()

    with pytest.raises(DuplicateDocumentError) as exc_info:
        svc.ingest("newsletter-copy.pdf", sample_pdf_bytes)

    assert exc_info.value.existing_document_id == first.document_id
    assert vector.added == []
    assert keyword.added == []


def test_ingest_empty_pdf_marks_failed_and_raises(service, empty_pdf_bytes):
    svc, repo, vector, keyword = service
    with pytest.raises(PDFExtractionError):
        svc.ingest("empty.pdf", empty_pdf_bytes)

    documents = repo.list_all()
    assert len(documents) == 1
    assert documents[0].status == ProcessingStatus.FAILED


def test_delete_removes_from_registry_and_indexes(service, sample_pdf_bytes):
    svc, repo, vector, keyword = service
    metadata = svc.ingest("newsletter.pdf", sample_pdf_bytes)

    assert svc.delete(metadata.document_id) is True
    assert repo.get(metadata.document_id) is None
    assert metadata.document_id in vector.deleted
    assert metadata.document_id in keyword.deleted


def test_delete_nonexistent_returns_false(service):
    svc, *_ = service
    assert svc.delete("does-not-exist") is False


def test_chunks_carry_correct_document_id_and_filename(service, sample_pdf_bytes):
    svc, repo, vector, keyword = service
    metadata = svc.ingest("newsletter.pdf", sample_pdf_bytes)

    assert all(c.document_id == metadata.document_id for c in vector.added)
    assert all(c.filename == "newsletter.pdf" for c in vector.added)
    assert all(1 <= c.page_number <= 4 for c in vector.added)
