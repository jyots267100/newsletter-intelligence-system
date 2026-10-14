from datetime import datetime, timezone
from pathlib import Path

from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentMetadata, ProcessingStatus


def _metadata(document_id="doc-1", content_hash="hash-1") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        filename="n.pdf",
        content_hash=content_hash,
        status=ProcessingStatus.READY,
        uploaded_at=datetime.now(timezone.utc),
        file_size_bytes=1234,
        page_count=4,
        chunk_count=10,
    )


def test_save_and_get(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    repo.save(_metadata())
    fetched = repo.get("doc-1")
    assert fetched is not None
    assert fetched.filename == "n.pdf"
    assert fetched.chunk_count == 10


def test_get_nonexistent_returns_none(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    assert repo.get("missing") is None


def test_get_by_content_hash(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    repo.save(_metadata(document_id="doc-1", content_hash="abc"))
    found = repo.get_by_content_hash("abc")
    assert found is not None
    assert found.document_id == "doc-1"
    assert repo.get_by_content_hash("zzz") is None


def test_list_all(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    repo.save(_metadata(document_id="doc-1", content_hash="h1"))
    repo.save(_metadata(document_id="doc-2", content_hash="h2"))
    assert {d.document_id for d in repo.list_all()} == {"doc-1", "doc-2"}


def test_delete(tmp_path: Path):
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    repo.save(_metadata())
    assert repo.delete("doc-1") is True
    assert repo.get("doc-1") is None
    assert repo.delete("doc-1") is False


def test_persists_across_instances(tmp_path: Path):
    path = str(tmp_path / "docs.json")
    repo = DocumentRepository(path)
    repo.save(_metadata())

    reloaded = DocumentRepository(path)
    assert reloaded.get("doc-1") is not None
