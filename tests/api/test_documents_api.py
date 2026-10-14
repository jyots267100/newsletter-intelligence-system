from datetime import datetime, timezone

from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidFileError,
    LLMRateLimitError,
    LLMResponseValidationError,
)
from app.schemas.analysis import AnalysisResult, ExecutiveSummary, ExtractedEntities
from app.schemas.document import DocumentMetadata, ProcessingStatus


def _metadata(document_id="doc-1") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id, filename="n.pdf", content_hash="h",
        status=ProcessingStatus.READY, uploaded_at=datetime.now(timezone.utc),
        file_size_bytes=100, page_count=2, chunk_count=4,
    )


class _FakeDocumentService:
    def __init__(self):
        self.uploaded = None
        self.raise_on_upload = None
        self.documents = {"doc-1": _metadata()}
        self.raise_on_get = None
        self.raise_on_delete = None

    def upload(self, filename, file_bytes):
        if self.raise_on_upload:
            raise self.raise_on_upload
        self.uploaded = (filename, file_bytes)
        return _metadata()

    def list_documents(self):
        return list(self.documents.values())

    def get_document(self, document_id):
        if self.raise_on_get:
            raise self.raise_on_get
        if document_id not in self.documents:
            raise DocumentNotFoundError(document_id)
        return self.documents[document_id]

    def delete_document(self, document_id):
        if self.raise_on_delete:
            raise self.raise_on_delete
        self.documents.pop(document_id, None)


class _FakeAnalysisService:
    def __init__(self):
        self.raise_error = None

    def analyze(self, document, top_k=None):
        if self.raise_error:
            raise self.raise_error
        return AnalysisResult(
            document_id=document.document_id,
            filename=document.filename,
            executive_summary=ExecutiveSummary(summary="Summary text.", citations=[]),
            key_insights=[],
            action_items=[],
            entities=ExtractedEntities(),
        )


def test_upload_document_success(client, override_document_service):
    fake = _FakeDocumentService()
    override_document_service(fake)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("newsletter.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["status"] == "ready"


def test_upload_document_invalid_file_returns_400(client, override_document_service):
    fake = _FakeDocumentService()
    fake.raise_on_upload = InvalidFileError("Only PDF files are supported")
    override_document_service(fake)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("newsletter.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_document_duplicate_returns_409(client, override_document_service):
    fake = _FakeDocumentService()
    fake.raise_on_upload = DuplicateDocumentError("Duplicate", existing_document_id="doc-1")
    override_document_service(fake)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("newsletter.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 409
    assert "doc-1" in response.json()["detail"]


def test_list_documents(client, override_document_service):
    override_document_service(_FakeDocumentService())
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_get_document_not_found_returns_404(client, override_document_service):
    override_document_service(_FakeDocumentService())
    response = client.get("/api/v1/documents/missing-id")
    assert response.status_code == 404


def test_get_document_found(client, override_document_service):
    override_document_service(_FakeDocumentService())
    response = client.get("/api/v1/documents/doc-1")
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"


def test_delete_document_success(client, override_document_service):
    fake = _FakeDocumentService()
    override_document_service(fake)
    response = client.delete("/api/v1/documents/doc-1")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_delete_document_not_found(client, override_document_service):
    fake = _FakeDocumentService()
    fake.raise_on_delete = DocumentNotFoundError("missing")
    override_document_service(fake)
    response = client.delete("/api/v1/documents/missing-id")
    assert response.status_code == 404


def test_analyze_document_success(client, override_document_service, override_analysis_service):
    override_document_service(_FakeDocumentService())
    override_analysis_service(_FakeAnalysisService())

    response = client.post("/api/v1/documents/doc-1/analyze")
    assert response.status_code == 200
    assert response.json()["executive_summary"]["summary"] == "Summary text."


def test_analyze_document_not_found(client, override_document_service, override_analysis_service):
    override_document_service(_FakeDocumentService())
    override_analysis_service(_FakeAnalysisService())

    response = client.post("/api/v1/documents/missing-id/analyze")
    assert response.status_code == 404


def test_analyze_document_rate_limited_returns_429(client, override_document_service, override_analysis_service):
    override_document_service(_FakeDocumentService())
    fake_analysis = _FakeAnalysisService()
    fake_analysis.raise_error = LLMRateLimitError("quota exceeded")
    override_analysis_service(fake_analysis)

    response = client.post("/api/v1/documents/doc-1/analyze")
    assert response.status_code == 429


def test_analyze_document_malformed_llm_output_returns_502(client, override_document_service, override_analysis_service):
    override_document_service(_FakeDocumentService())
    fake_analysis = _FakeAnalysisService()
    fake_analysis.raise_error = LLMResponseValidationError("bad json")
    override_analysis_service(fake_analysis)

    response = client.post("/api/v1/documents/doc-1/analyze")
    assert response.status_code == 502
