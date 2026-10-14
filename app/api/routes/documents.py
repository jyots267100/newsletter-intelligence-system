"""Document upload, listing, retrieval, deletion, and analysis endpoints.

Routes are intentionally thin: they read input, call a service, and return
its result. Exception-to-HTTP-status mapping (InvalidFileError -> 400,
DocumentNotFoundError -> 404, LLM failures -> 429/502/503/504, ...) is
centralized in app/main.py's app-level exception handlers rather than
repeated here - see the comment there for why that matters beyond
tidiness (it also covers errors raised during dependency injection, which
a route-local try/except would never see).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_analysis_service, get_document_service
from app.schemas.analysis import AnalysisRequest, AnalysisResult
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.analysis_service import AnalysisService
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    file_bytes = await file.read()
    document = service.upload(file.filename or "upload.pdf", file_bytes)
    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        message="Document processed and indexed successfully.",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentMetadata)
def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentMetadata:
    return service.get_document(document_id)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDeleteResponse:
    service.delete_document(document_id)
    return DocumentDeleteResponse(
        document_id=document_id, deleted=True, message="Document deleted."
    )


@router.post("/{document_id}/analyze", response_model=AnalysisResult)
def analyze_document(
    document_id: str,
    request: AnalysisRequest = AnalysisRequest(),
    document_service: DocumentService = Depends(get_document_service),
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResult:
    document = document_service.get_document(document_id)
    return analysis_service.analyze(document, top_k=request.top_k)
