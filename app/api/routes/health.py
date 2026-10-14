"""Liveness/readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_document_repository
from app.repositories.document_repository import DocumentRepository

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness probe: process is up."""
    return {"status": "ok"}


@router.get("/ready")
def ready(repo: DocumentRepository = Depends(get_document_repository)) -> dict:
    """Readiness probe: core dependencies (document registry) are reachable."""
    try:
        repo.list_all()
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "not_ready", "detail": str(exc)}
    return {"status": "ready"}
