"""FastAPI test fixtures. Every route test overrides the service-level
dependency (DocumentService/AnalysisService/ChatService) with a test
double, so no test in tests/api touches a real embedding model, ChromaDB,
BM25 index, or Gemini.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_analysis_service, get_chat_service, get_document_service
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def override(dependency, fake):
    app.dependency_overrides[dependency] = lambda: fake


@pytest.fixture
def override_document_service():
    def _apply(fake):
        override(get_document_service, fake)
    return _apply


@pytest.fixture
def override_analysis_service():
    def _apply(fake):
        override(get_analysis_service, fake)
    return _apply


@pytest.fixture
def override_chat_service():
    def _apply(fake):
        override(get_chat_service, fake)
    return _apply
