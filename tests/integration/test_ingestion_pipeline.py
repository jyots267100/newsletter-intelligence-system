"""End-to-end ingestion + retrieval integration tests using the REAL
ChromaDB + sentence-transformers + BM25 stack (no mocking of the
retrieval layer). No Gemini calls are made here - pure retrieval.

These download/run a real embedding model on first use, so they are
slower than the unit tests but still make zero network calls to any paid
API.
"""
from pathlib import Path

import pytest

from app.repositories.document_repository import DocumentRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.ingestion_service import IngestionService
from tests.conftest import make_pdf_bytes

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pipeline")
    repo = DocumentRepository(str(tmp_path / "docs.json"))
    vector = VectorRetriever(str(tmp_path / "chroma"), EMBEDDING_MODEL)
    keyword = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    ingestion = IngestionService(
        repo, vector, keyword, chunk_size=500, chunk_overlap=50, max_upload_size_bytes=20 * 1024 * 1024
    )
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)
    return repo, vector, keyword, ingestion, hybrid


def test_full_ingestion_indexes_into_both_stores(pipeline, sample_pdf_bytes):
    repo, vector, keyword, ingestion, hybrid = pipeline
    metadata = ingestion.ingest("newsletter.pdf", sample_pdf_bytes)

    assert metadata.status.value == "ready"
    assert vector.count(metadata.document_id) == metadata.chunk_count
    assert keyword.count(metadata.document_id) == metadata.chunk_count


def test_hybrid_retrieval_finds_correct_page_for_targeted_question(pipeline, sample_pdf_bytes):
    repo, vector, keyword, ingestion, hybrid = pipeline
    documents = repo.list_all()
    document_id = documents[0].document_id

    results = hybrid.retrieve("When does DataSync Pro launch?", top_k=3, document_id=document_id)
    assert results
    assert any(r.page_number == 1 for r in results)
    assert all(r.document_id == document_id for r in results)
    assert all(r.chunk_id and r.filename == "newsletter.pdf" for r in results)


def test_hybrid_retrieval_finds_revenue_statistic(pipeline):
    repo, vector, keyword, ingestion, hybrid = pipeline
    document_id = repo.list_all()[0].document_id

    results = hybrid.retrieve("How much did revenue grow?", top_k=3, document_id=document_id)
    assert any(r.page_number == 2 for r in results)


def test_deletion_removes_document_from_both_indexes(pipeline):
    repo, vector, keyword, ingestion, hybrid = pipeline
    distinct_pdf_bytes = make_pdf_bytes(["Completely unrelated content about OtherCorp widgets."])
    metadata = ingestion.ingest("second.pdf", distinct_pdf_bytes)

    ingestion.delete(metadata.document_id)

    assert repo.get(metadata.document_id) is None
    assert vector.count(metadata.document_id) == 0
    assert keyword.count(metadata.document_id) == 0


def test_retrieval_is_scoped_and_does_not_mix_documents(pipeline):
    repo, vector, keyword, ingestion, hybrid = pipeline
    document_id = repo.list_all()[0].document_id

    results = hybrid.retrieve("DataSync Pro", top_k=10, document_id=document_id)
    assert all(r.document_id == document_id for r in results)
