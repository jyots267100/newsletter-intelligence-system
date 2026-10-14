"""Dependency-injection wiring for FastAPI routes.

Singletons (vector store, BM25 index, document registry, LLM client) are
built once per process via `lru_cache`-wrapped factories and shared across
requests. This keeps route handlers thin - they only declare `Depends(...)`
and call into a service.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.repositories.document_repository import DocumentRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.services.analysis_service import AnalysisService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService


@lru_cache
def get_document_repository(settings: Settings = Depends(get_settings)) -> DocumentRepository:
    return DocumentRepository(settings.document_registry_path)


@lru_cache
def get_vector_retriever(settings: Settings = Depends(get_settings)) -> VectorRetriever:
    return VectorRetriever(settings.chroma_persist_directory, settings.embedding_model)


@lru_cache
def get_keyword_retriever(settings: Settings = Depends(get_settings)) -> KeywordRetriever:
    return KeywordRetriever(settings.bm25_index_path)


@lru_cache
def get_hybrid_retriever(
    vector: VectorRetriever = Depends(get_vector_retriever),
    keyword: KeywordRetriever = Depends(get_keyword_retriever),
    settings: Settings = Depends(get_settings),
) -> HybridRetriever:
    return HybridRetriever(vector, keyword, settings.retrieval_candidate_k, settings.rrf_k)


@lru_cache
def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_retries=settings.llm_max_retries,
        timeout_seconds=settings.llm_timeout_seconds,
    )


@lru_cache
def get_ingestion_service(
    repo: DocumentRepository = Depends(get_document_repository),
    vector: VectorRetriever = Depends(get_vector_retriever),
    keyword: KeywordRetriever = Depends(get_keyword_retriever),
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    return IngestionService(
        repo, vector, keyword,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )


@lru_cache
def get_document_service(
    repo: DocumentRepository = Depends(get_document_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> DocumentService:
    return DocumentService(repo, ingestion)


@lru_cache
def get_analysis_service(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> AnalysisService:
    return AnalysisService(retriever, llm, settings.analysis_top_k)


@lru_cache
def get_chat_service(
    repo: DocumentRepository = Depends(get_document_repository),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    return ChatService(
        repo, retriever, llm,
        top_k=settings.retrieval_top_k,
        history_turns=settings.chat_history_turns,
    )
