"""Hybrid retrieval: vector search + BM25, combined via Reciprocal Rank
Fusion, scoped optionally to a single document.

This is the ONE retrieval path used by both the analysis pipeline and the
chatbot (see `app/services/analysis_service.py` and `chat_service.py`) -
there is intentionally no separate "chat retrieval" implementation.
"""
from __future__ import annotations

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.common import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        candidate_k: int,
        rrf_k: int,
    ):
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        vector_results = self._vector.query(query, self._candidate_k, document_id)
        keyword_results = self._keyword.query(query, self._candidate_k, document_id)

        vector_by_id = {r["chunk_id"]: r for r in vector_results}
        keyword_by_id = {r["chunk_id"]: r for r in keyword_results}

        fused = reciprocal_rank_fusion(
            [
                [r["chunk_id"] for r in vector_results],
                [r["chunk_id"] for r in keyword_results],
            ],
            k=self._rrf_k,
        )

        chunks: list[RetrievedChunk] = []
        for chunk_id, fused_score in fused[:top_k]:
            source = vector_by_id.get(chunk_id) or keyword_by_id.get(chunk_id)
            if source is None:
                continue  # defensive; should not happen
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=source["document_id"],
                    filename=source["filename"],
                    page_number=source["page_number"],
                    text=source["text"],
                    score=fused_score,
                    vector_rank=vector_by_id.get(chunk_id, {}).get("rank"),
                    bm25_rank=keyword_by_id.get(chunk_id, {}).get("rank"),
                )
            )
        return chunks
