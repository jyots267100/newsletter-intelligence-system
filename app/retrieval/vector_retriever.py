"""Semantic retrieval via ChromaDB + Hugging Face sentence-transformers.

ChromaDB is used in persistent (local, embedded) mode - no separate server
process to run, which keeps the stack simple while still giving real
on-disk persistence across restarts.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from app.core.logging import get_logger
from app.utils.chunking import Chunk

logger = get_logger(__name__)

COLLECTION_NAME = "newsletter_chunks"


class VectorRetriever:
    def __init__(self, persist_directory: str, embedding_model: str):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def query(
        self, query_text: str, top_k: int, document_id: str | None = None
    ) -> list[dict]:
        """Returns ranked results as plain dicts: chunk_id, document_id,
        filename, page_number, text, rank (1-indexed), distance."""
        where = {"document_id": document_id} if document_id else None
        try:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
            )
        except Exception:
            logger.exception("Vector query failed")
            return []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        results = []
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "page_number": meta["page_number"],
                    "text": text,
                    "rank": rank,
                    "distance": distance,
                }
            )
        return results

    def count(self, document_id: str | None = None) -> int:
        if document_id:
            return len(self._collection.get(where={"document_id": document_id})["ids"])
        return self._collection.count()
