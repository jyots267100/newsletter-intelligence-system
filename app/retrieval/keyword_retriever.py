"""Keyword retrieval via BM25 (rank_bm25).

Semantic (embedding) search misses exact-term matches that matter a lot
for newsletters - product names, acronyms, specific figures. BM25 covers
that gap. The BM25 index is rebuilt in-memory from a persisted list of
chunk records on startup/mutation; this avoids pickling the BM25Okapi
object itself (which is brittle across rank_bm25 versions) while still
surviving restarts.
"""
from __future__ import annotations

import pickle
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.utils.chunking import Chunk

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Record:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    tokens: list[str] = field(default_factory=list)


class KeywordRetriever:
    def __init__(self, index_path: str):
        self._path = Path(index_path)
        self._lock = threading.Lock()
        self._records: list[_Record] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open("rb") as f:
                    self._records = pickle.load(f)
                self._rebuild()
            except Exception:
                logger.exception("Failed to load BM25 index; starting empty")
                self._records = []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump(self._records, f)

    def _rebuild(self) -> None:
        if self._records:
            self._bm25 = BM25Okapi([r.tokens for r in self._records])
        else:
            self._bm25 = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            for c in chunks:
                self._records.append(
                    _Record(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        filename=c.filename,
                        page_number=c.page_number,
                        text=c.text,
                        tokens=_tokenize(c.text),
                    )
                )
            self._rebuild()
            self._persist()

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._records = [r for r in self._records if r.document_id != document_id]
            self._rebuild()
            self._persist()

    def query(
        self, query_text: str, top_k: int, document_id: str | None = None
    ) -> list[dict]:
        with self._lock:
            records = self._records
            bm25 = self._bm25

        if not records or bm25 is None:
            return []

        candidate_indices = [
            i for i, r in enumerate(records)
            if document_id is None or r.document_id == document_id
        ]
        if not candidate_indices:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)

        # Filter to documents that actually share at least one query term,
        # rather than filtering on score > 0: BM25's classic IDF formula
        # is exactly zero for a term that appears in precisely half of the
        # corpus (log((N-n+0.5)/(n+0.5)) == 0 when n == N/2). With small
        # corpora - a short newsletter might only have a few dozen chunks -
        # that's a real, reachable case, not just a theoretical one, and a
        # score-based filter would silently drop a genuine keyword match.
        matching_indices = [
            i for i in candidate_indices if query_token_set & set(records[i].tokens)
        ]
        if not matching_indices:
            return []

        scores = bm25.get_scores(query_tokens)
        scored = sorted(
            ((scores[i], i) for i in matching_indices),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for rank, (score, idx) in enumerate(scored, start=1):
            r = records[idx]
            results.append(
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "filename": r.filename,
                    "page_number": r.page_number,
                    "text": r.text,
                    "rank": rank,
                    "score": float(score),
                }
            )
        return results

    def count(self, document_id: str | None = None) -> int:
        with self._lock:
            if document_id:
                return sum(1 for r in self._records if r.document_id == document_id)
            return len(self._records)
