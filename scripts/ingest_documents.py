#!/usr/bin/env python
"""CLI to ingest one or more PDFs directly into the vector/BM25 indexes,
without going through the API. Useful for bulk-loading a sample corpus.

Usage:
    python scripts/ingest_documents.py data/sample/*.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import NewsletterIntelligenceError  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.retrieval.keyword_retriever import KeywordRetriever  # noqa: E402
from app.retrieval.vector_retriever import VectorRetriever  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402

logger = get_logger(__name__)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1

    settings = get_settings()
    configure_logging(settings.log_level)

    service = IngestionService(
        document_repository=DocumentRepository(settings.document_registry_path),
        vector_retriever=VectorRetriever(settings.chroma_persist_directory, settings.embedding_model),
        keyword_retriever=KeywordRetriever(settings.bm25_index_path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )

    exit_code = 0
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"[SKIP] {path}: file not found")
            exit_code = 1
            continue
        try:
            metadata = service.ingest(path.name, path.read_bytes())
            print(f"[OK] {path.name}: document_id={metadata.document_id} "
                  f"pages={metadata.page_count} chunks={metadata.chunk_count}")
        except NewsletterIntelligenceError as exc:
            print(f"[FAIL] {path.name}: {exc}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
