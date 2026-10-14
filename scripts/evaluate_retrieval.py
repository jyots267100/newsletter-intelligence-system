#!/usr/bin/env python
"""Retrieval evaluation harness.

Measures Recall@k and Mean Reciprocal Rank (MRR) for the hybrid retriever
against a small hand-labeled question set (data/sample/eval_dataset.json),
where each question has a known expected source page. This exercises only
the retrieval layer (ChromaDB + BM25 + RRF) and makes NO Gemini API calls,
so it is free to run repeatedly.

With `--with-llm`, it additionally runs the real chat pipeline (Gemini
included) and checks whether the expected answer facts appear in the
generated answer and whether the returned citation page matches the
expected page. This is an approximate "groundedness" proxy, not a
rigorous NLI-based evaluation, and consumes Gemini API quota.

Usage:
    python scripts/generate_sample_newsletter.py   # once, if not already run
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --with-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import DuplicateDocumentError  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from app.retrieval.keyword_retriever import KeywordRetriever  # noqa: E402
from app.retrieval.vector_retriever import VectorRetriever  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "eval_dataset.json"
SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_newsletter.pdf"


def _ensure_sample_indexed(repo, ingestion, filename: str) -> str:
    for doc in repo.list_all():
        if doc.filename == filename and doc.status.value == "ready":
            return doc.document_id

    if not SAMPLE_PDF_PATH.exists():
        raise SystemExit(
            f"{SAMPLE_PDF_PATH} not found. Run "
            "`python scripts/generate_sample_newsletter.py` first."
        )
    try:
        metadata = ingestion.ingest(filename, SAMPLE_PDF_PATH.read_bytes())
    except DuplicateDocumentError as exc:
        metadata = repo.get(exc.existing_document_id)
    return metadata.document_id


def run(with_llm: bool) -> dict:
    settings = get_settings()
    dataset = json.loads(DATASET_PATH.read_text())

    repo = DocumentRepository(settings.document_registry_path)
    vector = VectorRetriever(settings.chroma_persist_directory, settings.embedding_model)
    keyword = KeywordRetriever(settings.bm25_index_path)
    hybrid = HybridRetriever(vector, keyword, settings.retrieval_candidate_k, settings.rrf_k)
    ingestion = IngestionService(
        repo, vector, keyword,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_size_bytes=settings.max_upload_size_bytes,
    )

    document_id = _ensure_sample_indexed(repo, ingestion, dataset["document_filename"])

    chat_service = None
    if with_llm:
        from app.llm.gemini import GeminiClient

        llm = GeminiClient(
            settings.gemini_api_key, settings.gemini_model,
            settings.llm_max_retries, settings.llm_timeout_seconds,
        )
        chat_service = ChatService(repo, hybrid, llm, settings.retrieval_top_k, settings.chat_history_turns)

    top_k = settings.retrieval_top_k
    hits = 0
    reciprocal_ranks = []
    llm_fact_hits = 0
    llm_citation_hits = 0
    per_question_results = []

    for item in dataset["questions"]:
        chunks = hybrid.retrieve(item["question"], top_k=top_k, document_id=document_id)
        pages_retrieved = [c.page_number for c in chunks]
        hit = item["expected_page"] in pages_retrieved
        hits += int(hit)
        rank = (
            pages_retrieved.index(item["expected_page"]) + 1
            if hit else None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        result = {
            "question": item["question"],
            "expected_page": item["expected_page"],
            "pages_retrieved": pages_retrieved,
            "retrieval_hit": hit,
        }

        if with_llm and chat_service is not None:
            response = chat_service.chat(document_id, item["question"], [])
            answer_lower = response.answer.lower()
            facts_found = [f for f in item["expected_answer_facts"] if f.lower() in answer_lower]
            fact_hit = len(facts_found) > 0
            llm_fact_hits += int(fact_hit)
            citation_pages = [c.page_number for c in response.citations]
            citation_hit = item["expected_page"] in citation_pages
            llm_citation_hits += int(citation_hit)
            result.update(
                {
                    "answer": response.answer,
                    "facts_found": facts_found,
                    "citation_pages": citation_pages,
                    "citation_hit": citation_hit,
                }
            )

        per_question_results.append(result)

    n = len(dataset["questions"])
    report = {
        "document_id": document_id,
        "num_questions": n,
        "retrieval_recall_at_k": hits / n,
        "retrieval_mrr": sum(reciprocal_ranks) / n,
        "top_k": top_k,
        "questions": per_question_results,
    }
    if with_llm:
        report["llm_answer_fact_hit_rate"] = llm_fact_hits / n
        report["llm_citation_correctness"] = llm_citation_hits / n

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="Also evaluate end-to-end chat answers via Gemini.")
    args = parser.parse_args()

    report = run(with_llm=args.with_llm)

    print(json.dumps(report, indent=2))
    print("\n--- Summary ---")
    print(f"Questions evaluated:        {report['num_questions']}")
    print(f"Retrieval Recall@{report['top_k']}:         {report['retrieval_recall_at_k']:.2%}")
    print(f"Retrieval MRR:              {report['retrieval_mrr']:.3f}")
    if args.with_llm:
        print(f"LLM answer fact hit rate:   {report['llm_answer_fact_hit_rate']:.2%}")
        print(f"LLM citation correctness:   {report['llm_citation_correctness']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
