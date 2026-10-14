"""Hybrid retriever tests use lightweight stub vector/keyword retrievers
(same dict-based interface as the real ones) so this layer is tested in
isolation from ChromaDB/BM25/embedding models.
"""
from app.retrieval.hybrid_retriever import HybridRetriever


class _StubRetriever:
    def __init__(self, results: list[dict]):
        self._results = results

    def query(self, query_text, top_k, document_id=None):
        results = self._results
        if document_id:
            results = [r for r in results if r["document_id"] == document_id]
        return results[:top_k]


def _result(chunk_id, document_id, page, rank, text="text", **extra):
    base = {
        "chunk_id": chunk_id, "document_id": document_id, "filename": "n.pdf",
        "page_number": page, "text": text, "rank": rank,
    }
    base.update(extra)
    return base


def test_hybrid_retriever_fuses_and_preserves_metadata():
    vector = _StubRetriever([_result("c1", "doc1", 1, 1), _result("c2", "doc1", 2, 2)])
    keyword = _StubRetriever([_result("c2", "doc1", 2, 1), _result("c3", "doc1", 3, 2)])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)

    results = hybrid.retrieve("query", top_k=5, document_id="doc1")
    ids = [r.chunk_id for r in results]

    assert ids[0] == "c2"  # appears in both lists -> ranks first
    assert set(ids) == {"c1", "c2", "c3"}
    for r in results:
        assert r.document_id == "doc1"
        assert r.filename == "n.pdf"
        assert r.page_number >= 1


def test_hybrid_retriever_deduplicates_chunks_found_by_both_retrievers():
    vector = _StubRetriever([_result("c1", "doc1", 1, 1)])
    keyword = _StubRetriever([_result("c1", "doc1", 1, 1)])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)

    results = hybrid.retrieve("query", top_k=5)
    assert len(results) == 1


def test_hybrid_retriever_scopes_to_document_id():
    vector = _StubRetriever([_result("c1", "doc1", 1, 1), _result("c2", "doc2", 1, 2)])
    keyword = _StubRetriever([])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)

    results = hybrid.retrieve("query", top_k=5, document_id="doc1")
    assert [r.document_id for r in results] == ["doc1"]


def test_hybrid_retriever_respects_top_k():
    vector = _StubRetriever([_result(f"c{i}", "doc1", i, i) for i in range(1, 6)])
    keyword = _StubRetriever([])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)

    results = hybrid.retrieve("query", top_k=2)
    assert len(results) == 2


def test_hybrid_retriever_tracks_vector_and_bm25_ranks():
    vector = _StubRetriever([_result("c1", "doc1", 1, 1)])
    keyword = _StubRetriever([_result("c1", "doc1", 1, 3)])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)

    result = hybrid.retrieve("query", top_k=5)[0]
    assert result.vector_rank == 1
    assert result.bm25_rank == 3


def test_hybrid_retriever_empty_results_from_both():
    vector = _StubRetriever([])
    keyword = _StubRetriever([])
    hybrid = HybridRetriever(vector, keyword, candidate_k=10, rrf_k=60)
    assert hybrid.retrieve("query", top_k=5) == []
