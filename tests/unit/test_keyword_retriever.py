from pathlib import Path

from app.retrieval.keyword_retriever import KeywordRetriever
from app.utils.chunking import Chunk


def _chunk(chunk_id, document_id, page, text):
    return Chunk(chunk_id=chunk_id, document_id=document_id, filename="n.pdf", page_number=page, chunk_index=0, text=text)


def test_bm25_query_finds_exact_keyword_match(tmp_path: Path):
    retriever = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    retriever.add_chunks(
        [
            _chunk("c1", "doc1", 1, "DataSync Pro launches in November."),
            _chunk("c2", "doc1", 2, "Revenue grew twenty three percent this quarter."),
        ]
    )

    results = retriever.query("DataSync Pro", top_k=5)
    assert results
    assert results[0]["chunk_id"] == "c1"


def test_bm25_scopes_query_to_document_id(tmp_path: Path):
    retriever = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    retriever.add_chunks(
        [
            _chunk("c1", "doc1", 1, "unique keyword alpha"),
            _chunk("c2", "doc2", 1, "unique keyword alpha"),
        ]
    )

    results = retriever.query("alpha", top_k=5, document_id="doc1")
    assert [r["chunk_id"] for r in results] == ["c1"]


def test_bm25_delete_document_removes_its_chunks(tmp_path: Path):
    retriever = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    retriever.add_chunks([_chunk("c1", "doc1", 1, "keyword gamma content")])
    assert retriever.count() == 1

    retriever.delete_document("doc1")
    assert retriever.count() == 0
    assert retriever.query("gamma", top_k=5) == []


def test_bm25_index_persists_across_instances(tmp_path: Path):
    path = str(tmp_path / "bm25.pkl")
    retriever = KeywordRetriever(path)
    retriever.add_chunks([_chunk("c1", "doc1", 1, "persisted keyword delta")])

    reloaded = KeywordRetriever(path)
    results = reloaded.query("delta", top_k=5)
    assert [r["chunk_id"] for r in results] == ["c1"]


def test_bm25_query_with_no_matches_returns_empty(tmp_path: Path):
    retriever = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    retriever.add_chunks([_chunk("c1", "doc1", 1, "something entirely different")])
    assert retriever.query("zzzznonexistentzzzz", top_k=5) == []


def test_bm25_query_on_empty_index_returns_empty(tmp_path: Path):
    retriever = KeywordRetriever(str(tmp_path / "bm25.pkl"))
    assert retriever.query("anything", top_k=5) == []
