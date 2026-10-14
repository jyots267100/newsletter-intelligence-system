from datetime import datetime, timezone

import pytest

from app.core.exceptions import DocumentNotFoundError
from app.repositories.document_repository import DocumentRepository
from app.schemas.chat import ChatRole, ChatTurn
from app.schemas.common import RetrievedChunk
from app.schemas.document import DocumentMetadata, ProcessingStatus
from app.services.chat_service import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    ChatService,
    rewrite_query_for_retrieval,
)

from tests.conftest import FakeLLMClient


class _StubHybridRetriever:
    def __init__(self, chunks_by_doc: dict[str, list[RetrievedChunk]]):
        self._chunks_by_doc = chunks_by_doc
        self.last_query = None
        self.last_document_id = None

    def retrieve(self, query, top_k, document_id=None):
        self.last_query = query
        self.last_document_id = document_id
        return self._chunks_by_doc.get(document_id, [])[:top_k]


def _chunk(n, page, document_id="doc-1"):
    return RetrievedChunk(
        chunk_id=f"c{n}", document_id=document_id, filename="n.pdf", page_number=page, text=f"excerpt {n}"
    )


@pytest.fixture
def repo(tmp_path):
    r = DocumentRepository(str(tmp_path / "docs.json"))
    r.save(
        DocumentMetadata(
            document_id="doc-1", filename="n.pdf", content_hash="h", status=ProcessingStatus.READY,
            uploaded_at=datetime.now(timezone.utc), file_size_bytes=1, page_count=2, chunk_count=2,
        )
    )
    return r


def test_chat_returns_grounded_answer_with_citations(repo):
    retriever = _StubHybridRetriever({"doc-1": [_chunk(1, page=1)]})
    llm = FakeLLMClient(response="The answer based on the document.")
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=6)

    response = service.chat("doc-1", "What happened?", [])

    assert response.grounded is True
    assert response.answer == "The answer based on the document."
    assert len(response.citations) == 1
    assert response.citations[0].page_number == 1
    assert response.citations[0].document_id == "doc-1"


def test_chat_with_no_retrieved_chunks_returns_insufficient_evidence(repo):
    retriever = _StubHybridRetriever({"doc-1": []})
    llm = FakeLLMClient(response="should not be used")
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=6)

    response = service.chat("doc-1", "irrelevant question", [])

    assert response.grounded is False
    assert response.citations == []
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert llm.calls == []  # LLM is never called without evidence


def test_chat_llm_declines_to_answer_yields_no_citations(repo):
    retriever = _StubHybridRetriever({"doc-1": [_chunk(1, page=1)]})
    llm = FakeLLMClient(response=INSUFFICIENT_EVIDENCE_MESSAGE)
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=6)

    response = service.chat("doc-1", "unanswerable question", [])
    assert response.grounded is False
    assert response.citations == []


def test_chat_raises_for_unknown_document(repo):
    retriever = _StubHybridRetriever({})
    llm = FakeLLMClient(response="x")
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=6)

    with pytest.raises(DocumentNotFoundError):
        service.chat("missing-doc", "hello", [])


def test_chat_never_leaks_another_documents_chunks(repo):
    retriever = _StubHybridRetriever(
        {"doc-1": [_chunk(1, page=1, document_id="doc-1")]}
    )
    llm = FakeLLMClient(response="answer")
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=6)

    service.chat("doc-1", "question", [])
    assert retriever.last_document_id == "doc-1"


def test_rewrite_query_prepends_last_user_turn_for_pronoun_followup():
    history = [
        ChatTurn(role=ChatRole.USER, content="What was the main announcement?"),
        ChatTurn(role=ChatRole.ASSISTANT, content="DataSync Pro was announced."),
    ]
    rewritten = rewrite_query_for_retrieval("What are its risks?", history)
    assert "What was the main announcement?" in rewritten
    assert "its risks" in rewritten


def test_rewrite_query_leaves_standalone_question_unchanged():
    question = "What is the customer base size mentioned in the newsletter this quarter?"
    assert rewrite_query_for_retrieval(question, []) == question


def test_rewrite_query_with_no_history_returns_original():
    assert rewrite_query_for_retrieval("What about it?", []) == "What about it?"


def test_chat_history_is_bounded(repo):
    retriever = _StubHybridRetriever({"doc-1": [_chunk(1, page=1)]})
    llm = FakeLLMClient(response="answer")
    service = ChatService(repo, retriever, llm, top_k=5, history_turns=2)

    long_history = [
        ChatTurn(role=ChatRole.USER, content=f"question {i}") for i in range(10)
    ]
    service.chat("doc-1", "follow up", long_history)

    system_instruction, prompt = llm.calls[0]
    # Only the last 2 turns should appear in the prompt's history section.
    assert "question 9" in prompt
    assert "question 0" not in prompt
