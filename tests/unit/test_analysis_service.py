import json

import pytest

from app.core.exceptions import LLMResponseValidationError
from app.schemas.common import RetrievedChunk
from app.schemas.document import DocumentMetadata, ProcessingStatus
from app.services.analysis_service import AnalysisService
from datetime import datetime, timezone

from tests.conftest import FakeLLMClient


class _StubHybridRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_k, document_id=None):
        return self._chunks[:top_k]


def _chunk(n, page):
    return RetrievedChunk(
        chunk_id=f"c{n}", document_id="doc-1", filename="n.pdf", page_number=page, text=f"excerpt {n}"
    )


def _document():
    return DocumentMetadata(
        document_id="doc-1", filename="n.pdf", content_hash="h", status=ProcessingStatus.READY,
        uploaded_at=datetime.now(timezone.utc), file_size_bytes=100, page_count=3, chunk_count=3,
    )


VALID_RESPONSE = {
    "executive_summary": {"text": "A concise summary.", "excerpt_refs": [1, 2]},
    "key_insights": [
        {"text": "Revenue grew.", "category": "statistic", "source_type": "explicit", "excerpt_refs": [1]}
    ],
    "action_items": [
        {"action": "Complete audit.", "deadline": "Nov 15", "owner": None, "source_type": "explicit", "excerpt_refs": [2]}
    ],
    "entities": {"dates": ["Nov 15"], "organizations": [], "products": [], "people": [], "numerical_facts": []},
}


def test_analyze_resolves_citations_from_retrieved_chunks():
    chunks = [_chunk(1, page=1), _chunk(2, page=2)]
    llm = FakeLLMClient(response=json.dumps(VALID_RESPONSE))
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    result = service.analyze(_document())

    assert result.executive_summary.summary == "A concise summary."
    assert {c.page_number for c in result.executive_summary.citations} == {1, 2}
    assert result.key_insights[0].citations[0].page_number == 1
    assert result.action_items[0].citations[0].page_number == 2
    assert result.action_items[0].owner is None


def test_analyze_discards_out_of_range_excerpt_refs():
    chunks = [_chunk(1, page=1)]
    response = json.loads(json.dumps(VALID_RESPONSE))
    response["key_insights"][0]["excerpt_refs"] = [1, 99]  # 99 is hallucinated/out of range
    llm = FakeLLMClient(response=json.dumps(response))
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    result = service.analyze(_document())
    assert len(result.key_insights[0].citations) == 1
    assert result.key_insights[0].citations[0].page_number == 1


def test_analyze_raises_on_malformed_json():
    chunks = [_chunk(1, page=1)]
    llm = FakeLLMClient(response="this is not json")
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    with pytest.raises(LLMResponseValidationError):
        service.analyze(_document())


def test_analyze_raises_on_schema_mismatch():
    chunks = [_chunk(1, page=1)]
    llm = FakeLLMClient(response=json.dumps({"executive_summary": "missing required shape"}))
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    with pytest.raises(LLMResponseValidationError):
        service.analyze(_document())


def test_analyze_strips_markdown_code_fences():
    chunks = [_chunk(1, page=1)]
    fenced = "```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
    llm = FakeLLMClient(response=fenced)
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    result = service.analyze(_document())
    assert result.executive_summary.summary == "A concise summary."


def test_analyze_raises_when_no_chunks_indexed():
    llm = FakeLLMClient(response=json.dumps(VALID_RESPONSE))
    service = AnalysisService(_StubHybridRetriever([]), llm, analysis_top_k=10)

    with pytest.raises(LLMResponseValidationError):
        service.analyze(_document())


def test_action_item_never_invents_placeholder_owner():
    chunks = [_chunk(1, page=1)]
    response = json.loads(json.dumps(VALID_RESPONSE))
    response["action_items"][0]["owner"] = "Unknown"
    llm = FakeLLMClient(response=json.dumps(response))
    service = AnalysisService(_StubHybridRetriever(chunks), llm, analysis_top_k=10)

    result = service.analyze(_document())
    assert result.action_items[0].owner is None
