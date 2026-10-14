"""Newsletter analysis: executive summary, key insights, actionable items,
and extracted entities - generated via the hybrid RAG pipeline + Gemini.

Citation handling: the LLM is asked to reference excerpt NUMBERS (not page
numbers) per item. This service resolves those numbers back to the actual
retrieved chunk metadata (document_id/filename/page_number/chunk_id) -
the LLM never gets to state a page number directly, which is what
prevents hallucinated citations.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from app.core.exceptions import LLMResponseValidationError
from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.llm.prompts import ANALYSIS_SYSTEM_INSTRUCTION, build_analysis_prompt
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.analysis import (
    ActionItem,
    AnalysisResult,
    ExecutiveSummary,
    ExtractedEntities,
    InsightCategory,
    KeyInsight,
    SourceType,
)
from app.schemas.common import Citation, RetrievedChunk
from app.schemas.document import DocumentMetadata

logger = get_logger(__name__)

# A broad, fixed retrieval query: analysis needs whole-document coverage
# rather than a narrow user question, so we query for the kinds of content
# an executive summary/insight/action extraction needs.
_ANALYSIS_QUERY = (
    "summary overview key findings insights statistics trends announcements "
    "recommended actions deadlines important dates organizations"
)


class _RawExecutiveSummary(BaseModel):
    text: str
    excerpt_refs: list[int] = Field(default_factory=list)


class _RawKeyInsight(BaseModel):
    text: str
    category: InsightCategory
    source_type: SourceType
    excerpt_refs: list[int] = Field(default_factory=list)


class _RawActionItem(BaseModel):
    action: str
    deadline: str | None = None
    owner: str | None = None
    source_type: SourceType = SourceType.EXPLICIT
    excerpt_refs: list[int] = Field(default_factory=list)


class _RawAnalysisResponse(BaseModel):
    executive_summary: _RawExecutiveSummary
    key_insights: list[_RawKeyInsight] = Field(default_factory=list)
    action_items: list[_RawActionItem] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)


class AnalysisService:
    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient, analysis_top_k: int):
        self._retriever = hybrid_retriever
        self._llm = llm_client
        self._top_k = analysis_top_k

    def analyze(self, document: DocumentMetadata, top_k: int | None = None) -> AnalysisResult:
        chunks = self._retriever.retrieve(
            _ANALYSIS_QUERY, top_k=top_k or self._top_k, document_id=document.document_id
        )
        if not chunks:
            raise LLMResponseValidationError(
                "No indexed content found for this document; cannot analyze."
            )

        prompt = build_analysis_prompt(chunks, document.filename)
        raw_text = self._llm.generate(ANALYSIS_SYSTEM_INSTRUCTION, prompt)
        raw = self._parse_response(raw_text)

        def resolve(refs: list[int]) -> list[Citation]:
            return self._resolve_citations(refs, chunks)

        return AnalysisResult(
            document_id=document.document_id,
            filename=document.filename,
            executive_summary=ExecutiveSummary(
                summary=raw.executive_summary.text,
                citations=resolve(raw.executive_summary.excerpt_refs),
            ),
            key_insights=[
                KeyInsight(
                    text=i.text,
                    category=i.category,
                    source_type=i.source_type,
                    citations=resolve(i.excerpt_refs),
                )
                for i in raw.key_insights
            ],
            action_items=[
                ActionItem(
                    action=a.action,
                    deadline=a.deadline,
                    owner=a.owner,
                    source_type=a.source_type,
                    citations=resolve(a.excerpt_refs),
                )
                for a in raw.action_items
            ],
            entities=raw.entities,
        )

    @staticmethod
    def _resolve_citations(refs: list[int], chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for ref in refs:
            index = ref - 1  # excerpt numbers are 1-indexed in the prompt
            if index < 0 or index >= len(chunks):
                continue  # discard out-of-range/hallucinated refs
            chunk = chunks[index]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                )
            )
        return citations

    @staticmethod
    def _parse_response(raw_text: str) -> _RawAnalysisResponse:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Gemini returned malformed JSON: %s", raw_text[:500])
            raise LLMResponseValidationError(
                f"Model response was not valid JSON: {exc}"
            ) from exc

        try:
            return _RawAnalysisResponse.model_validate(payload)
        except ValidationError as exc:
            logger.error("Gemini response failed schema validation: %s", exc)
            raise LLMResponseValidationError(
                f"Model response did not match the expected schema: {exc}"
            ) from exc
