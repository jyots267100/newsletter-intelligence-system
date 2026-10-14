"""Conversational chatbot over a single uploaded PDF.

Uses the SAME HybridRetriever as analysis_service - there is no separate
retrieval implementation for chat. Conversation history is used only to
(a) resolve pronoun-style follow-ups into a better retrieval query and
(b) give the model short-term conversational context; it is never treated
as evidence about the document's content (see prompts.CHAT_SYSTEM_INSTRUCTION).
"""
from __future__ import annotations

import re

from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.llm.prompts import CHAT_SYSTEM_INSTRUCTION, build_chat_prompt
from app.repositories.document_repository import DocumentRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.chat import ChatResponse, ChatRole, ChatTurn
from app.schemas.common import Citation

logger = get_logger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the uploaded document to answer "
    "that confidently."
)

_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|it's|that|this|they|them|those|these|he|she|his|her)\b", re.IGNORECASE
)


def rewrite_query_for_retrieval(message: str, history: list[ChatTurn]) -> str:
    """Deterministic, LLM-free follow-up rewriting: if the current message
    looks like it depends on prior context (short, or contains a pronoun)
    and there is prior user history, prepend the last user turn so the
    retriever has enough signal. This keeps follow-up handling fast,
    free, and independently testable without mocking the LLM.
    """
    is_short = len(message.split()) <= 6
    has_pronoun = bool(_PRONOUN_PATTERN.search(message))

    if not (is_short or has_pronoun):
        return message

    last_user_turns = [t.content for t in history if t.role == ChatRole.USER]
    if not last_user_turns:
        return message

    return f"{last_user_turns[-1]} {message}"


def format_history(history: list[ChatTurn], max_turns: int) -> str:
    bounded = history[-max_turns:] if max_turns > 0 else []
    lines = [f"{turn.role.value.capitalize()}: {turn.content}" for turn in bounded]
    return "\n".join(lines)


class ChatService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        hybrid_retriever: HybridRetriever,
        llm_client: LLMClient,
        top_k: int,
        history_turns: int,
    ):
        self._repo = document_repository
        self._retriever = hybrid_retriever
        self._llm = llm_client
        self._top_k = top_k
        self._history_turns = history_turns

    def chat(
        self, document_id: str, message: str, history: list[ChatTurn]
    ) -> ChatResponse:
        document = self._repo.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        bounded_history = history[-self._history_turns:] if self._history_turns > 0 else []
        retrieval_query = rewrite_query_for_retrieval(message, bounded_history)

        # Retrieval is always scoped to this one document_id, so a question
        # about document A can never pull context from document B.
        chunks = self._retriever.retrieve(
            retrieval_query, top_k=self._top_k, document_id=document_id
        )

        if not chunks:
            return ChatResponse(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                document_id=document_id,
                grounded=False,
            )

        history_text = format_history(bounded_history, self._history_turns)
        prompt = build_chat_prompt(chunks, message, history_text)
        answer = self._llm.generate(CHAT_SYSTEM_INSTRUCTION, prompt).strip()

        is_grounded = INSUFFICIENT_EVIDENCE_MESSAGE not in answer
        citations = (
            [
                Citation(
                    document_id=c.document_id,
                    filename=c.filename,
                    page_number=c.page_number,
                    chunk_id=c.chunk_id,
                )
                for c in chunks
            ]
            if is_grounded
            else []
        )

        return ChatResponse(
            answer=answer,
            citations=citations,
            document_id=document_id,
            grounded=is_grounded,
        )
