"""Schemas for the conversational chatbot."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import Citation


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatTurn(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    document_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    # Client-maintained history keeps the server stateless across requests.
    # Only the last N turns (see settings.chat_history_turns) are used, and
    # only for *conversational* context (e.g. resolving "it"/"its") - never
    # as evidence about the document itself.
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    document_id: str
    grounded: bool
