"""Chatbot endpoint.

See app/api/routes/documents.py's docstring: exception-to-HTTP mapping is
centralized in app/main.py, not repeated here.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return service.chat(request.document_id, request.message, request.history)
