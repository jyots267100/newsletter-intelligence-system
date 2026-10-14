"""FastAPI application entrypoint."""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidFileError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
    PDFExtractionError,
)
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Newsletter Intelligence System",
    description=(
        "Upload a newsletter PDF to get an executive summary, key insights, "
        "actionable items, and a grounded chatbot over its content."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code if response is not None else 500
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )


def _error_handler(status_code: int):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"error": type(exc).__name__, "detail": str(exc)})
    return handler


# Exception-to-HTTP-status mapping is centralized here, as app-level
# handlers, rather than duplicated in try/except blocks in every route.
# This matters beyond tidiness: dependency factories (e.g. constructing
# GeminiClient in app/api/dependencies.py) can raise these same typed
# exceptions during FastAPI's dependency-injection resolution, which runs
# BEFORE a route handler's own try/except ever gets a chance to run - a
# route-local try/except alone would never see that error. App-level
# handlers catch it regardless of where in the request lifecycle it's
# raised.
app.add_exception_handler(InvalidFileError, _error_handler(400))
app.add_exception_handler(DocumentNotFoundError, _error_handler(404))
app.add_exception_handler(PDFExtractionError, _error_handler(422))
app.add_exception_handler(LLMConfigurationError, _error_handler(503))
app.add_exception_handler(LLMRateLimitError, _error_handler(429))
app.add_exception_handler(LLMTimeoutError, _error_handler(504))
app.add_exception_handler(LLMUnavailableError, _error_handler(502))
app.add_exception_handler(LLMResponseValidationError, _error_handler(502))


@app.exception_handler(DuplicateDocumentError)
async def duplicate_document_handler(request: Request, exc: DuplicateDocumentError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "DuplicateDocumentError",
            "detail": f"{exc} (existing document_id={exc.existing_document_id})",
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
