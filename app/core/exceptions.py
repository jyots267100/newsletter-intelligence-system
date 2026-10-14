"""Application-specific exceptions.

Using typed exceptions (instead of raising generic `Exception` or
`HTTPException` deep in service code) keeps the service layer free of
HTTP concerns - routes translate these into HTTP responses in one place.
"""
from __future__ import annotations


class NewsletterIntelligenceError(Exception):
    """Base class for all application-raised errors."""


class InvalidFileError(NewsletterIntelligenceError):
    """Raised when an uploaded file fails validation (type, size, emptiness)."""


class PDFExtractionError(NewsletterIntelligenceError):
    """Raised when text cannot be extracted from a PDF."""


class DuplicateDocumentError(NewsletterIntelligenceError):
    """Raised when a document with the same content hash already exists."""

    def __init__(self, message: str, existing_document_id: str):
        super().__init__(message)
        self.existing_document_id = existing_document_id


class DocumentNotFoundError(NewsletterIntelligenceError):
    """Raised when a referenced document_id does not exist in the registry."""


class RetrievalError(NewsletterIntelligenceError):
    """Raised when vector/keyword retrieval fails unexpectedly."""


class LLMError(NewsletterIntelligenceError):
    """Base class for LLM-related failures."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM is misconfigured (e.g. missing API key)."""


class LLMTimeoutError(LLMError):
    """Raised when a call to the LLM exceeds the configured timeout."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider reports a rate limit or quota error."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM provider is temporarily unavailable (5xx)."""


class LLMResponseValidationError(LLMError):
    """Raised when the LLM's output cannot be parsed/validated into the
    expected structured schema."""
