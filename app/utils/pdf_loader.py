"""PDF text extraction with page tracking.

Uses PyMuPDF because it is fast, has no external system dependencies
(unlike poppler-based tools), and gives reliable per-page text extraction.

Limitation (documented, not silently glossed over): this extracts text
from text-based PDFs only. Scanned/image-only PDFs will yield little or
no text per page because OCR is not implemented. See README "Limitations".
"""
from __future__ import annotations

import hashlib

import pymupdf

from app.core.exceptions import PDFExtractionError


def compute_content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Returns a list of (page_number, text) tuples, 1-indexed pages.

    Pages with no extractable text are still returned (with empty text)
    so page counts stay accurate; empty pages are filtered out later
    during chunking.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pymupdf raises generic RuntimeError/ValueError
        raise PDFExtractionError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise PDFExtractionError("PDF has zero pages.")

    pages: list[tuple[int, str]] = []
    try:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text") or ""
            pages.append((index + 1, text.strip()))
    except Exception as exc:
        raise PDFExtractionError(f"Failed to extract text: {exc}") from exc
    finally:
        doc.close()

    return pages
