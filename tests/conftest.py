"""Shared pytest fixtures.

Storage-location environment variables are set at module import time
(before any `app.*` module is imported by test collection) so that
`get_settings()` never points at the real project `chroma_data/` directory
during tests, and a placeholder Gemini API key is set so that constructing
a GeminiClient doesn't fail for tests that don't care about it. No test in
this suite makes a real network call to Gemini - the LLM is always mocked.
"""
from __future__ import annotations

import io
import os
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="newsletter_intel_tests_")
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", os.path.join(_TMP_ROOT, "chroma"))
os.environ.setdefault("BM25_INDEX_PATH", os.path.join(_TMP_ROOT, "bm25_index.pkl"))
os.environ.setdefault("DOCUMENT_REGISTRY_PATH", os.path.join(_TMP_ROOT, "documents.json"))
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-key")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from app.llm.base import LLMClient  # noqa: E402


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Builds a simple multi-page PDF in memory, one string of text per page."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    for page_text in pages:
        c.setFont("Helvetica", 12)
        y = 750
        for line in page_text.split("\n"):
            c.drawString(72, y, line)
            y -= 18
        c.showPage()
    c.save()
    return buffer.getvalue()


class FakeLLMClient(LLMClient):
    """A scriptable LLMClient test double. Set `.response` to a fixed
    string, or `.responses` to a list consumed in order, or `.side_effect`
    to an exception/callable to simulate failures."""

    def __init__(self, response: str = ""):
        self.response = response
        self.responses: list[str] | None = None
        self.side_effect: Exception | None = None
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        self.calls.append((system_instruction, user_prompt))
        if self.side_effect is not None:
            raise self.side_effect
        if self.responses is not None:
            return self.responses.pop(0)
        return self.response


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return make_pdf_bytes(
        [
            "TechPulse Monthly\nTechPulse Inc. announces the launch of DataSync Pro "
            "on November 1, 2026. This is a major product announcement.",
            "Quarterly Highlights\nRevenue grew 23% year-over-year to $4.2 million in "
            "Q3 2026. The customer base expanded to 1,200 organizations.",
            "Risks\nA key risk is increasing competition from open-source alternatives, "
            "which could pressure subscription pricing.",
            "Action Items\nThe engineering team must complete the security audit by "
            "November 15, 2026. Maria Chen owns the CloudNet reseller agreement.",
        ]
    )


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    c.save()
    return buffer.getvalue()
