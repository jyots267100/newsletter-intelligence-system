import pytest

from app.core.exceptions import PDFExtractionError
from app.utils.pdf_loader import compute_content_hash, extract_pages


def test_extract_pages_preserves_page_numbers_and_text(sample_pdf_bytes):
    pages = extract_pages(sample_pdf_bytes)
    assert len(pages) == 4
    page_numbers = [p[0] for p in pages]
    assert page_numbers == [1, 2, 3, 4]
    assert "DataSync Pro" in pages[0][1]
    assert "Revenue grew" in pages[1][1]


def test_extract_pages_on_invalid_bytes_raises():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"this is not a pdf")


def test_extract_pages_on_empty_or_near_empty_pdf_has_no_meaningful_text(empty_pdf_bytes):
    # A PDF with no content drawn on it must yield either zero pages
    # (raises PDFExtractionError) or pages with no extractable text -
    # never fabricated text.
    try:
        pages = extract_pages(empty_pdf_bytes)
    except PDFExtractionError:
        return
    assert all(text.strip() == "" for _, text in pages)


def test_compute_content_hash_is_deterministic(sample_pdf_bytes):
    assert compute_content_hash(sample_pdf_bytes) == compute_content_hash(sample_pdf_bytes)


def test_compute_content_hash_differs_for_different_content(sample_pdf_bytes, empty_pdf_bytes):
    assert compute_content_hash(sample_pdf_bytes) != compute_content_hash(empty_pdf_bytes)
