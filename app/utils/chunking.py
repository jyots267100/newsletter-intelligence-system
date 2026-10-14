"""Page-aware chunking.

Chunking is done PER PAGE (not on the concatenated document) so that every
chunk can be tagged with exactly one page number. This is what makes
accurate page citations possible later - if we chunked across page
boundaries, a chunk could span two pages and citations would become
ambiguous or wrong.

Chunk size/overlap defaults (1000 chars / 150 overlap) are documented in
`.env.example` and `docs/architecture.md`: newsletters are short, prose-
heavy documents, so word-boundary-aware character chunking performs well
without needing token-level splitting.

The recursive-separator splitting algorithm below (try "\\n\\n", then
"\\n", then ". ", then " ", falling back to a hard character slice, then
greedily re-merge up to chunk_size with a chunk_overlap-sized tail carried
into the next chunk) is a small, self-contained reimplementation of the
same strategy LangChain's `RecursiveCharacterTextSplitter` uses. It's
written directly here - rather than taken as a dependency - because it's
the only piece of that functionality this project needs, and pulling in
`langchain-text-splitters` drags in `langchain-core` and `langsmith` (and
their own large, fast-moving transitive dependency trees) for one ~30-line
algorithm. Keeping it in-repo also makes it trivial to unit test in
isolation (see tests/unit/test_chunking.py).
"""
from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _split_into_units(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Recursively breaks text into pieces no longer than chunk_size,
    preferring to split on the earliest separator available. Each
    separator (other than the final hard-slice fallback) is kept attached
    to the piece that precedes it, so re-joining all pieces reconstructs
    the original text exactly."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, remaining = separators[0], separators[1:]
    raw_parts = text.split(sep)
    parts = [p + sep for p in raw_parts[:-1]] + [raw_parts[-1]]

    units: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= chunk_size:
            units.append(part)
        else:
            units.extend(_split_into_units(part, remaining, chunk_size))
    return units


def _merge_units(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merges small units back into chunks up to chunk_size,
    carrying a chunk_overlap-sized tail of the previous chunk forward so
    sentences near a boundary don't lose all surrounding context."""
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = current + unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())

        if len(unit) <= chunk_size:
            tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = tail + unit
        else:
            current = unit  # already hard-sliced upstream to fit on its own

    if current.strip():
        chunks.append(current.strip())

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    units = _split_into_units(text, _DEFAULT_SEPARATORS, chunk_size)
    return _merge_units(units, chunk_size, chunk_overlap)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[tuple[int, str]],
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_texts: set[str] = set()
    global_index = 0

    for page_number, page_text in pages:
        normalized = page_text.strip()
        if not normalized:
            continue  # avoid empty chunks

        for piece in split_text(normalized, chunk_size, chunk_overlap):
            piece = piece.strip()
            if not piece:
                continue
            # Avoid duplicate chunks (e.g. repeated headers/footers).
            dedup_key = f"{page_number}:{piece}"
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            chunk_id = f"{document_id}_p{page_number}_c{global_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=global_index,
                    text=piece,
                )
            )
            global_index += 1

    return chunks
