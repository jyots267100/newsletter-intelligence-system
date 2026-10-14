from app.utils.chunking import chunk_pages


def test_chunk_pages_tags_each_chunk_with_correct_page_and_ids():
    pages = [(1, "Alpha content here."), (2, "Beta content here.")]
    chunks = chunk_pages(pages, document_id="doc-1", filename="n.pdf", chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.filename == "n.pdf" for c in chunks)
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert all(c.chunk_id.startswith("doc-1_p") for c in chunks)


def test_chunk_pages_skips_empty_pages():
    pages = [(1, "Real content."), (2, "   "), (3, "")]
    chunks = chunk_pages(pages, document_id="doc-1", filename="n.pdf", chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1


def test_chunk_pages_deduplicates_identical_text_on_same_page():
    repeated = ("Header footer text. " * 5).strip()
    # chunk_size is small enough that the splitter breaks on the "\n\n"
    # separator into two pieces, both equal to `repeated` - the dedup
    # logic should drop the second occurrence.
    pages = [(1, f"{repeated}\n\n{repeated}")]
    chunks = chunk_pages(
        pages, document_id="doc-1", filename="n.pdf",
        chunk_size=len(repeated) + 10, chunk_overlap=0,
    )
    texts = [c.text for c in chunks]
    assert texts == [repeated]
    assert len(texts) == len(set(texts))


def test_chunk_pages_splits_long_pages_into_multiple_chunks():
    long_text = "Sentence number %d provides unique content. " * 1
    long_page = " ".join(f"Sentence number {i} provides unique content." for i in range(200))
    pages = [(1, long_page)]
    chunks = chunk_pages(pages, document_id="doc-1", filename="n.pdf", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    # chunk_index should be unique and increasing
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)


def test_chunk_pages_empty_input_returns_empty_list():
    assert chunk_pages([], document_id="doc-1", filename="n.pdf", chunk_size=1000, chunk_overlap=100) == []
