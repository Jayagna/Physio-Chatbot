from physiobot.embeddings import document_text, query_text
from physiobot.ingest import chunk_pages, clean, pending_chunks


def test_clean_rejoins_hyphenated_words():
    assert clean("physio-\ntherapy  is\n\ngreat") == "physiotherapy is great"


def test_chunks_keep_page_numbers_and_skip_fragments():
    pages = [(1, "12"), (2, "word " * 400)]
    chunks = chunk_pages(pages)
    assert chunks and all(c.page == 2 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content) <= 800 for c in chunks)


def test_embedding_prompt_formats():
    assert query_text("knee pain") == "task: question answering | query: knee pain"
    assert document_text("abc", "Tidy") == "title: Tidy | text: abc"
    assert document_text("abc") == "title: none | text: abc"


def test_pending_chunks_skips_stored_and_respects_limit():
    pages = [(1, "word " * 1000)]
    chunks = chunk_pages(pages)
    done = {0, 2}
    todo = pending_chunks(chunks, done, limit=3)
    assert [c.index for c in todo] == [1, 3, 4]
    assert len(pending_chunks(chunks, set())) == len(chunks)
