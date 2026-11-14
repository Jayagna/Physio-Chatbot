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


def test_quota_day_starts_at_midnight_pacific():
    from datetime import datetime, timezone

    from physiobot.ingest import quota_day_start

    # 06:30 UTC on 27 Oct is still 26 Oct in California (PDT, UTC-7).
    start = quota_day_start(datetime(2026, 10, 27, 6, 30, tzinfo=timezone.utc))
    assert start.astimezone(timezone.utc) == datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)
    start = quota_day_start(datetime(2026, 10, 27, 7, 30, tzinfo=timezone.utc))
    assert start.astimezone(timezone.utc) == datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc)


def test_second_run_on_same_quota_day_only_uses_what_is_left():
    from physiobot.ingest import remaining_budget

    assert remaining_budget(800, 0) == 800
    assert remaining_budget(800, 800) == 0
    assert remaining_budget(800, 950) == 0
    assert remaining_budget(None, 500) is None
