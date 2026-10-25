"""Index PDFs into the Supabase knowledge base.

Usage:
    python -m physiobot.ingest "Sources/Tidy's Physiotherapy.pdf" [--source "Tidy's Physiotherapy"]
        [--replace | --resume] [--limit 800]

The Gemini free tier allows about 1,000 embeddings a day, and the chatbot's own
questions share that quota. --resume skips chunks already stored and --limit
caps one run, so a large book is indexed over a few daily runs while the bot
keeps working.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client

from .config import get_settings
from .embeddings import DailyQuotaExceeded, Embedder

log = logging.getLogger("physiobot.ingest")

# ~800 characters keeps a full idea per chunk (the original 300 often cut
# sentences in half); the overlap preserves context across boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    page: int
    index: int
    content: str


def clean(text: str) -> str:
    # Re-join words hyphenated across line breaks, then collapse whitespace.
    text = text.replace("-\n", "")
    return " ".join(text.split())


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    with pymupdf.open(pdf_path) as doc:
        return [(i + 1, clean(page.get_text())) for i, page in enumerate(doc)]


def chunk_pages(pages: list[tuple[int, str]], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            if len(piece) >= 80:  # skip headers, page numbers and other fragments
                chunks.append(Chunk(page=page, index=len(chunks), content=piece))
    return chunks


def pending_chunks(chunks: list[Chunk], done: set[int], limit: int | None = None) -> list[Chunk]:
    todo = [c for c in chunks if c.index not in done]
    return todo if limit is None else todo[:limit]


def stored_indexes(db, source: str) -> set[int]:
    # PostgREST returns at most 1,000 rows per request, so page through.
    done: set[int] = set()
    page = 1000
    while True:
        res = (
            db.table("documents")
            .select("chunk_index")
            .eq("source", source)
            .order("chunk_index")
            .range(len(done), len(done) + page - 1)
            .execute()
        )
        done.update(r["chunk_index"] for r in res.data)
        if len(res.data) < page:
            return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--source", help="Display name for citations (defaults to the file name)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replace", action="store_true", help="Delete existing chunks for this source first")
    mode.add_argument("--resume", action="store_true", help="Skip chunks that are already stored")
    parser.add_argument("--limit", type=int, help="Embed at most this many chunks in this run")
    args = parser.parse_args()

    source = args.source or args.pdf.stem
    s = get_settings()
    db = create_client(s.supabase_url, s.supabase_service_key)
    embedder = Embedder()

    chunks = chunk_pages(extract_pages(args.pdf))
    log.info("%s: %d chunks", source, len(chunks))
    if args.replace:
        db.table("documents").delete().eq("source", source).execute()
    done = stored_indexes(db, source) if args.resume else set()
    todo = pending_chunks(chunks, done, args.limit)
    log.info("  %d already stored, embedding %d now", len(done), len(todo))

    batch = 100
    indexed = len(done)
    for start in range(0, len(todo), batch):
        part = todo[start : start + batch]
        try:
            vectors = embedder.embed_documents([c.content for c in part], title=source)
        except DailyQuotaExceeded:
            log.warning("Daily Gemini quota used up at %d/%d chunks. Run again tomorrow to continue.", indexed, len(chunks))
            return
        rows = [
            {"source": source, "page": c.page, "chunk_index": c.index, "content": c.content, "embedding": v}
            for c, v in zip(part, vectors, strict=True)
        ]
        db.table("documents").upsert(rows, on_conflict="source,chunk_index").execute()
        indexed += len(part)
        log.info("  indexed %d/%d", indexed, len(chunks))
    if indexed < len(chunks):
        log.info("Stopped at the per-run limit. The next run will continue from here.")
    else:
        log.info("All %d chunks are indexed.", len(chunks))


if __name__ == "__main__":
    main()
