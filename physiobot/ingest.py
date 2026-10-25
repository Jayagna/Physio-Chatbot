"""Index PDFs into the Supabase knowledge base.

Usage:
    python -m physiobot.ingest "Sources/Tidy's Physiotherapy.pdf" [--source "Tidy's Physiotherapy"] [--replace]
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
from .embeddings import Embedder

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--source", help="Display name for citations (defaults to the file name)")
    parser.add_argument("--replace", action="store_true", help="Delete existing chunks for this source first")
    args = parser.parse_args()

    source = args.source or args.pdf.stem
    s = get_settings()
    db = create_client(s.supabase_url, s.supabase_service_key)
    embedder = Embedder()

    chunks = chunk_pages(extract_pages(args.pdf))
    log.info("%s: %d chunks", source, len(chunks))
    if args.replace:
        db.table("documents").delete().eq("source", source).execute()

    batch = 100
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        vectors = embedder.embed_documents([c.content for c in part], title=source)
        rows = [
            {"source": source, "page": c.page, "chunk_index": c.index, "content": c.content, "embedding": v}
            for c, v in zip(part, vectors, strict=True)
        ]
        db.table("documents").upsert(rows, on_conflict="source,chunk_index").execute()
        log.info("  indexed %d/%d", start + len(part), len(chunks))


if __name__ == "__main__":
    main()
