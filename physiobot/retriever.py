"""Vector search over the knowledge base stored in Supabase pgvector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from supabase import Client, create_client

from .config import get_settings


@dataclass(frozen=True)
class Passage:
    source: str
    page: int | None
    content: str
    similarity: float


class Retriever(Protocol):
    def search(self, embedding: list[float]) -> list[Passage]: ...


class SupabaseRetriever:
    def __init__(self, client: Client | None = None):
        s = get_settings()
        self.client = client or create_client(s.supabase_url, s.supabase_service_key)
        self.top_k = s.top_k
        self.min_similarity = s.min_similarity

    def search(self, embedding: list[float]) -> list[Passage]:
        result = self.client.rpc(
            "match_documents",
            {"query_embedding": embedding, "match_count": self.top_k, "min_similarity": self.min_similarity},
        ).execute()
        return [
            Passage(source=r["source"], page=r.get("page"), content=r["content"], similarity=r["similarity"])
            for r in result.data or []
        ]

    def ping(self) -> bool:
        """Cheap query that also keeps the free-tier Supabase project active."""
        self.client.rpc("ping").execute()
        return True
