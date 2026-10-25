"""Settings read from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    google_api_key: str = ""
    supabase_url: str = ""
    supabase_service_key: str = ""
    chat_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768
    top_k: int = 6
    min_similarity: float = 0.5
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_per_minute: int = 8


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    env = os.environ.get
    return Settings(
        google_api_key=env("GOOGLE_API_KEY", ""),
        supabase_url=env("SUPABASE_URL", ""),
        supabase_service_key=env("SUPABASE_SERVICE_ROLE_KEY", ""),
        chat_model=env("GEMINI_CHAT_MODEL", Settings.chat_model),
        embedding_model=env("GEMINI_EMBEDDING_MODEL", Settings.embedding_model),
        top_k=int(env("TOP_K", str(Settings.top_k))),
        min_similarity=float(env("MIN_SIMILARITY", str(Settings.min_similarity))),
        allowed_origins=_list(env("ALLOWED_ORIGINS", "http://localhost:3000")),
        rate_limit_per_minute=int(env("RATE_LIMIT_PER_MINUTE", str(Settings.rate_limit_per_minute))),
    )
