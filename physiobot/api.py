"""FastAPI service exposing PhysioBOT to the clinic website.

Run locally:  uvicorn physiobot.api:app --reload
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .service import PhysioBot, Turn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("physiobot")

settings = get_settings()
app = FastAPI(title="PhysioBOT", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)


class Source(BaseModel):
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    urgent: bool = False


class RateLimiter:
    """Sliding one-minute window per client IP. Enough for a single free instance."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


limiter = RateLimiter(settings.rate_limit_per_minute)


@lru_cache
def get_bot() -> PhysioBot:
    return PhysioBot()


def client_ip(request: Request) -> str:
    # Render and most hosts put the real client first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/health")
def health(bot: PhysioBot = Depends(get_bot)) -> dict:
    db = "ok"
    try:
        bot.retriever.ping()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - health must never raise
        log.exception("database ping failed")
        db = "error"
    return {"status": "ok", "db": db}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, bot: PhysioBot = Depends(get_bot)) -> ChatResponse:
    if not limiter.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many messages. Please wait a minute.")
    history = [Turn(role=t.role, content=t.content[:2000]) for t in body.history[-6:]]
    try:
        result = bot.answer(body.message.strip(), history)
    except Exception:
        log.exception("chat failed")
        raise HTTPException(status_code=502, detail="The assistant is unavailable right now.") from None
    return ChatResponse(answer=result.answer, sources=[Source(**s) for s in result.sources], urgent=result.urgent)
