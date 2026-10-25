"""PhysioBOT's answer pipeline: triage → retrieve → generate."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types

from .config import get_settings
from .embeddings import Embedder
from .prompts import system_prompt, user_turn
from .retriever import Passage, Retriever, SupabaseRetriever
from .safety import check_red_flags, urgent_reply

log = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    urgent: bool = False


def cited_sources(text: str, passages: list[Passage]) -> list[dict]:
    """The passages the model actually cited, deduplicated by source and page."""
    cited = {int(n) for n in re.findall(r"\[(\d{1,2})\]", text)}
    seen: set[tuple[str, int | None]] = set()
    out = []
    for i, p in enumerate(passages, start=1):
        key = (p.source, p.page)
        if i in cited and key not in seen:
            seen.add(key)
            out.append({"source": p.source, "page": p.page})
    return out


class PhysioBot:
    def __init__(
        self,
        client: genai.Client | None = None,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
    ):
        s = get_settings()
        self.client = client or genai.Client(api_key=s.google_api_key)
        self.embedder = embedder or Embedder(self.client)
        self.retriever = retriever or SupabaseRetriever()
        self.model = s.chat_model

    def answer(self, question: str, history: list[Turn] | None = None) -> Answer:
        flag = check_red_flags(question)
        if flag:
            log.info("red flag triggered: %s", flag.name)
            return Answer(answer=urgent_reply(flag), urgent=True)

        passages = self.retriever.search(self.embedder.embed_query(question))

        contents = [
            types.Content(role="model" if t.role == "assistant" else "user", parts=[types.Part(text=t.content)])
            for t in (history or [])
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=user_turn(question, passages))]))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt(),
                temperature=0.4,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            text = "Sorry, I couldn't put together an answer. Please try rephrasing, or ask the clinic directly."
        return Answer(answer=text, sources=cited_sources(text, passages))
