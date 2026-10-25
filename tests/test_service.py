from types import SimpleNamespace

from physiobot.retriever import Passage
from physiobot.service import PhysioBot, Turn, cited_sources

PASSAGES = [
    Passage("Tidy's Physiotherapy", 120, "Low back pain is commonly mechanical...", 0.8),
    Passage("Tidy's Physiotherapy", 121, "McKenzie extension exercises...", 0.7),
    Passage("Tidy's Physiotherapy", 120, "Posture advice...", 0.6),
]


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, q):
        self.queries.append(q)
        return [0.0] * 768


class FakeRetriever:
    def search(self, embedding):
        return PASSAGES


def make_bot(text="Try gentle extension exercises [2]. Keep active [1][3]."):
    models = FakeModels(text)
    client = SimpleNamespace(models=models)
    embedder = FakeEmbedder()
    bot = PhysioBot(client=client, embedder=embedder, retriever=FakeRetriever())
    return bot, models, embedder


def test_cited_sources_dedupes_by_page():
    assert cited_sources("see [1], [2] and [3]", PASSAGES) == [
        {"source": "Tidy's Physiotherapy", "page": 120},
        {"source": "Tidy's Physiotherapy", "page": 121},
    ]


def test_cited_sources_ignores_unknown_numbers():
    assert cited_sources("see [9]", PASSAGES) == []


def test_answer_runs_rag_pipeline_with_history():
    bot, models, embedder = make_bot()
    result = bot.answer("What helps back pain?", [Turn("user", "hi"), Turn("assistant", "hello")])

    assert embedder.queries == ["What helps back pain?"]
    assert not result.urgent
    assert {"source": "Tidy's Physiotherapy", "page": 121} in result.sources

    call = models.calls[0]
    roles = [c.role for c in call["contents"]]
    assert roles == ["user", "model", "user"]
    prompt = call["contents"][-1].parts[0].text
    assert "[2] (Tidy's Physiotherapy, page 121)" in prompt
    assert "Nistha Physiotherapy" in call["config"].system_instruction


def test_red_flag_skips_llm():
    bot, models, embedder = make_bot()
    result = bot.answer("I have chest pain and sweating")
    assert result.urgent
    assert models.calls == [] and embedder.queries == []


def test_empty_model_reply_falls_back():
    bot, _, _ = make_bot(text="")
    assert "couldn't" in bot.answer("hello").answer
