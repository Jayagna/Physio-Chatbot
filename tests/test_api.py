from fastapi.testclient import TestClient

from physiobot import api
from physiobot.service import Answer


class StubBot:
    def __init__(self):
        self.calls = []

        class R:
            def ping(self):
                return True

        self.retriever = R()

    def answer(self, question, history):
        self.calls.append((question, history))
        return Answer(answer="Gentle walking helps [1].", sources=[{"source": "Tidy", "page": 3}])


def client_with(bot):
    api.app.dependency_overrides[api.get_bot] = lambda: bot
    api.limiter.hits.clear()
    return TestClient(api.app)


def test_chat_returns_answer_and_sources():
    bot = StubBot()
    res = client_with(bot).post("/chat", json={"message": "back pain?", "history": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    assert res.json() == {"answer": "Gentle walking helps [1].", "sources": [{"source": "Tidy", "page": 3}], "urgent": False}
    assert bot.calls[0][0] == "back pain?"


def test_chat_validates_input():
    c = client_with(StubBot())
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "x" * 1001}).status_code == 422


def test_rate_limit():
    c = client_with(StubBot())
    codes = [c.post("/chat", json={"message": "hi"}).status_code for _ in range(api.limiter.per_minute + 1)]
    assert codes[-1] == 429 and set(codes[:-1]) == {200}


def test_health():
    assert client_with(StubBot()).get("/health").json() == {"status": "ok", "db": "ok"}


def test_rate_limiter_window():
    rl = api.RateLimiter(2)
    assert rl.allow("a", 0) and rl.allow("a", 1)
    assert not rl.allow("a", 2)
    assert rl.allow("a", 61.5)
