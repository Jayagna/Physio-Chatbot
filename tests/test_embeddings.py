from types import SimpleNamespace

from google.genai import _transformers as t

from physiobot.embeddings import Embedder, _as_contents


def test_each_text_is_its_own_content():
    # The SDK normalises contents this way before calling gemini-embedding-2.
    assert len(t.t_contents(_as_contents(["a", "b", "c"]))) == 3


def test_embed_documents_batches_and_returns_one_vector_per_chunk(monkeypatch):
    calls = []

    class Models:
        def embed_content(self, model, contents, config):
            calls.append(contents)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents])

    embedder = Embedder.__new__(Embedder)
    embedder.client = SimpleNamespace(models=Models())
    embedder.model = "gemini-embedding-2"
    embedder.config = None

    vectors = embedder.embed_documents([f"chunk {i}" for i in range(150)], title="Book")
    assert len(vectors) == 150
    assert [len(c) for c in calls] == [100, 50]
    assert calls[0][0].parts[0].text == "title: Book | text: chunk 0"
