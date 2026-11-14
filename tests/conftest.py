"""Test fixtures: real API, real redactor (regex mode) and real Chroma index, with small fakes for the
models and the LLM so tests run offline in seconds."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.container import Services  # noqa: E402
from app.services.priority import PriorityService  # noqa: E402
from app.services.redact import Redactor  # noqa: E402
from app.services.summarise import Summariser  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


class HashEmbedder:
    """Bag-of-words hashing embedder: good enough for keyword-level retrieval in tests."""
    model_id = "test/hash-embedder"

    def encode(self, texts, is_query=False):
        out = []
        for t in texts:
            v = [0.0] * 512
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) > 2:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM:
    model = "fake-llm"
    configured = True

    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens=400, json_mode=False):
        self.calls.append((system, user))
        if "summarise" in system.lower():
            return "Finance team VPN drops every five minutes since the update and blocks month-end closing work."
        return "Follow the VPN SOP: switch to a wired connection and disable adapter power saving [1]."

    def complete_json(self, system, user, max_tokens=200):
        return {"high": 0.7, "medium": 0.2, "low": 0.1}


class FakeClassifier:
    model_id = "fake/queue-classifier"

    def __init__(self, conf=0.81):
        self.conf = conf

    def predict(self, texts, top_k=3):
        rows = [{"label": "Technical Support", "score": self.conf},
                {"label": "IT Support", "score": round((1 - self.conf) * 0.7, 4)},
                {"label": "Product Support", "score": round((1 - self.conf) * 0.3, 4)}]
        return [rows[:top_k] for _ in texts]


class FakeSentiment:
    def __call__(self, text, **kw):
        return [[{"label": "negative", "score": 0.88}, {"label": "neutral", "score": 0.1},
                 {"label": "positive", "score": 0.02}]]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def rag(tmp_path, fake_llm):
    pipe = RAGPipeline(index_dir=str(tmp_path / "index"), score_threshold=0.15, embedder=HashEmbedder(),
                       llm=fake_llm, collection_name="test_kb")
    pipe.ingest([str(ROOT / "rag" / "docs")])
    return pipe


@pytest.fixture
def services(rag, fake_llm):
    settings = Settings(_env_file=None, openai_api_key="test", api_key=None)
    return Services(settings, llm=fake_llm, redactor=Redactor(None), classifier=FakeClassifier(),
                    priority=PriorityService(None, FakeSentiment(), fake_llm, "llm", "fake-sentiment"),
                    summariser=Summariser(fake_llm), rag=rag)


@pytest.fixture
def client(services, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main, "get_settings", lambda: services.settings)
    main.app.state.services = services
    with TestClient(main.app) as c:
        yield c
    del main.app.state.services
