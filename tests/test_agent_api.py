from fastapi.testclient import TestClient

from conftest import FakeEmbedder
from docpilot.agent import DocPilotAgent
from docpilot.cache import KVCache
from docpilot.config import Settings
from docpilot.llm import LLM
from docpilot.retrieval import HybridRetriever


def make_agent(chunks, tmp_path, provider="none"):
    s = Settings(data_dir=tmp_path, llm_provider=provider)
    cache = KVCache(tmp_path / "c.sqlite")
    return DocPilotAgent(HybridRetriever(chunks, FakeEmbedder()), LLM(s), cache, top_k=2, rerank=False)


def test_routes(chunks, tmp_path):
    a = make_agent(chunks, tmp_path)
    assert a.ask("hello there")["route"] == "smalltalk"
    assert a.ask("pizza?")["route"] == "out_of_scope"
    r = a.ask("How do I enable CORS middleware?")
    assert r["route"] == "rag" and r["citations"][0]["page"] == "tutorial/cors.md"


def test_response_cache(chunks, tmp_path):
    a = make_agent(chunks, tmp_path)
    first = a.ask("How do I build a docker image?")
    second = a.ask("how do i build a DOCKER image?")
    assert not first["cached"] and second["cached"] and second["answer"] == first["answer"]


def test_llm_called_with_context(chunks, tmp_path, monkeypatch):
    a = make_agent(chunks, tmp_path, provider="ollama")
    seen = {}

    def fake_chat(system, user, max_tokens=None):
        from docpilot.llm import LLMResult

        seen["user"] = user
        return LLMResult("Use CORSMiddleware [1].", 100, 7, 0.01)

    monkeypatch.setattr(a.llm, "chat", fake_chat)
    r = a.ask("How do I allow cross-origin requests with CORS?")
    assert "[1]" in seen["user"] and r["usage"] == {"prompt_tokens": 100, "completion_tokens": 7}


def test_api(chunks, tmp_path, monkeypatch):
    from docpilot import api

    agent = make_agent(chunks, tmp_path)
    monkeypatch.setattr(api, "default_agent", lambda: agent)
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ask", json={"question": "What are path parameters?"})
    assert r.status_code == 200 and r.json()["citations"]
    s = c.get("/search", params={"q": "jwt tokens", "k": 2, "rerank": False})
    assert s.status_code == 200 and len(s.json()) == 2
