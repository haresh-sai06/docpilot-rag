from conftest import FakeEmbedder
from docpilot.bm25 import BM25, tokenize
from docpilot.retrieval import HybridRetriever


def test_tokenize_drops_stopwords():
    assert tokenize("How do I use the CORS middleware?") == ["cors", "middleware"]


def test_bm25_ranks_lexical_match_first(chunks):
    bm = BM25([c.text for c in chunks])
    top = bm.top("cors allow_origins", k=3)
    assert chunks[top[0][0]].page == "tutorial/cors.md"
    assert bm.top("zzzz nothing", k=3) == []


def test_hybrid_rrf_and_modes(chunks):
    r = HybridRetriever(chunks, FakeEmbedder())
    for mode in ("bm25", "dense", "hybrid"):
        hits = r.search("docker image uvicorn container", k=2, mode=mode, rerank=False)
        assert hits[0].chunk.page == "deployment/docker.md", mode
    h = r.search("jwt access tokens", k=3, mode="hybrid", rerank=False)[0]
    assert {"bm25_rank", "dense_rank"} <= set(h.sources)


def test_reranker_reorders(chunks):
    class ReverseReranker:
        def rerank(self, q, hits):
            for i, h in enumerate(hits):
                h.score = float(i)
                h.sources["rerank_score"] = float(i)
            return sorted(hits, key=lambda h: -h.score)

    r = HybridRetriever(chunks, FakeEmbedder(), ReverseReranker())
    plain = r.search("path parameters", k=5, rerank=False)
    rr = r.search("path parameters", k=5, rerank=True, candidates=5)
    assert plain[0].chunk.id != rr[0].chunk.id
