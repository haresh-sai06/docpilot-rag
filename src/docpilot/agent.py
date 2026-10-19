"""LangGraph agent: route -> retrieve (hybrid) -> grade -> [rewrite -> retrieve] -> generate.

Routing is deterministic and cheap (no LLM call): small-talk / out-of-scope questions are answered
without retrieval; documentation questions go through the RAG path. If the best reranked passage is
weak, the agent rewrites the query once (keyword expansion) and retries before answering, and it
abstains instead of hallucinating when evidence is still weak.
"""
from __future__ import annotations

import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .bm25 import tokenize
from .cache import KVCache
from .llm import LLM
from .retrieval import Hit, HybridRetriever

SYSTEM = ("You are DocPilot, an assistant for the FastAPI documentation. Answer ONLY from the numbered context. "
          "Be concise (<= 5 sentences). Cite sources like [1]. If the context does not contain the answer, say you don't know.")
SMALL_TALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|who are you|what can you do)\b", re.I)
DOMAIN_TERMS = set("fastapi api endpoint path query body request response pydantic dependency depends security oauth2 jwt "
                   "cors middleware websocket background task router apirouter test testclient deploy docker uvicorn "
                   "header cookie form file upload status code exception openapi swagger async await sql database "
                   "model schema validation parameter field settings event lifespan template static".split())
WEAK_RERANK_SCORE = 0.0  # ms-marco logits: < 0 means "probably not relevant"


class State(TypedDict, total=False):
    question: str
    route: str
    query: str
    hits: list[Hit]
    rewrites: int
    answer: str
    citations: list[dict]
    usage: dict
    timings: dict


class DocPilotAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLM, cache: KVCache, top_k: int = 4, candidates: int = 20,
                 rerank: bool = True, mode: str = "hybrid"):
        self.r, self.llm, self.cache = retriever, llm, cache
        self.top_k, self.candidates, self.rerank, self.mode = top_k, candidates, rerank, mode
        self.graph = self._build()

    # ---- nodes ----
    def route(self, s: State) -> State:
        q = s["question"]
        if SMALL_TALK.match(q):
            return {"route": "smalltalk"}
        toks = set(tokenize(q))
        in_scope = bool(toks & DOMAIN_TERMS) or len(toks) >= 4
        return {"route": "rag" if in_scope else "out_of_scope", "query": q, "rewrites": 0}

    def retrieve(self, s: State) -> State:
        t = time.perf_counter()
        hits = self.r.search(s["query"], k=self.top_k, mode=self.mode, rerank=self.rerank, candidates=self.candidates)
        tm = dict(s.get("timings", {}))
        tm["retrieval_ms"] = tm.get("retrieval_ms", 0) + (time.perf_counter() - t) * 1000
        tm.update({k: v for k, v in self.r.timings.items()})
        return {"hits": hits, "timings": tm}

    def grade(self, s: State) -> str:
        hits = s.get("hits", [])
        if not self.rerank:
            return "generate"
        weak = not hits or hits[0].sources.get("rerank_score", 1.0) < WEAK_RERANK_SCORE
        return "rewrite" if weak and s.get("rewrites", 0) < 1 else "generate"

    def rewrite(self, s: State) -> State:
        # cheap deterministic rewrite: keep content words, add the product name for lexical recall
        toks = tokenize(s["question"])
        return {"query": "FastAPI " + " ".join(toks), "rewrites": s.get("rewrites", 0) + 1}

    def generate(self, s: State) -> State:
        hits = s.get("hits", [])
        cites = [{"n": i + 1, "page": h.chunk.page, "section": h.chunk.section, "chunk_id": h.chunk.id} for i, h in enumerate(hits)]
        context = "\n\n".join(f"[{i + 1}] {h.chunk.text[:1000]}" for i, h in enumerate(hits))
        t = time.perf_counter()
        if self.llm.available:
            res = self.llm.chat(SYSTEM, f"Context:\n{context}\n\nQuestion: {s['question']}\nAnswer:")
            answer, usage = res.text, {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}
        else:  # extractive fallback: first prose paragraph of the best passage
            body = hits[0].chunk.text.split("\n\n", 1)[-1] if hits else ""
            answer = (body[:600] + " [1]") if body else "I couldn't find this in the FastAPI docs."
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
        tm = dict(s.get("timings", {}))
        tm["generate_ms"] = (time.perf_counter() - t) * 1000
        return {"answer": answer, "citations": cites, "usage": usage, "timings": tm}

    def smalltalk(self, s: State) -> State:
        return {"answer": "Hi! I'm DocPilot. Ask me anything about FastAPI: routing, validation, dependencies, security, deployment...",
                "citations": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    def out_of_scope(self, s: State) -> State:
        return {"answer": "That looks outside the FastAPI documentation, so I won't guess.", "citations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    def _build(self):
        g = StateGraph(State)
        for name in ("route", "retrieve", "rewrite", "generate", "smalltalk", "out_of_scope"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("route")
        g.add_conditional_edges("route", lambda s: s["route"], {"rag": "retrieve", "smalltalk": "smalltalk", "out_of_scope": "out_of_scope"})
        g.add_conditional_edges("retrieve", self.grade, {"rewrite": "rewrite", "generate": "generate"})
        g.add_edge("rewrite", "retrieve")
        for n in ("generate", "smalltalk", "out_of_scope"):
            g.add_edge(n, END)
        return g.compile()

    # ---- public ----
    def cache_key(self, question: str) -> str:
        return f"{question}|{self.mode}|{self.rerank}|{self.top_k}|{self.llm.provider}:{self.llm.s.llm_model}"

    def ask(self, question: str) -> dict:
        t = time.perf_counter()
        cached = self.cache.get("answer", self.cache_key(question))
        if cached is not None:
            cached["cached"] = True
            cached["latency_ms"] = (time.perf_counter() - t) * 1000
            return cached
        out = self.graph.invoke({"question": question})
        resp = {"answer": out["answer"], "citations": out.get("citations", []), "route": out.get("route"),
                "rewrites": out.get("rewrites", 0), "usage": out.get("usage", {}), "timings": out.get("timings", {}),
                "contexts": [h.chunk.text for h in out.get("hits", [])], "cached": False}
        self.cache.set("answer", self.cache_key(question), resp)
        resp["latency_ms"] = (time.perf_counter() - t) * 1000
        return resp
