"""Builds the shared components once (corpus, index, models, cache)."""
from __future__ import annotations

import os
from functools import lru_cache

from qdrant_client import QdrantClient

from .agent import DocPilotAgent
from .cache import KVCache
from .config import Settings, get_settings
from .corpus import load_chunks
from .llm import LLM
from .retrieval import Embedder, HybridRetriever, Reranker


def build(settings: Settings | None = None, cache_enabled: bool | None = None, rerank: bool = True, mode: str = "hybrid",
          retriever: HybridRetriever | None = None) -> DocPilotAgent:
    s = settings or get_settings()
    cache = KVCache(s.cache_path, enabled=s.cache_enabled if cache_enabled is None else cache_enabled)
    if retriever is None:
        chunks = load_chunks(s.data_dir, s.corpus_version)
        retriever = HybridRetriever(chunks, Embedder(s.embed_model, cache), Reranker(s.rerank_model), qdrant=qdrant_client())
    return DocPilotAgent(retriever, LLM(s), cache, top_k=s.top_k, candidates=s.candidates, rerank=rerank, mode=mode)


def qdrant_client() -> QdrantClient:
    """QDRANT_URL -> remote server; DOCPILOT_QDRANT_PATH -> embedded on-disk (index survives restarts); else in-memory."""
    if os.getenv("QDRANT_URL"):
        return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.getenv("QDRANT_API_KEY"))
    if os.getenv("DOCPILOT_QDRANT_PATH"):
        return QdrantClient(path=os.environ["DOCPILOT_QDRANT_PATH"])
    return QdrantClient(":memory:")


@lru_cache(maxsize=1)
def default_agent() -> DocPilotAgent:
    return build()
