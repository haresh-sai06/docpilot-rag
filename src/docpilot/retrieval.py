"""Hybrid retrieval: BM25 + dense (bge-small in Qdrant) fused with Reciprocal Rank Fusion,
optionally re-ranked by a cross-encoder (ms-marco-MiniLM-L-6-v2). All models run on CPU via ONNX."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient, models

from .bm25 import BM25
from .cache import KVCache
from .corpus import Chunk

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # per-retriever rank, for debugging / UI


class Embedder:
    def __init__(self, model_name: str, cache: KVCache):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.name, self.cache = model_name, cache

    def embed_query(self, text: str) -> list[float]:
        cached = self.cache.get(f"emb:{self.name}", text)
        if cached is not None:
            return cached
        vec = next(iter(self.model.query_embed([text]))).tolist()
        self.cache.set(f"emb:{self.name}", text, vec)
        return vec

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.passage_embed(texts, batch_size=64)))


class Reranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = list(self.model.rerank(query, [h.chunk.text for h in hits]))
        for h, s in zip(hits, scores):
            h.sources["rerank_score"] = float(s)
            h.score = float(s)
        return sorted(hits, key=lambda h: -h.score)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, reranker: Reranker | None = None,
                 qdrant: QdrantClient | None = None, collection: str = "docs"):
        self.chunks, self.embedder, self.reranker = chunks, embedder, reranker
        self.bm25 = BM25([c.text for c in chunks])
        self.qdrant = qdrant or QdrantClient(":memory:")
        self.collection = collection
        if not self.qdrant.collection_exists(collection):
            vecs = embedder.embed_docs([c.text for c in chunks])
            self.qdrant.create_collection(collection, vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE))
            self.qdrant.upload_points(collection, [models.PointStruct(id=i, vector=v.tolist(), payload={"chunk_id": c.id})
                                                   for i, (c, v) in enumerate(zip(chunks, vecs))])
        self.timings: dict[str, float] = {}

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        res = self.qdrant.query_points(self.collection, query=self.embedder.embed_query(query), limit=k).points
        return [(int(p.id), float(p.score)) for p in res]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.top(query, k)

    def search(self, query: str, k: int = 4, mode: str = "hybrid", rerank: bool = True, candidates: int = 20) -> list[Hit]:
        """mode: 'bm25' | 'dense' | 'hybrid'. With rerank=True the top `candidates` are re-scored."""
        t0 = time.perf_counter()
        n = candidates if rerank else k
        ranked: dict[int, Hit] = {}
        lists = {}
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = self.sparse(query, candidates)
        if mode in ("dense", "hybrid"):
            lists["dense"] = self.dense(query, candidates)
        t1 = time.perf_counter()
        for name, lst in lists.items():
            for rank, (i, _) in enumerate(lst):
                h = ranked.setdefault(i, Hit(self.chunks[i], 0.0))
                h.score += 1.0 / (RRF_K + rank + 1) if mode == "hybrid" else (1.0 / (rank + 1))
                h.sources[f"{name}_rank"] = rank + 1
        hits = sorted(ranked.values(), key=lambda h: -h.score)[:n]
        t2 = time.perf_counter()
        if rerank and self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        t3 = time.perf_counter()
        self.timings = {"retrieve_ms": (t1 - t0) * 1000, "fuse_ms": (t2 - t1) * 1000, "rerank_ms": (t3 - t2) * 1000}
        return hits[:k]
