"""Retrieval ablation over the 100-question test set.

Metrics (one gold chunk per question): Recall@k (= hit rate @k), MRR@10, and page-level Hit@5
(any chunk from the gold page). Also per-query retrieval latency.

python eval/run_retrieval_eval.py
"""
from __future__ import annotations

import json
import time

from common import ROOT, load_testset, pct
from docpilot.cache import KVCache
from docpilot.config import get_settings
from docpilot.corpus import load_chunks
from docpilot.retrieval import Embedder, HybridRetriever, Reranker

CONFIGS = [("BM25", "bm25", False), ("Dense (bge-small)", "dense", False), ("Hybrid (RRF)", "hybrid", False),
           ("BM25 + rerank", "bm25", True), ("Dense + rerank", "dense", True), ("Hybrid + rerank", "hybrid", True)]


def evaluate(retriever, tests, mode, rerank, k=10):
    r_at = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = page_hit5 = 0.0
    lat = []
    for t in tests:
        t0 = time.perf_counter()
        hits = retriever.search(t["question"], k=k, mode=mode, rerank=rerank, candidates=20)
        lat.append((time.perf_counter() - t0) * 1000)
        ids = [h.chunk.id for h in hits]
        if t["chunk_id"] in ids:
            rank = ids.index(t["chunk_id"]) + 1
            mrr += 1 / rank
            for kk in r_at:
                r_at[kk] += rank <= kk
        page_hit5 += t["page"] in [h.chunk.page for h in hits[:5]]
    n = len(tests)
    return {**{f"recall@{kk}": round(v / n, 3) for kk, v in r_at.items()}, "mrr@10": round(mrr / n, 3),
            "page_hit@5": round(page_hit5 / n, 3), "p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1)}


def main():
    s = get_settings()
    tests = load_testset()
    chunks = load_chunks(s.data_dir, s.corpus_version)
    retriever = HybridRetriever(chunks, Embedder(s.embed_model, KVCache(s.cache_path, enabled=False)), Reranker(s.rerank_model))
    for t in tests[:5]:  # warm-up (model load / ONNX graph init)
        retriever.search(t["question"], k=10, rerank=True)
    rows = []
    for name, mode, rr in CONFIGS:
        m = evaluate(retriever, tests, mode, rr)
        rows.append({"config": name, **m})
        print(rows[-1], flush=True)
    out = {"testset_size": len(tests), "corpus_chunks": len(chunks), "corpus_pages": len({c.page for c in chunks}),
           "embed_model": s.embed_model, "rerank_model": s.rerank_model, "rerank_candidates": 20, "results": rows}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "retrieval_eval.json").write_text(json.dumps(out, indent=2))
    lines = ["| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Page Hit@5 | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['recall@1']} | {r['recall@3']} | {r['recall@5']} | {r['recall@10']} | {r['mrr@10']} | {r['page_hit@5']} | {r['p50_ms']} | {r['p95_ms']} |" for r in rows]
    (ROOT / "reports" / "retrieval_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
