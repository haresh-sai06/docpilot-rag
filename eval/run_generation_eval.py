"""End-to-end agent evaluation on a seeded subset of the test set, with vs without the reranker,
plus latency/cost with the cache cold vs warm.

Generation metrics (no LLM judge needed, so they are cheap and reproducible):
  faithfulness      - share of answer sentences entailed by the retrieved context, scored with an NLI
                      cross-encoder (cross-encoder/nli-deberta-v3-xsmall); a sentence counts as supported
                      if P(entailment) >= 0.5 for at least one context passage
  answer_relevance  - cosine similarity between question and answer embeddings (bge-small)
  context_recall    - gold chunk is among the passages given to the LLM

python eval/run_generation_eval.py --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import re

import numpy as np

from common import ROOT, load_testset, pct
from docpilot.cache import KVCache
from docpilot.config import get_settings
from docpilot.corpus import load_chunks
from docpilot.retrieval import Embedder, HybridRetriever, Reranker
from docpilot.service import build

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def sentences(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12]


class Scorer:
    def __init__(self, embedder: Embedder):
        from sentence_transformers import CrossEncoder

        self.nli = CrossEncoder(NLI_MODEL, device="cpu")
        labels = {v.lower(): k for k, v in self.nli.model.config.id2label.items()}
        self.ent = labels["entailment"]
        self.emb = embedder

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        sents = sentences(answer)
        if not sents or not contexts:
            return None
        pairs = [(c[:1500], s) for s in sents for c in contexts]
        logits = np.asarray(self.nli.predict(pairs, batch_size=16))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        ent = probs[:, self.ent].reshape(len(sents), len(contexts)).max(axis=1)
        return float((ent >= 0.5).mean())

    def relevance(self, q: str, a: str) -> float:
        vq = np.array(next(iter(self.emb.model.query_embed([q]))))
        va = np.array(next(iter(self.emb.model.passage_embed([a]))))
        return float(vq @ va / (np.linalg.norm(vq) * np.linalg.norm(va)))


def run_pass(agent, tests):
    out = []
    for t in tests:
        r = agent.ask(t["question"])
        out.append({**r, "id": t["id"], "question": t["question"], "gold": t["chunk_id"]})
        print(t["id"], f"{r['latency_ms']:.0f}ms cached={r['cached']}", flush=True)
    return out


def summarise(rows):
    lat = [r["latency_ms"] for r in rows]
    pt = [r["usage"].get("prompt_tokens", 0) for r in rows if not r["cached"]]
    ct = [r["usage"].get("completion_tokens", 0) for r in rows if not r["cached"]]
    return {"p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1), "mean_ms": round(float(np.mean(lat)), 1),
            "llm_calls": len(pt), "avg_prompt_tokens": round(float(np.mean(pt)), 1) if pt else 0,
            "avg_completion_tokens": round(float(np.mean(ct)), 1) if ct else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    s = get_settings()
    tests = load_testset()
    tests = random.Random(a.seed).sample(tests, min(a.n, len(tests)))
    chunks = load_chunks(s.data_dir, s.corpus_version)
    cache = KVCache(s.cache_path, enabled=True)
    retriever = HybridRetriever(chunks, Embedder(s.embed_model, cache), Reranker(s.rerank_model))
    scorer = Scorer(retriever.embedder)
    from docpilot.llm import LLM

    LLM(s).chat("You are terse.", "Say OK.", max_tokens=4)  # warm-up: load the model before timing
    retriever.search("warm up the reranker", k=4, rerank=True)
    results, raw = [], {}
    for name, rerank in [("Hybrid, no rerank", False), ("Hybrid + rerank", True)]:
        cache.clear()
        agent = build(s, cache_enabled=True, rerank=rerank, retriever=retriever)
        retriever.embedder.cache = agent.cache
        cold = run_pass(agent, tests)
        warm = run_pass(agent, tests)  # identical repeat traffic -> response cache hits
        faith = [scorer.faithfulness(r["answer"], r["contexts"]) for r in cold]
        rel = [scorer.relevance(r["question"], r["answer"]) for r in cold]
        ctx = [r["gold"] in [c["chunk_id"] for c in r["citations"]] for r in cold]
        f_valid = [f for f in faith if f is not None]
        results.append({"config": name, "n": len(tests), "faithfulness": round(float(np.mean(f_valid)), 3),
                        "answer_relevance": round(float(np.mean(rel)), 3), "context_recall": round(float(np.mean(ctx)), 3),
                        "cache_off": summarise(cold), "cache_on_repeat": summarise(warm)})
        raw[name] = [{k: r[k] for k in ("id", "question", "answer", "citations", "latency_ms", "usage", "timings")} for r in cold]
        print(results[-1], flush=True)
    meta = {"llm_provider": s.llm_provider, "llm_model": s.llm_model, "embed_model": s.embed_model, "nli_model": NLI_MODEL,
            "subset_seed": a.seed, "subset_ids": [t["id"] for t in tests], "results": results}
    (ROOT / "reports" / "generation_eval.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "reports" / "generation_answers.json").write_text(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
