"""End-to-end agent evaluation on a seeded subset of the test set, with vs without the reranker,
plus latency/cost with the cache cold vs warm.

Generation metrics (no LLM judge needed, so they are cheap and reproducible):
  faithfulness      - share of answer sentences entailed by the retrieved context, scored with an NLI
                      cross-encoder (cross-encoder/nli-deberta-v3-xsmall); a sentence counts as supported
                      if P(entailment) >= 0.5 for at least one context passage
  answer_relevance  - cosine similarity between question and answer embeddings (bge-small)
  context_hit_rate_at_4 - gold passage is among the top-4 passages given to the LLM (retrieval measure, NOT answer correctness)

python eval/run_generation_eval.py --n 25 --phase generate --configs norerank
  python eval/run_generation_eval.py --n 25 --phase generate --configs rerank
  python eval/run_generation_eval.py --n 25 --phase score
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
        pairs = [(c[:1000], s) for s in sents for c in contexts]
        logits = np.asarray(self.nli.predict(pairs, batch_size=8))
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


CONFIGS = {"norerank": ("Hybrid, no rerank", False), "rerank": ("Hybrid + rerank", True)}


def generate(s, a, tests):
    """Phase 1: run the agent (cache cold, then identical repeat traffic) and save raw outputs per config."""
    from docpilot.llm import LLM

    chunks = load_chunks(s.data_dir, s.corpus_version)
    cache = KVCache(s.cache_path, enabled=True)
    retriever = HybridRetriever(chunks, Embedder(s.embed_model, cache), Reranker(s.rerank_model))
    LLM(s).chat("You are terse.", "Say OK.", max_tokens=4)  # warm-up: load the model before timing
    retriever.search("warm up the reranker", k=4, rerank=True)
    for slug in a.configs:
        name, rerank = CONFIGS[slug]
        cache.clear()
        agent = build(s, cache_enabled=True, rerank=rerank, retriever=retriever)
        retriever.embedder.cache = agent.cache
        cold = run_pass(agent, tests)
        warm = run_pass(agent, tests)  # identical repeat traffic -> response cache hits
        keep = ("id", "question", "gold", "answer", "citations", "latency_ms", "usage", "timings", "cached")
        raw = {"config": name, "cold": [{k: r[k] for k in keep} for r in cold], "warm": [{k: r[k] for k in keep} for r in warm]}
        (ROOT / "reports" / f"gen_raw_{slug}.json").write_text(json.dumps(raw, indent=2))
        print("saved", slug, summarise(cold), flush=True)


def score(s, a, tests):
    """Phase 2: score saved answers (separate process keeps peak memory low). Contexts = the cited chunks,
    i.e. exactly the passages the LLM saw (truncated to 1,000 chars as in the prompt)."""
    chunks = {c.id: c for c in load_chunks(s.data_dir, s.corpus_version)}
    scorer = Scorer(Embedder(s.embed_model, KVCache(s.cache_path, enabled=False)))
    results, answers = [], {}
    for slug in CONFIGS:
        f = ROOT / "reports" / f"gen_raw_{slug}.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())
        cold, warm = raw["cold"], raw["warm"]
        faith = [scorer.faithfulness(r["answer"], [chunks[c["chunk_id"]].text[:1000] for c in r["citations"]]) for r in cold]
        rel = [scorer.relevance(r["question"], r["answer"]) for r in cold]
        ctx = [r["gold"] in [c["chunk_id"] for c in r["citations"]] for r in cold]
        f_valid = [x for x in faith if x is not None]
        results.append({"config": raw["config"], "n": len(cold), "faithfulness": round(float(np.mean(f_valid)), 3),
                        "answer_relevance": round(float(np.mean(rel)), 3), "context_hit_rate_at_4": round(float(np.mean(ctx)), 3),
                        "cache_off": summarise(cold), "cache_on_repeat": summarise(warm)})
        answers[raw["config"]] = cold
        print(results[-1], flush=True)
    meta = {"llm_provider": s.llm_provider, "llm_model": s.llm_model, "embed_model": s.embed_model, "nli_model": NLI_MODEL,
            "subset_seed": a.seed, "subset_ids": [t["id"] for t in tests], "results": results}
    (ROOT / "reports" / "generation_eval.json").write_text(json.dumps(meta, indent=2))
    (ROOT / "reports" / "generation_answers.json").write_text(json.dumps(answers, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--phase", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    a = ap.parse_args()
    s = get_settings()
    tests = random.Random(a.seed).sample(load_testset(), a.n)
    if a.phase in ("generate", "all"):
        generate(s, a, tests)
    if a.phase in ("score", "all"):
        score(s, a, tests)


if __name__ == "__main__":
    main()
