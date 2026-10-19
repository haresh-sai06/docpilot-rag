"""Turn reports/*.json into the Markdown tables used in the README, incl. per-stage latency and a
cost estimate derived from MEASURED token counts.

The $ figures are an estimate: measured tokens x an assumed per-token price (pass your provider's
current price with --price-in/--price-out, USD per 1M tokens). Local Ollama inference has no API cost.

python eval/summarize.py --price-in 0.15 --price-out 0.60
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-in", type=float, default=0.15)
    ap.add_argument("--price-out", type=float, default=0.60)
    a = ap.parse_args()
    rep = ROOT / "reports"
    gen = json.loads((rep / "generation_eval.json").read_text())
    answers = json.loads((rep / "generation_answers.json").read_text())
    out = [f"# Generation eval ({gen['llm_model']} via {gen['llm_provider']}, n={gen['results'][0]['n']} questions, seed {gen['subset_seed']})\n",
           "| Config | Faithfulness (NLI) | Answer relevance (cos) | Context hit rate@4 (gold passage in top-4 context) | p50 cache off | p95 cache off | p50 cache on (repeat) | p95 cache on (repeat) | Avg prompt tok | Avg completion tok |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in gen["results"]:
        off, on = r["cache_off"], r["cache_on_repeat"]
        out.append(f"| {r['config']} | {r['faithfulness']} | {r['answer_relevance']} | {r['context_hit_rate_at_4']} | {off['p50_ms']:.0f} ms | "
                   f"{off['p95_ms']:.0f} ms | {on['p50_ms']:.2f} ms | {on['p95_ms']:.2f} ms | {off['avg_prompt_tokens']} | {off['avg_completion_tokens']} |")
    out += ["", "## Per-stage latency, cache off (median over questions)", "", "| Config | Retrieval (BM25+dense+fuse) | Rerank | LLM generate |", "|---|---|---|---|"]
    for name, rows in answers.items():
        t = [r["timings"] for r in rows]
        med = lambda k, t=t: np.median([x.get(k, 0) for x in t])  # noqa: E731
        out.append(f"| {name} | {med('retrieve_ms') + med('fuse_ms'):.0f} ms | {med('rerank_ms'):.0f} ms | {med('generate_ms'):.0f} ms |")
    out += ["", f"## Cost estimate per 1,000 queries (assumed price ${a.price_in}/1M input, ${a.price_out}/1M output tokens)", "",
            "| Config | Cache off (measured tokens) | Cache on, repeat of the same questions (measured LLM calls) |", "|---|---|---|"]
    for r in gen["results"]:
        off = r["cache_off"]
        per_q = (off["avg_prompt_tokens"] * a.price_in + off["avg_completion_tokens"] * a.price_out) / 1e6
        llm_calls_on = r["cache_on_repeat"]["llm_calls"] / r["n"]
        out.append(f"| {r['config']} | ${per_q * 1000:.4f} | ${per_q * 1000 * llm_calls_on:.4f} |")
    lat = [x["latency_ms"] for rows in answers.values() for x in rows]
    out.append(f"\nAll cold end-to-end latencies: p50 {pct(lat, 50):.0f} ms, p95 {pct(lat, 95):.0f} ms over {len(lat)} runs.")
    (rep / "generation_eval.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
