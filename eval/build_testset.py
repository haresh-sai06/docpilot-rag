"""Build the synthetic evaluation set: sample chunks, ask the LLM to write one question each
chunk answers, keep the chunk (and its page) as retrieval ground truth.

python eval/build_testset.py --n 100 --out eval/testset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docpilot.config import get_settings  # noqa: E402
from docpilot.corpus import load_chunks  # noqa: E402
from docpilot.llm import LLM  # noqa: E402

SYSTEM = ("You write evaluation questions for a documentation search engine. Given a passage from the FastAPI docs, "
          "write ONE specific question that a developer could ask and that this passage answers. Do not mention "
          "'the passage' or 'the text'. Output only the question.")


def prose_ratio(text: str) -> float:
    code = sum(len(b) for b in re.findall(r"```.*?```", text, re.S))
    return 1 - code / max(len(text), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="eval/testset.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    s = get_settings()
    chunks = [c for c in load_chunks(s.data_dir, s.corpus_version) if len(c.text) >= 400 and prose_ratio(c.text) > 0.6]
    rng = random.Random(a.seed)
    rng.shuffle(chunks)
    # at most one chunk per page first (coverage), then fill
    seen, ordered, rest = set(), [], []
    for c in chunks:
        (rest if c.page in seen else ordered).append(c)
        seen.add(c.page)
    pool = ordered + rest
    llm = LLM(s)
    out = Path(a.out)
    done = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    used = {d["chunk_id"] for d in done}
    with out.open("a") as f:
        for c in pool:
            if len(done) >= a.n:
                break
            if c.id in used:
                continue
            q = llm.chat(SYSTEM, c.text[:1500], max_tokens=60).text.strip().strip('"').split("\n")[0].strip()
            if not (15 <= len(q) <= 220 and q.endswith("?")) or "passage" in q.lower():
                continue
            row = {"id": len(done) + 1, "question": q, "chunk_id": c.id, "page": c.page, "section": c.section,
                   "generator": s.llm_model}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done.append(row)
            print(row["id"], q, flush=True)


if __name__ == "__main__":
    main()
