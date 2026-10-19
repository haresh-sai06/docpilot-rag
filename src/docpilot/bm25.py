"""Okapi BM25 over chunk texts (sparse lexical retrieval), implemented with NumPy/SciPy-free dicts."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9_]+")
STOP = set("a an the and or of to in on for is are be with as by it this that from at how what which can do does i you "
           "your my we when where why use using".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.tf = [Counter(t) for t in toks]
        df = Counter(term for t in toks for term in set(t))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, t in enumerate(self.tf):
            for term in t:
                self.postings.setdefault(term, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return s

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]
