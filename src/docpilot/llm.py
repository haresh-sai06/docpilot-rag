"""Pluggable chat LLM. Providers:
  ollama  - local model via Ollama (default; OLLAMA_BASE_URL, DOCPILOT_LLM_MODEL)
  openai  - any OpenAI-compatible API (OPENAI_API_KEY, optional OPENAI_BASE_URL, DOCPILOT_LLM_MODEL)
  none    - no LLM: extractive answers built from the top reranked passages
Every call returns token usage so cost can be measured.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

from .config import Settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.provider = settings.llm_provider

    def chat(self, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        max_tokens = max_tokens or self.s.max_answer_tokens
        t = time.perf_counter()
        if self.provider == "ollama":
            r = requests.post(f"{self.s.ollama_url}/api/chat", timeout=600, json={
                "model": self.s.llm_model, "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_thread": self.s.llm_threads, "num_ctx": 4096},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            d = r.json()
            return LLMResult(d["message"]["content"].strip(), d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                             time.perf_counter() - t)
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=self.s.llm_model, temperature=0, max_tokens=max_tokens,
                             base_url=self.s.openai_base_url, api_key=os.environ["OPENAI_API_KEY"])
            msg = llm.invoke([("system", system), ("user", user)])
            u = msg.usage_metadata or {}
            return LLMResult(str(msg.content).strip(), u.get("input_tokens", 0), u.get("output_tokens", 0), time.perf_counter() - t)
        raise RuntimeError("llm_provider=none has no generative model")

    @property
    def available(self) -> bool:
        return self.provider in {"ollama", "openai"}
