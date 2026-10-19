"""Runtime configuration (all via environment variables; no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DOCPILOT_DATA_DIR", "data")))
    corpus_version: str = field(default_factory=lambda: _env("DOCPILOT_CORPUS_VERSION", "0.115.0"))
    embed_model: str = field(default_factory=lambda: _env("DOCPILOT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("DOCPILOT_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    # LLM provider: "ollama" | "openai" (any OpenAI-compatible endpoint) | "none" (extractive answers)
    llm_provider: str = field(default_factory=lambda: _env("DOCPILOT_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("DOCPILOT_LLM_MODEL", "qwen2.5:1.5b-instruct"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    llm_threads: int = field(default_factory=lambda: int(_env("DOCPILOT_LLM_THREADS", "4")))
    max_answer_tokens: int = field(default_factory=lambda: int(_env("DOCPILOT_MAX_ANSWER_TOKENS", "160")))
    top_k: int = field(default_factory=lambda: int(_env("DOCPILOT_TOP_K", "4")))
    candidates: int = field(default_factory=lambda: int(_env("DOCPILOT_CANDIDATES", "20")))
    cache_enabled: bool = field(default_factory=lambda: _env("DOCPILOT_CACHE", "1") == "1")

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache.sqlite"


def get_settings() -> Settings:
    return Settings()
