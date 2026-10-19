"""FastAPI backend: POST /ask (agent), GET /search (retrieval only), GET /health. Serves the React build if present."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import default_agent

app = FastAPI(title="DocPilot RAG", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Citation(BaseModel):
    n: int
    page: str
    section: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    route: str | None
    rewrites: int
    cached: bool
    latency_ms: float
    usage: dict
    timings: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return default_agent().ask(req.question)


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20), mode: str = "hybrid", rerank: bool = True):
    hits = default_agent().r.search(q, k=k, mode=mode, rerank=rerank)
    return [{"chunk_id": h.chunk.id, "page": h.chunk.page, "section": h.chunk.section, "score": round(h.score, 4),
             "sources": h.sources, "text": h.chunk.text[:500]} for h in hits]


_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
