"""Download corpus + models and build the on-disk Qdrant index (used at Docker build time)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("DOCPILOT_QDRANT_PATH", "data/qdrant")
from docpilot.service import build  # noqa: E402

agent = build()
print("indexed", len(agent.r.chunks), "chunks;", agent.r.search("How do I add CORS?", k=1)[0].chunk.page)
