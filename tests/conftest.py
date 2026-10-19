import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docpilot.cache import KVCache  # noqa: E402
from docpilot.corpus import Chunk  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder so tests need no model downloads."""

    def __init__(self, cache=None, dim=64):
        self.dim, self.cache, self.name = dim, cache or KVCache(":memory:", enabled=False), "fake"

    def _vec(self, text):
        v = np.zeros(self.dim)
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_query(self, text):
        return self._vec(text).tolist()

    def embed_docs(self, texts):
        return np.array([self._vec(t) for t in texts])


@pytest.fixture()
def chunks():
    texts = {
        "tutorial/cors.md": "CORS middleware: use CORSMiddleware with allow_origins to allow cross-origin requests from browsers.",
        "tutorial/path-params.md": "Path parameters are declared in the path string like /items/{item_id} and typed with annotations.",
        "tutorial/query-params.md": "Query parameters are function parameters not part of the path; they can have default values.",
        "tutorial/security/oauth2-jwt.md": "OAuth2 with Password and JWT tokens: hash passwords and create access tokens with an expiry.",
        "deployment/docker.md": "Build a Docker image for FastAPI using the official python image and run uvicorn in the container.",
    }
    return [Chunk(id=f"{p}#0", page=p, title=p, section="s", text=t) for p, t in texts.items()]
