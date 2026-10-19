"""Download the (MIT-licensed) FastAPI docs at a pinned version and split them into chunks.

Chunking: split on markdown headings (##, ###), then pack paragraphs into <= max_chars chunks
with one-paragraph overlap. Code blocks are never split. Each chunk keeps its page + heading path.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

TARBALL = "https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{version}"


@dataclass
class Chunk:
    id: str
    page: str
    title: str
    section: str
    text: str

    def to_dict(self):
        return asdict(self)


def download_docs(version: str, out_dir: Path) -> Path:
    out = out_dir / f"fastapi-docs-{version}"
    if out.exists():
        return out
    r = requests.get(TARBALL.format(version=version), timeout=120)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("docs/en/docs/") and m.name.endswith(".md") and m.isfile():
                rel = parts[1][len("docs/en/docs/"):]
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(m).read())
    return out


_INCLUDE = re.compile(r"^\{\*.*\*\}$|^\{!.*!\}$", re.M)  # mkdocs code-include directives
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.M)


def clean(md: str) -> str:
    md = _INCLUDE.sub("", md)
    md = re.sub(r"<[^>\n]+>", "", md)  # inline html tags
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _paragraphs(text: str) -> list[str]:
    out, buf, in_code = [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                out.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def chunk_page(page: str, md: str, max_chars: int = 1200) -> list[Chunk]:
    md = clean(md)
    m = re.search(r"^#\s+(.*)$", md, re.M)
    title = m.group(1).strip() if m else page
    sections, last, heading = [], 0, title
    for h in _HEADING.finditer(md):
        sections.append((heading, md[last:h.start()]))
        heading, last = h.group(2).strip(), h.end()
    sections.append((heading, md[last:]))
    chunks = []
    for section, body in sections:
        paras = [p for p in _paragraphs(body) if p.strip()]
        cur: list[str] = []
        for p in paras:
            if cur and sum(len(x) for x in cur) + len(p) > max_chars:
                chunks.append((section, "\n\n".join(cur)))
                cur = [cur[-1]] if len(cur[-1]) < max_chars // 3 else []
            cur.append(p)
        if cur and sum(len(x) for x in cur) > 80:
            chunks.append((section, "\n\n".join(cur)))
    return [Chunk(id=f"{page}#{i}", page=page, title=title, section=s, text=f"{title} > {s}\n\n{t}")
            for i, (s, t) in enumerate(chunks)]


def build_chunks(docs_dir: Path, max_chars: int = 1200) -> list[Chunk]:
    chunks = []
    for f in sorted(docs_dir.rglob("*.md")):
        if f.name in {"release-notes.md"} or "/translation" in str(f):  # changelog is noise for Q&A
            continue
        chunks.extend(chunk_page(str(f.relative_to(docs_dir)), f.read_text(encoding="utf-8")))
    return chunks


def load_chunks(data_dir: Path, version: str) -> list[Chunk]:
    path = data_dir / f"chunks-{version}.jsonl"
    if not path.exists():
        docs = download_docs(version, data_dir)
        chunks = build_chunks(docs)
        path.write_text("\n".join(json.dumps(c.to_dict()) for c in chunks))
    return [Chunk(**json.loads(line)) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    from .config import get_settings

    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    ch = load_chunks(s.data_dir, s.corpus_version)
    print(f"{len(ch)} chunks from {len({c.page for c in ch})} pages; avg {sum(len(c.text) for c in ch) / len(ch):.0f} chars")
