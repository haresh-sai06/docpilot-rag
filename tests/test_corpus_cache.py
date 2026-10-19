from docpilot.cache import KVCache, normalise
from docpilot.corpus import chunk_page, clean

MD = """# Tutorial

Intro paragraph about FastAPI that is long enough to be kept as a chunk on its own for sure.

## Install

Run pip install fastapi to install the framework together with its standard optional dependencies.

```python
def a():

    return 1
```

## Run

Use uvicorn main:app --reload to start the development server with auto-reload enabled.
"""


def test_chunker_keeps_headings_and_code_blocks():
    ch = chunk_page("tutorial/index.md", MD, max_chars=1200)
    assert [c.section for c in ch] == ["Tutorial", "Install", "Run"]
    install = [c for c in ch if c.section == "Install"]
    assert install and "def a():\n\n    return 1" in install[0].text  # code block not split
    assert all(c.text.startswith("Tutorial > ") for c in ch)


def test_clean_strips_includes():
    assert "{*" not in clean("x\n{* ../../docs_src/a.py hl[1] *}\ny")


def test_cache_roundtrip(tmp_path):
    c = KVCache(tmp_path / "c.sqlite")
    assert c.get("ns", "Hello  World") is None
    c.set("ns", "hello world", {"a": 1})
    assert c.get("ns", "HELLO world") == {"a": 1}  # normalised key
    assert (c.hits, c.misses) == (1, 1)
    assert normalise(" A  b ") == "a b"
    off = KVCache(tmp_path / "off.sqlite", enabled=False)
    off.set("ns", "x", 1)
    assert off.get("ns", "x") is None
