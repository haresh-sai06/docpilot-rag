"""Tiny SQLite key-value cache used for (1) embeddings and (2) full responses.

Keys are SHA-256 of a namespace + normalised input, so the cache is safe to share across
processes and survives restarts. Values are JSON.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


class KVCache:
    def __init__(self, path: Path | str, enabled: bool = True):
        self.enabled = enabled
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        if enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(namespace: str, text: str) -> str:
        return hashlib.sha256(f"{namespace}\x00{normalise(text)}".encode()).hexdigest()

    def get(self, namespace: str, text: str):
        if not self.enabled:
            return None
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (self.key(namespace, text),)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(self, namespace: str, text: str, value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (self.key(namespace, text), json.dumps(value)))
            self._db.commit()

    def clear(self) -> None:
        if self.enabled:
            with self._lock:
                self._db.execute("DELETE FROM kv")
                self._db.commit()
