from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

INDEX_PATH = Path("data/kb_index.faiss")
META_PATH = Path("data/kb_metadata.json")


def _load_embeddings():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, []
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


class KBStore:
    def __init__(self) -> None:
        self.index, self.metadata = _load_embeddings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def search(self, query: str, k: int = 3) -> List[dict]:
        if not self.index or not self.metadata:
            return []
        vector = self._embed_text(query)
        scores, idxs = self.index.search(np.array([vector]).astype("float32"), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            item["score"] = float(score)
            results.append(item)
        return results

    def fetch_chunk(self, chunk_id: str) -> dict | None:
        for item in self.metadata or []:
            if item.get("chunk_id") == chunk_id:
                return item
        return None

    def _embed_text(self, text: str) -> np.ndarray:
        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return np.array(resp.data[0].embedding, dtype="float32")
        except Exception:
            # deterministic fallback to avoid crashes during offline use
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.random(1536, dtype="float32")
