from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

INDEX_PATH = Path("data/kb_index.faiss")
META_PATH = Path("data/kb_metadata.json")
SHARDS_DIR = Path("data/kb_shards")
MANIFEST_PATH = Path("data/kb_manifest.json")
META_DB_PATH = Path("data/kb_metadata.sqlite")


def _load_legacy_embeddings():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, []
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def _load_sharded_index() -> tuple[faiss.IndexShards | None, sqlite3.Connection | None, list[dict]]:
    if not MANIFEST_PATH.exists() or not META_DB_PATH.exists():
        return None, None, []
    if not SHARDS_DIR.exists():
        return None, None, []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    if not shards:
        return None, None, []
    first_index = faiss.read_index(str(SHARDS_DIR / shards[0]["file"]))
    index_shards = faiss.IndexShards(first_index.d, False, True)
    index_shards.add_shard(first_index)
    for shard in shards[1:]:
        index_shards.add_shard(faiss.read_index(str(SHARDS_DIR / shard["file"])))
    conn = sqlite3.connect(META_DB_PATH)
    return index_shards, conn, shards


class KBStore:
    def __init__(self) -> None:
        self.index, self.metadata = _load_legacy_embeddings()
        self.shard_manifest: list[dict] = []
        self.metadata_db: sqlite3.Connection | None = None
        sharded_index, metadata_db, shard_manifest = _load_sharded_index()
        if sharded_index is not None and metadata_db is not None:
            self.index = sharded_index
            self.metadata = []
            self.metadata_db = metadata_db
            self.shard_manifest = shard_manifest
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def search(self, query: str, k: int = 3) -> List[dict]:
        if not self.index or not self.metadata:
            if not self.metadata_db:
                return []
        vector = self._embed_text(query)
        scores, idxs = self.index.search(np.array([vector]).astype("float32"), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or idx >= len(self.metadata):
                if self.metadata_db:
                    row = self._fetch_row_by_id(int(idx))
                    if row:
                        row["score"] = float(score)
                        results.append(row)
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)
        return results

    def fetch_chunk(self, chunk_id: str) -> dict | None:
        if self.metadata_db:
            cursor = self.metadata_db.execute(
                "select chunk_id, source, location, text from chunks where chunk_id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "chunk_id": row[0],
                    "source": row[1],
                    "location": row[2],
                    "text": row[3],
                }
        for item in self.metadata or []:
            if item.get("chunk_id") == chunk_id:
                return item
        return None

    def _fetch_row_by_id(self, row_id: int) -> dict | None:
        if not self.metadata_db:
            return None
        cursor = self.metadata_db.execute(
            "select chunk_id, source, location, text from chunks where id = ?",
            (row_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "chunk_id": row[0],
            "source": row[1],
            "location": row[2],
            "text": row[3],
        }

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
