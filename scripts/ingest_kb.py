from __future__ import annotations

import argparse
import json

import os

import sqlite3
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader

DATA_DIR = Path("data")
SOURCE_DIR = Path("sources")
HEARING_PACK = Path("hearing_pack.md")
SHARDS_DIR = DATA_DIR / "kb_shards"
MANIFEST_PATH = DATA_DIR / "kb_manifest.json"
META_DB_PATH = DATA_DIR / "kb_metadata.sqlite"
EMBED_DIM = 1536


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> Iterable[str]:
    words = text.split()
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        yield " ".join(words[start:end])
        start = end - overlap
        if start < 0:
            start = 0


def embed_texts(texts: List[str]) -> np.ndarray:
    try:
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [item.embedding for item in resp.data]
    except Exception:
        vectors = [np.random.default_rng(i).random(EMBED_DIM).tolist() for i, _ in enumerate(texts)]
    return np.array(vectors).astype("float32")


def estimate_text_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(page_count, int):
                return page_size * page_count
        except (ValueError, OSError):
            return None
    return None


def iter_markdown_chunks(chunk_size: int) -> Iterable[dict]:
    if not HEARING_PACK.exists():
        return
    text = HEARING_PACK.read_text(encoding="utf-8")
    for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size)):
        yield {
            "chunk_id": f"md-{idx}",
            "source": HEARING_PACK.name,
            "location": f"section {idx+1}",
            "text": chunk,
        }


def iter_pdf_chunks(chunk_size: int) -> Iterable[dict]:
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            content = page.extract_text() or ""
            for idx, chunk in enumerate(chunk_text(content, chunk_size=chunk_size)):
                yield {
                    "chunk_id": f"{pdf_path.stem}-p{i+1}-{idx}",
                    "source": pdf_path.name,
                    "location": f"page {i+1}",
                    "text": chunk,
                }


def iter_all_chunks(chunk_size: int) -> Iterable[dict]:
    yield from iter_markdown_chunks(chunk_size)
    yield from iter_pdf_chunks(chunk_size)


def init_metadata_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(META_DB_PATH)
    with conn:
        conn.execute(
            """
            create table if not exists chunks (
                id integer primary key,
                chunk_id text,
                source text,
                location text,
                text text
            )
            """
        )
        conn.execute("delete from chunks")
    return conn


def write_manifest(entries: List[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump({"shards": entries}, f, indent=2)


def create_index() -> faiss.IndexIDMap2:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))


def flush_shard(index: faiss.IndexIDMap2, shard_id: int) -> dict | None:
    if index.ntotal == 0:
        return None
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    shard_file = f"kb_shard_{shard_id}.faiss"
    faiss.write_index(index, str(SHARDS_DIR / shard_file))
    return {"file": shard_file, "size": int(index.ntotal)}


def build_index(chunk_size: int, batch_size: int, shard_size: int):
    index = create_index()
    batch_texts: List[str] = []
    batch_ids: List[int] = []
    total_chunks = 0
    shard_chunks = 0
    shard_id = 0
    manifest_entries: List[dict] = []
    conn = init_metadata_db()
    for chunk in iter_all_chunks(chunk_size):
        total_chunks += 1
        cursor = conn.execute(
            "insert into chunks (chunk_id, source, location, text) values (?, ?, ?, ?)",
            (chunk["chunk_id"], chunk["source"], chunk["location"], chunk["text"]),
        )
        chunk_row_id = cursor.lastrowid
        batch_texts.append(chunk["text"])
        batch_ids.append(chunk_row_id)
        shard_chunks += 1
        if len(batch_texts) >= batch_size:
            vectors = embed_texts(batch_texts)
            faiss.normalize_L2(vectors)
            index.add_with_ids(vectors, np.array(batch_ids, dtype="int64"))
            batch_texts = []
            batch_ids = []

        if shard_chunks >= shard_size:
            if batch_texts:
                vectors = embed_texts(batch_texts)
                faiss.normalize_L2(vectors)
                index.add_with_ids(vectors, np.array(batch_ids, dtype="int64"))
                batch_texts = []
                batch_ids = []
            entry = flush_shard(index, shard_id)
            if entry:
                manifest_entries.append(entry)
            shard_id += 1
            shard_chunks = 0
            index = create_index()

    if batch_texts:
        vectors = embed_texts(batch_texts)
        faiss.normalize_L2(vectors)
        index.add_with_ids(vectors, np.array(batch_ids, dtype="int64"))

    if total_chunks == 0:
        print("No chunks found; nothing to index.")
        return

    entry = flush_shard(index, shard_id)
    if entry:
        manifest_entries.append(entry)
    write_manifest(manifest_entries)
    conn.commit()
    conn.close()
    print(f"Wrote {total_chunks} chunks to {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Ingest hearing pack and sources into FAISS index")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2000)
    args = parser.parse_args()
    build_index(chunk_size=args.chunk_size, batch_size=args.batch_size, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
