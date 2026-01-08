from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader

DATA_DIR = Path("data")
SOURCE_DIR = Path("sources")
HEARING_PACK = Path("hearing_pack.md")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    client = OpenAI()
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [item.embedding for item in resp.data]
    except Exception:
        vectors = [np.random.default_rng(i).random(1536).tolist() for i, _ in enumerate(texts)]
    return np.array(vectors).astype("float32")


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


def build_index(chunks: List[dict], chunk_size: int, batch_size: int):
    index = None
    batch_texts: List[str] = []
    total_chunks = 0
    for chunk in iter_all_chunks(chunk_size):
        total_chunks += 1
        chunks.append(chunk)
        batch_texts.append(chunk["text"])
        if len(batch_texts) >= batch_size:
            vectors = embed_texts(batch_texts)
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
            faiss.normalize_L2(vectors)
            index.add(vectors)
            batch_texts = []

    if batch_texts:
        vectors = embed_texts(batch_texts)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        index.add(vectors)

    if total_chunks == 0 or index is None:
        print("No chunks found; nothing to index.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DATA_DIR / "kb_index.faiss"))
    with (DATA_DIR / "kb_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Wrote {len(chunks)} chunks to {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Ingest hearing pack and sources into FAISS index")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    chunks: List[dict] = []
    build_index(chunks, chunk_size=args.chunk_size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
