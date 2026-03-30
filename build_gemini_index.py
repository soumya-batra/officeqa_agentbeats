#!/usr/bin/env python3
"""Build Gemini FAISS index from existing chunks.json (no Docker needed).

Usage:
    python build_gemini_index.py

Reads chunks from faiss_index/chunks.json, embeds with Gemini, saves to faiss_index/gemini/.
"""
import json
import os
import time
import sys

import numpy as np

os.environ.setdefault("GOOGLE_API_KEY", "REDACTED")

GEMINI_MODEL = "gemini-embedding-001"
CHUNKS_PATH = "faiss_index/chunks.json"
OUTPUT_DIR = "faiss_index/gemini"
BATCH_SIZE = 100
MAX_RETRIES = 10


def main():
    from google import genai

    client = genai.Client()

    # Load chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Prepare output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for partial progress (resume support)
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings_partial.npy")
    start_idx = 0
    embeddings = []

    if os.path.exists(embeddings_path):
        partial = np.load(embeddings_path)
        start_idx = len(partial)
        embeddings = list(partial)
        print(f"Resuming from chunk {start_idx}/{len(chunks)}")

    # Embed in batches
    total_batches = (len(chunks) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for batch_num, i in enumerate(range(start_idx, len(chunks), BATCH_SIZE)):
        batch = [c["content"][:2000] for c in chunks[i:i + BATCH_SIZE]]

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.embed_content(model=GEMINI_MODEL, contents=batch)
                embeddings.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    wait = min(2 ** attempt * 2, 60)
                    print(f"  Rate limit, retrying in {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {MAX_RETRIES} retries at batch {batch_num}")

        elapsed = time.time() - t0
        rate = (batch_num + 1) / elapsed * 60 if elapsed > 0 else 0
        eta = (total_batches - batch_num - 1) / rate if rate > 0 else 0
        done = i + len(batch)
        if (batch_num + 1) % 10 == 0 or batch_num == 0:
            print(f"  Batch {batch_num+1}/{total_batches} | {done}/{len(chunks)} chunks | "
                  f"{rate:.0f} batches/min | ETA {eta:.0f}min", flush=True)

        # Save checkpoint every 100 batches
        if (batch_num + 1) % 100 == 0:
            np.save(embeddings_path, np.array(embeddings, dtype=np.float32))
            print(f"\n  Checkpoint saved at {done} chunks")

    print(f"\nEmbedding complete: {len(embeddings)} vectors in {time.time()-t0:.0f}s")

    # Build FAISS index
    import faiss

    dim = len(embeddings[0])
    matrix = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save
    index_path = os.path.join(OUTPUT_DIR, "index.faiss")
    meta_path = os.path.join(OUTPUT_DIR, "chunks.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(chunks, f)

    # Clean up partial
    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)

    print(f"Saved FAISS index ({dim}D, {len(chunks)} vectors) to {OUTPUT_DIR}/")
    print(f"Index file: {os.path.getsize(index_path) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
