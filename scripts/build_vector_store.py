#!/usr/bin/env python3
"""
scripts/build_vector_store.py

Reads data_processed/rag_docs.jsonl, computes embeddings with BGE,
and persists them into a local Chroma collection at embeddings/chroma/.
Splits the add() call into chunks of <= 5000 to avoid Chroma's max-batch limitation.
"""

import json
import tqdm
import pathlib
import torch

# Compatibility patch for sentence-transformers requiring torch.get_default_device()
if not hasattr(torch, "get_default_device"):
    def get_default_device() -> torch.device:
        """
        Return the default device for PyTorch:
        - 'cuda' if a CUDA-capable GPU is available
        - otherwise 'cpu'
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ─── 1. Paths ─────────────────────────────────────────────────────────────────
ROOT = pathlib.Path("data_processed")
INPUT_FILE = ROOT / "rag_docs.jsonl"
OUT_DIR = pathlib.Path("embeddings/chroma")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 2. Load your documents ──────────────────────────────────────────────────
docs = []
metadatas = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # each JSON line is expected to have a "text" field; adjust if yours differs
        text = obj.get("text", "")
        docs.append(text)
        # optional: store any metadata dictionary under "meta" in your JSON lines
        metadatas.append(obj.get("meta", {}))

print(f"Loaded {len(docs)} documents from {INPUT_FILE}")

# ─── 3. Initialize embedding model ────────────────────────────────────────────
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
print(f"Loading embedding model '{EMB_MODEL_NAME}'…")
embedder = SentenceTransformer(EMB_MODEL_NAME)

# ─── 4. Compute embeddings ─────────────────────────────────────────────────────
batch_size = 64  # you can tune this up/down based on your GPU/CPU memory
embeddings = []

for i in tqdm.tqdm(range(0, len(docs), batch_size), desc="Embedding"):
    batch_texts = docs[i : i + batch_size]
    batch_emb = embedder.encode(
        batch_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    embeddings.extend(batch_emb)

assert len(embeddings) == len(docs), "Number of embeddings must match number of docs"

# ─── 5. Connect to Chroma and get/create a collection ────────────────────────
print(f"Connecting to Chroma at {OUT_DIR}…")
client = PersistentClient(path=str(OUT_DIR))
collection = client.get_or_create_collection(name="startup_docs")

# ─── 6. Insert in chunks to respect max-batch-size (≈ 5461) ───────────────────
max_chunk = 5000
ids = [str(i) for i in range(len(docs))]

print(f"Adding {len(docs)} embeddings into Chroma collection in chunks of {max_chunk}…")
for start in tqdm.tqdm(range(0, len(docs), max_chunk), desc="Adding chunks"):
    end = start + max_chunk
    chunk_ids = ids[start:end]
    chunk_docs = docs[start:end]
    chunk_embs = embeddings[start:end]
    chunk_meta = metadatas[start:end]

    collection.add(
        ids=chunk_ids,
        embeddings=chunk_embs,
        documents=chunk_docs,
        metadatas=chunk_meta
    )

print(f"✓  Chroma store ready at {OUT_DIR}")
