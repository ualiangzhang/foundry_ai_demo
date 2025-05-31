#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import json, tqdm, pathlib

ROOT = pathlib.Path("data_processed")
DOCS = [json.loads(l) for l in open(ROOT/"rag_docs.jsonl")]
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

embs = model.encode(
    [d["text"] for d in tqdm.tqdm(DOCS, desc="embed")],
    normalize_embeddings=True,
    show_progress_bar=False)

client = PersistentClient(path="embeddings/chroma")
col = client.get_or_create_collection("startup_docs")

col.add(
    ids=[str(i) for i in range(len(DOCS))],
    embeddings=embs,
    documents=[d["text"] for d in DOCS],
    metadatas=[d.get("meta", {}) for d in DOCS])

print("✅  Chroma collection ready at embeddings/chroma/")
