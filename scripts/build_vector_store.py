#!/usr/bin/env python3
"""
scripts/build_vector_store.py

This script reads startup documents from a JSONL file (data_processed/rag_docs.jsonl),
computes embeddings for each document using the BGE model, and stores them in a
persistent Chroma collection at embeddings/chroma/.

Key Steps:
1. Load texts and metadata from rag_docs.jsonl.
2. Initialize the SentenceTransformer embedding model (BGE).
3. Compute embeddings in batches to leverage GPU/CPU resources.
4. Connect to a local Chroma instance and create or retrieve the "startup_docs" collection.
5. Insert embeddings, documents, and metadata into Chroma in chunks to respect Chroma's
   maximum batch size limitation (~5000).

Usage:
    python scripts/build_vector_store.py

Ensure:
    - The input file 'data_processed/rag_docs.jsonl' exists and is properly formatted.
    - The directory 'embeddings/chroma/' is writable (will be created if not).
    - The BGE model ('BAAI/bge-base-en-v1.5') is accessible via Hugging Face.
    - sentence-transformers, torch, and chromadb libraries are installed.

Output:
    - A local Chroma collection stored under 'embeddings/chroma/' containing:
        • ids: string identifiers for each document
        • embeddings: high-dimensional vectors
        • documents: original text
        • metadatas: associated metadata per document

Example:
    mkdir -p embeddings/chroma
    # Ensure rag_docs.jsonl is present in data_processed/
    python scripts/build_vector_store.py
"""

import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.api import Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Compatibility patch for sentence-transformers requiring torch.get_default_device()
if not hasattr(torch, "get_default_device"):
    def get_default_device() -> torch.device:
        """Return the default PyTorch device ('cuda' if available, else 'cpu')."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    torch.get_default_device = get_default_device


def load_documents(input_path: pathlib.Path) -> Tuple[List[str], List[Dict]]:
    """Load documents and metadata from a JSONL file.

    Args:
        input_path: Path to the JSONL file containing startup documents.

    Returns:
        A tuple of:
            - docs: List of document texts.
            - metadatas: List of metadata dicts corresponding to each document.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
        IOError: If reading the file fails.
    """
    if not input_path.exists():
        msg = f"Input file '{input_path}' not found."
        logger.error(msg)
        raise FileNotFoundError(msg)

    docs: List[str] = []
    metadatas: List[Dict] = []
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_number}: {e}")
                    continue

                text = obj.get("text", "")
                metadata = obj.get("meta", {})
                docs.append(text)
                metadatas.append(metadata)
    except (OSError, IOError) as e:
        logger.error(f"Error reading from '{input_path}': {e}")
        raise

    logger.info(f"Loaded {len(docs)} documents from {input_path}")
    return docs, metadatas


def compute_embeddings(
        docs: List[str],
        model_name: str,
        batch_size: int
) -> List[List[float]]:
    """Compute normalized embeddings for a list of documents using SentenceTransformer.

    Args:
        docs: List of document texts to embed.
        model_name: Hugging Face model name or path for SentenceTransformer.
        batch_size: Number of documents to process per batch.

    Returns:
        A list of embedding vectors (one list of floats per document).

    Raises:
        OSError: If loading the embedding model fails.
        RuntimeError: If embedding computation fails.
    """
    try:
        logger.info(f"Loading embedding model '{model_name}'…")
        embedder = SentenceTransformer(model_name)
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise

    embeddings: List[List[float]] = []
    total_docs = len(docs)
    device = torch.get_default_device()
    logger.info(f"Using device: {device}")

    for i in tqdm.tqdm(range(0, total_docs, batch_size), desc="Computing Embeddings"):
        batch_texts = docs[i: i + batch_size]
        try:
            batch_emb = embedder.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=device
            )
        except RuntimeError as e:
            logger.error(f"Embedding failed for batch starting at index {i}: {e}")
            raise
        embeddings.extend(batch_emb)

    assert len(embeddings) == total_docs, "Mismatch between docs and embeddings count"
    logger.info("Embedding computation complete.")
    return embeddings


def get_chroma_collection(
        db_path: pathlib.Path,
        collection_name: str
) -> Collection:
    """Connect to a persistent Chroma database and get or create a collection.

    Args:
        db_path: Filesystem path where Chroma data is stored.
        collection_name: Name of the collection to retrieve or create.

    Returns:
        A Chroma Collection object.

    Raises:
        RuntimeError: If connecting to Chroma or creating the collection fails.
    """
    try:
        logger.info(f"Connecting to Chroma at '{db_path}'…")
        client = PersistentClient(path=str(db_path))
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        msg = f"Failed to connect to Chroma or get/create collection '{collection_name}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info(f"Connected to collection '{collection_name}'.")
    return collection


def insert_embeddings_to_chroma(
        collection: Collection,
        docs: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        max_chunk_size: int = 5000
) -> int:
    """Insert documents, embeddings, and metadata into a Chroma collection in chunks.

    Args:
        collection: Chroma Collection object to insert into.
        docs: List of document texts.
        embeddings: List of embedding vectors.
        metadatas: List of metadata dictionaries.
        max_chunk_size: Maximum number of items to insert in one batch.

    Returns:
        Total number of items inserted.

    Raises:
        ValueError: If the lengths of docs, embeddings, and metadatas do not match.
        RuntimeError: If insertion into Chroma fails.
    """
    total = len(docs)
    if not (len(docs) == len(embeddings) == len(metadatas)):
        msg = "Lengths of docs, embeddings, and metadatas must match."
        logger.error(msg)
        raise ValueError(msg)

    # Generate string IDs for each document
    ids: List[str] = [str(i) for i in range(total)]
    inserted_count = 0

    logger.info(f"Adding {total} items to Chroma in chunks of {max_chunk_size}…")
    try:
        for start in tqdm.tqdm(range(0, total, max_chunk_size), desc="Inserting into Chroma"):
            end = min(start + max_chunk_size, total)
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
            inserted_count += len(chunk_ids)
    except Exception as e:
        msg = f"Failed to insert embeddings into Chroma: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Insertion complete: {inserted_count}/{total} items added.")
    return inserted_count


def main() -> None:
    """Main function orchestrating the vector store build process."""
    # Define paths
    root = pathlib.Path("data_processed")
    input_file = root / "rag_docs.jsonl"
    chroma_dir = pathlib.Path("embeddings/chroma")

    # Load documents
    try:
        docs, metadatas = load_documents(input_file)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # Compute embeddings
    model_name = "BAAI/bge-base-en-v1.5"
    batch_size = 64
    try:
        embeddings = compute_embeddings(docs, model_name, batch_size)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # Ensure Chroma directory exists
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Connect to Chroma and get collection
    collection_name = "startup_docs"
    try:
        collection = get_chroma_collection(chroma_dir, collection_name)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # Insert embeddings into Chroma
    try:
        insert_embeddings_to_chroma(collection, docs, embeddings, metadatas)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    logger.info(f"✓  Chroma store ready at '{chroma_dir}'")


if __name__ == "__main__":
    main()
