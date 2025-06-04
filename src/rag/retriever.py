#!/usr/bin/env python3
"""
src/rag/retriever.py

Provides functions to create LangChain retrievers backed by Chroma and Qdrant vector stores
using BGE embeddings.

Functions:
    chroma_retriever: Returns a Chroma-based retriever.
    qdrant_retriever: Returns a Qdrant-based retriever.
"""

import logging
from typing import Any, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
from langchain.schema import BaseRetriever
from qdrant_client import QdrantClient

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# Embedding model identifier for BGE (BaiDu General Embeddings)
EMB_NAME: str = "BAAI/bge-base-en-v1.5"


def chroma_retriever(k: int = 6) -> BaseRetriever:
    """Create and return a Chroma-based retriever using BGE embeddings.

    This function initializes HuggingFaceEmbeddings with the BGE model, connects to
    a local Chroma collection named "startup_docs", and returns a retriever that
    can fetch the top-k similar documents.

    Args:
        k (int): The number of top similar documents to return during retrieval.

    Returns:
        BaseRetriever: Configured retriever for the "startup_docs" Chroma collection.

    Raises:
        RuntimeError: If embedding model initialization or Chroma connection fails.
    """
    logger.info("Initializing HuggingFaceEmbeddings for Chroma retriever...")
    try:
        emb: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=EMB_NAME)
    except Exception as e:
        msg: str = f"Failed to initialize HuggingFaceEmbeddings with model '{EMB_NAME}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Connecting to Chroma vector store at 'embeddings/chroma'...")
    try:
        store: Chroma = Chroma(
            collection_name="startup_docs",
            persist_directory="embeddings/chroma",
            embedding_function=emb
        )
    except Exception as e:
        msg: str = f"Failed to connect to Chroma collection 'startup_docs': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Returning Chroma retriever with top_k={k}...")
    try:
        retriever: BaseRetriever = store.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        msg: str = f"Failed to create Chroma retriever: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    return retriever


def qdrant_retriever(
        host: str = "localhost",
        port: int = 6333,
        k: int = 6,
        ssl: bool = False,
        api_key: Optional[str] = None
) -> BaseRetriever:
    """Create and return a Qdrant-based retriever using BGE embeddings.

    This function initializes HuggingFaceEmbeddings with the BGE model, connects to
    a Qdrant instance at the specified host and port, and returns a retriever for
    the "startup_docs" collection that fetches the top-k similar documents.

    Args:
        host (str): The hostname or IP address of the Qdrant server.
        port (int): The port on which Qdrant is listening.
        k (int): The number of top similar documents to return during retrieval.
        ssl (bool): Whether to use SSL/TLS when connecting to Qdrant.
        api_key (Optional[str]): Optional API key for Qdrant authentication.

    Returns:
        BaseRetriever: Configured retriever for the "startup_docs" Qdrant collection.

    Raises:
        RuntimeError: If embedding model initialization or Qdrant connection fails.
    """
    logger.info("Initializing HuggingFaceEmbeddings for Qdrant retriever...")
    try:
        emb: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=EMB_NAME)
    except Exception as e:
        msg: str = f"Failed to initialize HuggingFaceEmbeddings with model '{EMB_NAME}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Connecting to Qdrant at {host}:{port} (ssl={ssl})...")
    try:
        client: QdrantClient = QdrantClient(
            url=f"{host}:{port}", prefer_grpc=True, api_key=api_key, ssl=ssl
        )
    except Exception as e:
        msg: str = f"Failed to connect to Qdrant at {host}:{port}: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Retrieving or creating Qdrant collection 'startup_docs'...")
    try:
        store: Qdrant = Qdrant(client, "startup_docs", emb)
    except Exception as e:
        msg: str = f"Failed to initialize Qdrant store for collection 'startup_docs': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Returning Qdrant retriever with top_k={k}...")
    try:
        retriever: BaseRetriever = store.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        msg: str = f"Failed to create Qdrant retriever: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    return retriever
