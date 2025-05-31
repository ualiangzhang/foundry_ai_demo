# src/rag/retriever.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from qdrant_client import QdrantClient

EMB_NAME = "BAAI/bge-base-en-v1.5"

def chroma_retriever(k: int = 6):
    emb = HuggingFaceEmbeddings(model_name=EMB_NAME)
    store = Chroma(
        collection_name="startup_docs",
        persist_directory="embeddings/chroma",
        embedding_function=emb
    )
    return store.as_retriever(search_kwargs={"k": k})

def qdrant_retriever(host="localhost", port=6333, k: int = 6):
    emb = HuggingFaceEmbeddings(model_name=EMB_NAME)
    client = QdrantClient(host=host, port=port)
    store = Qdrant(client, "startup_docs", emb)
    return store.as_retriever(search_kwargs={"k": k})
