#!/usr/bin/env python3
"""
tests/test_rag.py

Integration tests for the RAG-based LLaMA-3 chains. This script:

1. Adds the project root to sys.path so that src modules can be imported.
2. Defines helper functions to build:
   - A "base" RetrievalQA chain using the unmodified LLaMA-3.
   - A "LoRA" RetrievalQA chain that loads the LoRA-fine-tuned LLaMA-3.
3. Compares both chains by:
   a. Retrieving the top-3 semantically similar documents for a sample query.
   b. Generating responses with and without LoRA.
4. Prints retrieved snippets and both model responses side by side.

Usage:
    python3 tests/test_rag.py

Ensure:
    - The "data_processed" and "embeddings/chroma" directories are populated.
    - The LoRA adapter exists at 'models/adapters/llama3_lora'.
    - All dependencies (transformers, langchain, qdrant-client) are installed.

Note:
    This is not a unit test—it's an integration check. It prints output
    to stdout rather than asserting exact values.
"""

import logging
import sys
from pathlib import Path
from typing import Any, List, Dict, Union

import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# ─── Add project root (one level up from tests/) to Python path ──────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Core imports from src ────────────────────────────────────────────────────
from src.rag.chains import build_chain
from src.rag.model_loader import load_llama
from src.rag.prompts import PROJECT_EVAL

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_base_chain(
    retriever: Any,
    max_new_tokens: int = 512
) -> RetrievalQA:
    """
    Load the base (no-LoRA) LLaMA-3 model and wrap it in a RetrievalQA chain.

    Args:
        retriever: A LangChain Retriever (e.g., from Chroma or Qdrant).
        max_new_tokens: Maximum tokens to generate per call (default: 512).

    Returns:
        A RetrievalQA chain that uses the base LLaMA-3 model.

    Raises:
        RuntimeError: If model or tokenizer loading fails.
    """
    logger.info("Loading base (no-LoRA) LLaMA-3 model and tokenizer...")
    try:
        base_model, base_tokenizer = load_llama(use_lora=False)
    except Exception as e:
        msg = f"Failed to load base LLaMA-3 model/tokenizer: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Initializing HuggingFace text-generation pipeline (greedy)...")
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=base_model,
            tokenizer=base_tokenizer,
            max_new_tokens=max_new_tokens,
            # Removed temperature and do_sample to avoid invalid flags
        )
    except Exception as e:
        msg = f"Failed to create HF pipeline for base model: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    base_llm = HuggingFacePipeline(pipeline=pipeline)

    logger.info("Building RetrievalQA chain with PROJECT_EVAL prompt (base model)...")
    try:
        base_chain = RetrievalQA.from_chain_type(
            llm=base_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROJECT_EVAL},
        )
    except Exception as e:
        msg = f"Failed to build base RetrievalQA chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    return base_chain


def make_lora_chain(store: str = "chroma") -> RetrievalQA:
    """
    Load the LoRA-merged LLaMA-3 and create a RetrievalQA chain exactly as in training.

    Args:
        store: Which vector store to use ("chroma" or "qdrant").

    Returns:
        A RetrievalQA chain configured with LoRA-fine-tuned LLaMA-3.

    Raises:
        RuntimeError: If chain creation fails.
    """
    logger.info(f"Building LoRA-fine-tuned chain using '{store}' store...")
    try:
        lora_chain = build_chain(kind="eval", store=store)
    except Exception as e:
        msg = f"Failed to build LoRA RetrievalQA chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    return lora_chain


def retrieve_top_k(
    retriever: Any,
    query: str,
    k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k documents for a given query from the retriever.

    Args:
        retriever: A LangChain retriever (as returned by chain.retriever).
        query: The textual query to retrieve against.
        k: Number of top documents to return (default: 3).

    Returns:
        A list of dictionaries, each representing a retrieved document.

    Raises:
        RuntimeError: If the retriever invocation fails.
    """
    logger.info(f"Retrieving top {k} documents for query: {query}")
    try:
        docs = retriever.invoke({"query": query, "k": k})
        return docs
    except Exception as e:
        msg = f"Retriever invocation failed: {e}"
        logger.error(msg)
        raise RuntimeError(msg)


def format_snippet(doc: Union[Dict[str, Any], Any]) -> str:
    """
    Extract and format the snippet text from a retrieved doc.

    Args:
        doc: A dictionary or Document object representing a retrieved document.

    Returns:
        A clean string snippet for printing.
    """
    # Some retrievers return a Document object with page_content; others return a dict
    if hasattr(doc, "page_content"):
        content = doc.page_content
    elif isinstance(doc, dict):
        # Chroma/Qdrant retrievers often return {'id':..., 'text':..., 'metadata':...}
        content = doc.get("page_content") or doc.get("text") or ""
    else:
        content = str(doc)

    return content.strip().replace("\n", " ")


if __name__ == "__main__":
    # ─── Sample query for testing retrieval and generation ───────────────────
    query: str = (
        "Our startup produces mushroom-based leather. "
        "Could you critique our go-to-market plan?"
    )

    # ─── 1) Build the LoRA-fine-tuned chain ───────────────────────────────────
    try:
        lora_chain: RetrievalQA = make_lora_chain(store="chroma")
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # ─── 2) Extract the retriever from the LoRA chain ────────────────────────
    retriever = lora_chain.retriever

    # ─── 3) Build the "base" chain using the same retriever but no LoRA ──────
    try:
        base_chain: RetrievalQA = make_base_chain(retriever)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # ─── 4) Retrieve the top-3 docs (identical for both models) ──────────────
    try:
        top_docs = retrieve_top_k(retriever, query, k=3)
    except Exception as e:
        logger.error(f"Aborting: {e}")
        sys.exit(1)

    # ─── 5) Print the top-3 retrieved snippets once ──────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Top-3 retrieved snippets (shared by both models)")
    print("=" * 80)
    for i, doc in enumerate(top_docs, start=1):
        snippet = format_snippet(doc)
        print(f"Snippet {i}: {snippet}\n")

    # ─── 6) Generate and print the base-model response ───────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: Base LLaMA-3 (no LoRA) response")
    print("=" * 80)
    try:
        base_result: Dict[str, Any] = base_chain.invoke({"query": query})
        print(base_result.get("result", "").strip())
    except Exception as e:
        logger.error(f"Base chain invocation failed: {e}")
        sys.exit(1)

    # ─── 7) Generate and print the LoRA-fine-tuned response ──────────────────
    print("\n" + "=" * 80)
    print("STEP 3: LoRA-Fine-Tuned LLaMA-3 response")
    print("=" * 80)
    try:
        lora_result: Dict[str, Any] = lora_chain.invoke({"query": query})
        print(lora_result.get("result", "").strip())
    except Exception as e:
        logger.error(f"LoRA chain invocation failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Comparison complete.\n")