#!/usr/bin/env python3
"""
tests/test_rag.py

Integration test to compare retrieval and response between the base LLaMA-3 model
(no LoRA) and the LoRA-fine-tuned LLaMA-3 for a sample query. This test:

1. Builds a RetrievalQA chain using the LoRA-fine-tuned LLaMA-3 (via build_chain).
2. Extracts the retriever from that chain.
3. Builds a “base” chain without LoRA using the same retriever.
4. Retrieves the top-3 documents for a sample query and prints them once.
5. Generates responses from the base chain and the LoRA chain, printing both
   to allow manual comparison.

Usage:
    python3 tests/test_rag.py

Dependencies:
    - A local Chroma or Qdrant vector store populated with embeddings.
    - The SFT LoRA adapter must be merged and available under models/adapters/llama3_lora.
    - The base LLaMA-3 model must exist under models/base/Meta-Llama-3-8B-Instruct.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# ─── Add project root (one level up from tests/) to Python path ──────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Core imports ─────────────────────────────────────────────────────────────
from src.rag.chains import build_chain  # RetrievalQA chain with LoRA merged by default
from src.rag.model_loader import load_llama
from src.rag.prompts import PROJECT_EVAL

import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_base_chain(
    retriever: BaseRetriever,
    max_new_tokens: int = 512,
    temperature: float = 0.0
) -> RetrievalQA:
    """
    Load the base (no-LoRA) LLaMA-3 model and wrap it in a RetrievalQA chain.

    Args:
        retriever: An existing LangChain retriever (e.g., from a Chroma or Qdrant store).
        max_new_tokens: Maximum number of tokens to generate for each call.
        temperature: Sampling temperature for text generation (0.0 = deterministic).

    Returns:
        A RetrievalQA chain using the base LLaMA-3 model without LoRA.

    Raises:
        RuntimeError: If loading the base model or building the pipeline fails.
    """
    logger.info("Loading base LLaMA-3 model (no LoRA)...")
    try:
        # 1) Load base model + tokenizer without LoRA
        base_model, base_tok = load_llama(use_lora=False)
    except Exception as e:
        msg = f"Failed to load base LLaMA-3 model: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Initializing HuggingFace text-generation pipeline for base model...")
    try:
        pipe = transformers.pipeline(
            "text-generation",
            model=base_model,
            tokenizer=base_tok,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            repetition_penalty=1.1
        )
    except Exception as e:
        msg = f"Failed to create text-generation pipeline: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    base_llm = HuggingFacePipeline(pipeline=pipe)

    logger.info("Building RetrievalQA chain for base model...")
    try:
        base_chain = RetrievalQA.from_chain_type(
            llm=base_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROJECT_EVAL}
        )
    except Exception as e:
        msg = f"Failed to create RetrievalQA chain for base model: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    return base_chain


def make_lora_chain(store: str = "chroma") -> RetrievalQA:
    """
    Load the LoRA-merged LLaMA-3 and create a RetrievalQA chain exactly as in training.

    Args:
        store: Which vector store to use for retrieval; "chroma" or "qdrant".

    Returns:
        A RetrievalQA chain using the LoRA-fine-tuned LLaMA-3 model.

    Raises:
        RuntimeError: If building the LoRA chain fails.
    """
    logger.info(f"Building LoRA-fine-tuned chain using '{store}' store...")
    try:
        lora_chain = build_chain(kind="eval", store=store)
    except Exception as e:
        msg = f"Failed to build LoRA chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)
    return lora_chain


def main() -> None:
    """
    Main execution flow for comparing base vs. LoRA-fine-tuned chains.

    Steps:
        1) Build the LoRA-fine-tuned chain and extract its retriever.
        2) Build the base chain using the same retriever.
        3) Retrieve the top-3 documents for a sample query and print them.
        4) Generate and print responses from both chains for manual comparison.
    """
    # Sample query to test both chains
    query: str = (
        "Our startup produces mushroom-based leather. "
        "Could you critique our go-to-market plan?"
    )

    # 1) Build the LoRA-fine-tuned chain
    try:
        lora_chain: RetrievalQA = make_lora_chain(store="chroma")
    except RuntimeError:
        logger.error("Exiting due to failure in creating LoRA chain.")
        sys.exit(1)

    # 2) Extract the retriever from the LoRA chain
    retriever: BaseRetriever = lora_chain.retriever

    # 3) Build the "base" chain (no LoRA) with the same retriever
    try:
        base_chain: RetrievalQA = make_base_chain(retriever)
    except RuntimeError:
        logger.error("Exiting due to failure in creating base chain.")
        sys.exit(1)

    # 4) Retrieve the top-3 documents (identical for both models)
    try:
        top_docs = retriever.get_relevant_documents(query)[:3]
    except Exception as e:
        logger.error(f"Failed to retrieve documents for query: {e}")
        sys.exit(1)

    # 5) Print the top-3 retrieved snippets once
    print("\n" + "=" * 80)
    print("STEP 1: Top-3 retrieved snippets (shared by both models)")
    print("=" * 80)
    for i, doc in enumerate(top_docs, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"Snippet {i}: {snippet}\n")

    # 6) Generate and print the base-model response
    print("\n" + "=" * 80)
    print("STEP 2: Base LLaMA-3 (no LoRA) response")
    print("=" * 80)
    try:
        base_result: Dict[str, Any] = base_chain({"query": query})
        print(base_result.get("result", "").strip())
    except Exception as e:
        logger.error(f"Failed to generate response from base chain: {e}")
        sys.exit(1)

    # 7) Generate and print the LoRA-fine-tuned response
    print("\n" + "=" * 80)
    print("STEP 3: LoRA-Fine-Tuned LLaMA-3 response")
    print("=" * 80)
    try:
        lora_result: Dict[str, Any] = lora_chain({"query": query})
        print(lora_result.get("result", "").strip())
    except Exception as e:
        logger.error(f"Failed to generate response from LoRA chain: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Comparison complete.\n")


if __name__ == "__main__":
    main()