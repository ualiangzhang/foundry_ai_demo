#!/usr/bin/env python3
"""
tests/test_rag.py

Compares retrieval + answers from the base LLaMA-3 (no LoRA) vs. the LoRA-
fine-tuned model for a single query.

Steps
-----
1. Build LoRA chain with build_chain(kind="rag").
2. Re-use its retriever to build the base (no-LoRA) chain.
3. Show the top-3 retrieved docs once.
4. Print responses from each chain for manual inspection.

Run:
    python -m tests.test_rag
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# ── Add project root to import path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core imports ─────────────────────────────────────────────────────────────
from src.rag.chains import build_chain  # LoRA chain factory
from src.rag.model_loader import load_llama  # base model loader
from src.rag.prompts import RAG_WRAPPER  # **same prompt for both chains**

import transformers
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def make_base_chain(
        retriever: BaseRetriever,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
) -> RetrievalQA:
    """Return a RetrievalQA chain that uses the *base* (no-LoRA) weights."""
    logger.info("Loading base LLaMA-3 model (no LoRA)…")
    base_model, base_tok = load_llama(use_lora=False)

    pipe = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=base_tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.1,
    )
    base_llm = HuggingFacePipeline(pipeline=pipe)

    # we rebuild the *eval* chain logic here: retrieve snippet + context → 4 bullets
    eval_chain = build_chain(kind="eval", store="chroma")
    # Monkey-patch its llm to the base weights and keep same retriever
    eval_chain.__closure__[0].cell_contents["llm"] = base_llm  # type: ignore
    eval_chain.__closure__[0].cell_contents["retriever"] = retriever  # type: ignore
    return eval_chain


def make_lora_chain(store: str = "chroma") -> RetrievalQA:
    """Return the LoRA-merged RetrievalQA chain (kind='rag')."""
    logger.info(f"Building LoRA chain (store='{store}')…")
    return build_chain(kind="eval", store=store)  # already a RetrievalQA


# --------------------------------------------------------------------------- #
# Main test                                                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    query = (
        "Develop a sustainable leather alternative by cultivating and processing mycelium-based materials into "
        "high-quality, biodegradable leather substitutes for fashion and upholstery, reducing reliance on animal hides "
        "and minimizing environmental impact."
    )

    # 1. LoRA chain
    lora_chain: RetrievalQA = make_lora_chain(store="chroma")
    retriever: BaseRetriever = lora_chain.retriever

    # 2. Base chain (same retriever, same prompt)
    base_chain: RetrievalQA = make_base_chain(retriever)

    # 3. Show top-3 retrieved snippets
    top_docs = retriever.get_relevant_documents(query)
    print("\n" + "=" * 80)
    print("STEP 1  ·  Top-3 retrieved snippets")
    print("=" * 80)
    for idx, doc in enumerate(top_docs, 1):
        print(f"Snippet {idx}: {doc.page_content.strip().replace(chr(10), ' ')}\n")

    # 4. Base model answer
    print("\n" + "=" * 80)
    print("STEP 2  ·  Base LLaMA-3 (no LoRA)")
    print("=" * 80)
    base_result: Dict[str, Any] = base_chain({"query": query})
    print(base_result.get("result", "").strip())

    # 5. LoRA model answer
    print("\n" + "=" * 80)
    print("STEP 3  ·  LoRA-Fine-Tuned LLaMA-3")
    print("=" * 80)
    lora_result: Dict[str, Any] = lora_chain({"query": query})
    print(lora_result.get("result", "").strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")


if __name__ == "__main__":
    main()
