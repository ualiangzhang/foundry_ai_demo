#!/usr/bin/env python3
"""
tests/test_rag.py

Compare the base LLaMA-3 (no-LoRA) with the LoRA-fine-tuned model on one query.
Uses:

• build_chain(kind="rag")  → RetrievalQA (for .retriever)
• build_chain(kind="eval") → callable   (returns {"result": …})
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Callable

import transformers
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_community.llms import HuggingFacePipeline

# ── path --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── local imports -----------------------------------------------------------
from src.rag.chains import build_chain
from src.rag.model_loader import load_llama

# ── logger ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _patch_eval_callable(
    eval_fn: Callable[[Dict[str, str]], Dict[str, Any]],
    *,
    new_llm: HuggingFacePipeline,
    new_retriever: BaseRetriever,
) -> None:
    """Replace captured llm / retriever inside the eval closure."""
    mapping = dict(zip(eval_fn.__code__.co_freevars, eval_fn.__closure__))
    mapping["llm"].cell_contents = new_llm           # type: ignore
    mapping["retriever"].cell_contents = new_retriever   # type: ignore


def make_base_eval(
    retriever: BaseRetriever,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[[Dict[str, str]], Dict[str, Any]]:
    """Callable that executes the *eval* logic using **base** weights."""
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

    base_eval = build_chain(kind="eval", store="chroma")  # callable
    _patch_eval_callable(base_eval, new_llm=base_llm, new_retriever=retriever)
    return base_eval


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    query = (
        "Build a telemedicine platform that uses AI to diagnose skin conditions "
        "from smartphone photos, delivering timely dermatology advice to remote patients."
    )

    # ---------- LoRA chains --------------------------------------------------
    lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")
    retriever: BaseRetriever = lora_rag.retriever
    lora_eval = build_chain(kind="eval", store="chroma")

    # ---------- base chains --------------------------------------------------
    base_eval = make_base_eval(retriever)

    # ---------- show docs ----------------------------------------------------
    docs = retriever.get_relevant_documents(query)
    print("\n" + "=" * 80)
    print("Top-3 retrieved snippets")
    print("=" * 80)
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content.strip().replace(chr(10), ' ')}\n")

    # ---------- base answer --------------------------------------------------
    print("\n" + "=" * 80)
    print("Base LLaMA-3 (no-LoRA) → four recommendations")
    print("=" * 80)
    base_out = base_eval({"question": query})
    print(base_out["result"].strip())

    # ---------- LoRA answer --------------------------------------------------
    print("\n" + "=" * 80)
    print("LoRA-fine-tuned LLaMA-3 → four recommendations")
    print("=" * 80)
    lora_out = lora_eval({"question": query})
    print(lora_out["result"].strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")


if __name__ == "__main__":
    main()
