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

# ── Add project root to PYTHONPATH ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Local imports ────────────────────────────────────────────────────────────
from src.rag.chains import build_chain
from src.rag.model_loader import load_llama

# ── Logger ───────────────────────────────────────────────────────────────────
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
    """
    Replace captured 'llm' and 'retriever' inside the eval closure
    of a build_chain(kind="eval") callable.
    """
    freevars = eval_fn.__code__.co_freevars
    cells = eval_fn.__closure__ or []
    mapping = dict(zip(freevars, cells))
    if "llm" in mapping and "retriever" in mapping:
        mapping["llm"].cell_contents = new_llm           # type: ignore
        mapping["retriever"].cell_contents = new_retriever   # type: ignore
    else:
        raise RuntimeError("Failed to patch eval callable: missing free variables")


def make_base_eval(
    retriever: BaseRetriever,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[[Dict[str, str]], Dict[str, Any]]:
    """
    Return a callable that implements the 'eval' logic using base (no-LoRA) weights.

    Steps inside:
      1. build_chain(kind="eval", store="chroma") → an eval-callable
      2. Monkey-patch its closure so 'llm' uses base LLaMA-3 and
         'retriever' is the shared retriever.
    """
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

    base_eval = build_chain(kind="eval", store="chroma")  # returns a callable
    _patch_eval_callable(base_eval, new_llm=base_llm, new_retriever=retriever)
    return base_eval


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    1. Build the LoRA-fine-tuned eval callable and the rag retriever.
    2. Build the base eval callable (monkey-patched) with the same retriever.
    3. Print top-3 docs (shared).
    4. Print four-bullet recommendations from both chains.
    """

    # Use a query that is likely to yield a numeric snippet when combined
    # with "market size 2025" etc.
    query = (
        "Develop a VR fitness platform with real-time coaching features "
        "and personalized workout plans to improve user engagement."
    )

    # ---------- 1. LoRA chains ------------------------------------------------
    # build a RetrievalQA purely to get .retriever
    lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")
    retriever: BaseRetriever = lora_rag.retriever

    # build the eval callable (four bullets) using LoRA weights
    lora_eval = build_chain(kind="eval", store="chroma")

    # ---------- 2. Base eval callable -----------------------------------------
    # monkey-patch the eval closure so it uses base LLaMA-3 weights + same retriever
    base_eval = make_base_eval(retriever)

    # ---------- 3. Show top-3 retrieved docs ----------------------------------
    docs = retriever.get_relevant_documents(query)
    print("\n" + "=" * 80)
    print("STEP 1  ·  Top-3 retrieved snippets (shared by both models)")
    print("=" * 80)
    for idx, d in enumerate(docs[:3], start=1):
        text = d.page_content.strip().replace("\n", " ")
        print(f"[{idx}] {text}\n")

    # ---------- 4. Base model answer ------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2  ·  Base LLaMA-3 (no-LoRA) → four recommendations")
    print("=" * 80)
    base_out = base_eval({"question": query})
    print(base_out["result"].strip())

    # ---------- 5. LoRA model answer ------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3  ·  LoRA-Fine-Tuned LLaMA-3 → four recommendations")
    print("=" * 80)
    lora_out = lora_eval({"question": query})
    print(lora_out["result"].strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")


if __name__ == "__main__":
    main()
