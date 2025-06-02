#!/usr/bin/env python3
"""
tests/test_rag.py

Compare answers from base LLaMA-3 vs. LoRA-fine-tuned LLaMA-3
using the new chain API:

• build_chain(kind="rag")  → RetrievalQA (has .retriever)
• build_chain(kind="eval") → callable producing four bullets
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Callable

import transformers
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_community.llms import HuggingFacePipeline

# ── add repo root to PYTHONPATH ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── local imports (after path tweak) ─────────────────────────────────────────
from src.rag.chains import build_chain
from src.rag.model_loader import load_llama

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _patch_eval_callable(
    eval_fn: Callable[[Dict[str, str]], Dict[str, Any]],
    *,
    new_llm: HuggingFacePipeline,
    new_retriever: BaseRetriever,
) -> None:
    """
    Replace the 'llm' and 'retriever' free vars captured in the closure of an
    eval-callable produced by build_chain(kind="eval").
    """
    freevars = eval_fn.__code__.co_freevars         # tuple of names
    cells = eval_fn.__closure__ or ()

    mapping = dict(zip(freevars, cells))
    if "llm" not in mapping or "retriever" not in mapping:
        raise RuntimeError("Cannot monkey-patch eval_fn – variables missing")

    # mutate cell contents
    mapping["llm"].cell_contents = new_llm
    mapping["retriever"].cell_contents = new_retriever


def _make_base_eval(
    retriever: BaseRetriever,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
):
    """Return an eval-callable that uses **base** LLaMA-3 weights."""
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
# main test                                                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    query = (
        "Develop a sustainable leather alternative by cultivating and processing "
        "mycelium-based materials into high-quality, biodegradable leather substitutes "
        "for fashion and upholstery, reducing reliance on animal hides and "
        "minimizing environmental impact."
    )

    # ── 1.  LoRA chains ------------------------------------------------------
    lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")   # has retriever
    retriever: BaseRetriever = lora_rag.retriever

    lora_eval = build_chain(kind="eval", store="chroma")              # callable

    # ── 2.  Base chains ------------------------------------------------------
    base_rag_llm, _ = load_llama(use_lora=False)  # just to ensure weights load
    # we don't need the base_rag chain itself; we only need base_eval callable
    base_eval = _make_base_eval(retriever)

    # ── 3.  Display top-3 retrieved docs ------------------------------------
    top_docs = retriever.get_relevant_documents(query)
    print("\n" + "=" * 80)
    print("STEP 1 · Top-3 retrieved snippets")
    print("=" * 80)
    for i, d in enumerate(top_docs, 1):
        snippet = d.page_content.strip().replace("\n", " ")
        print(f"Snippet {i}: {snippet}\n")

    # ── 4.  Base answer ------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 · Base LLaMA-3 (no LoRA)")
    print("=" * 80)
    base_out = base_eval({"question": query})
    print(base_out["result"].strip())

    # ── 5.  LoRA answer ------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3 · LoRA-Fine-Tuned LLaMA-3")
    print("=" * 80)
    lora_out = lora_eval({"question": query})
    print(lora_out["result"].strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")


if __name__ == "__main__":
    main()
