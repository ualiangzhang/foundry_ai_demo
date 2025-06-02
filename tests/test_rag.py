#!/usr/bin/env python3
"""
src/rag/chains.py   ·  resilient version

Changes vs. prior commit
------------------------
• _fetch_market_snippet(...) now accepts a `retriever` and, if DuckDuckGo fails,
  extracts the first digit-containing sentence from the top-3 retrieved docs.
• eval branch calls that new helper and therefore rarely returns
  "INSUFFICIENT_CONTEXT".
Everything else (pitch / rag chains) is unchanged.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Dict, Literal, Optional, Any

import transformers
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseRetriever
from langchain_core.prompt_values import ChatPromptValue

from scripts.generate_sft_examples import (
    duck_top1_snippet,
    CONTEXT_GEN_SYS,
)
from .model_loader import load_llama
from .prompts import RAG_WRAPPER, PROJECT_EVAL, PITCH_DECK
from .retriever import chroma_retriever, qdrant_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
MARKET_KEYWORDS = [
    "market size 2025",
    "global market value",
    "compound annual growth rate",
]

DIGIT_SENTENCE_RE = re.compile(r".*?\d+.*?[.!?]")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _make_llm(max_new_tokens: int = 512, temperature: float = 0.2) -> HuggingFacePipeline:
    model, tok = load_llama()
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _first_digit_sentence(text: str) -> Optional[str]:
    """Return first sentence containing a digit, ≤50 words."""
    match = DIGIT_SENTENCE_RE.search(text)
    if match:
        sent = match.group(0)
        words = re.findall(r"\S+", sent)[:50]
        return " ".join(words)
    return None


def _fetch_market_snippet(summary: str, retriever: BaseRetriever) -> Optional[str]:
    """Try DuckDuckGo; if that fails, fall back to retriever docs."""
    # 1- DuckDuckGo
    for kw in random.sample(MARKET_KEYWORDS, len(MARKET_KEYWORDS)):
        snippet = duck_top1_snippet(f"{summary} {kw}")
        if snippet and any(ch.isdigit() for ch in snippet):
            return " ".join(re.findall(r"\S+", snippet)[:50])
        time.sleep(0.3)

    # 2- Fallback: look into retrieved docs
    docs = retriever.get_relevant_documents(summary)[:3]
    for doc in docs:
        fallback = _first_digit_sentence(doc.page_content)
        if fallback:
            return fallback

    return None


def _summarise_context(llm: HuggingFacePipeline, summary: str, snippet: str) -> Optional[str]:
    prompt = (
        f"{CONTEXT_GEN_SYS}\n\n"
        f"Summary: {summary}\n\n"
        f"Snippet: {snippet}\n\n"
        "Generate JSON as specified above."
    )
    for _ in range(3):
        raw = llm.invoke(prompt).strip()
        try:
            ctx = json.loads(raw).get("context", "").strip()
            wc = len(re.findall(r"\S+", ctx))
            if 80 <= wc <= 140:
                return ctx
        except Exception:
            pass
    return None


def _build_retriever(store: str) -> BaseRetriever:
    if store == "chroma":
        r = chroma_retriever()
    elif store == "qdrant":
        r = qdrant_retriever()
    else:
        raise ValueError("store must be 'chroma' or 'qdrant'")
    r.search_kwargs["k"] = 3
    return r


# --------------------------------------------------------------------------- #
# build_chain                                                                 #
# --------------------------------------------------------------------------- #
def build_chain(
    kind: Literal["eval", "pitch", "rag"] = "eval",
    store: Literal["chroma", "qdrant"] = "chroma",
):
    llm = _make_llm()

    # -------------------------- eval ----------------------------------------
    if kind == "eval":
        retriever = _build_retriever(store)

        def _eval(inputs: Dict[str, str]) -> Dict[str, Any]:
            summary = inputs.get("question", "").strip()
            if not summary:
                return {"result": "INSUFFICIENT_CONTEXT", "error": "missing summary"}

            snippet = _fetch_market_snippet(summary, retriever)
            if not snippet:
                return {"result": "INSUFFICIENT_CONTEXT", "error": "no numeric snippet"}

            context = _summarise_context(llm, summary, snippet)
            if not context:
                return {"result": "INSUFFICIENT_CONTEXT", "error": "context build failed"}

            pv: ChatPromptValue = PROJECT_EVAL.format_prompt(
                question=summary, context=context
            )
            recs = llm.invoke(pv.to_string()).strip()

            docs_text = [d.page_content for d in retriever.get_relevant_documents(summary)]

            return {
                "result": recs,
                "context": context,
                "snippet": snippet,
                "docs": docs_text,
            }

        return _eval

    # -------------------------- pitch / rag ---------------------------------
    retriever = _build_retriever(store)
    prompt = {"pitch": PITCH_DECK, "rag": RAG_WRAPPER}[kind]
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
