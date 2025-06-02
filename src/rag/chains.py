#!/usr/bin/env python3
"""
src/rag/chains.py  ·  Updated

Constructs three types of chains:
- eval  :  DuckDuckGo → market snippet → LLaMA-3 summarization → four VC recommendations
- pitch :  Vector retrieval (top-3) → LLaMA-3 generates pitch-deck bullets
- rag   :  Vector retrieval (top-3) → generic RAG QA

Return values:
    eval  → Callable({"question": summary}) → {
              "recommendations": str,
              "context": str,
              "snippet": str,
              "docs": list[str]
           }
    pitch → RetrievalQA
    rag   → RetrievalQA
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Dict, Literal, Optional

import transformers
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseRetriever

# Reuse DuckDuckGo helper and context-generation instructions from the SFT script
from scripts.generate_sft_examples import (
    duck_top1_snippet,
    CONTEXT_GEN_SYS,  # Instructions for ~100-word context JSON
)
from .model_loader import load_llama
from .prompts import RAG_WRAPPER, PROJECT_EVAL, PITCH_DECK
from .retriever import chroma_retriever, qdrant_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MARKET_KEYWORDS = [
    "market size 2025",
    "Total Addressable Market",
    "Compound Annual Growth Rate",
]


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------
def _make_llm(max_new_tokens: int = 512, temperature: float = 0.2) -> HuggingFacePipeline:
    """Wrap LLaMA-3 in a HuggingFacePipeline for LangChain."""
    logger.info("Loading LLaMA-3 model and tokenizer...")
    model, tokenizer = load_llama()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipeline)


def _fetch_market_snippet(summary: str) -> Optional[str]:
    """Query DuckDuckGo for a numeric snippet related to `summary`. Return ≤50 words with at least one digit."""
    for kw in random.sample(MARKET_KEYWORDS, len(MARKET_KEYWORDS)):
        snippet = duck_top1_snippet(f"{summary} {kw}")
        if snippet and any(ch.isdigit() for ch in snippet):
            # Truncate to 50 words
            words = re.findall(r"\S+", snippet)[:50]
            return " ".join(words)
        time.sleep(0.3)
    return None


def _summarize_context(
    llm: HuggingFacePipeline,
    summary: str,
    snippet: str
) -> Optional[str]:
    """
    Use the same LLaMA-3 model to convert snippet + summary into a ~100-word
    market context. Follows the CONTEXT_GEN_SYS instructions from SFT.
    """
    template = (
        CONTEXT_GEN_SYS
        + "\n\nSummary: {summary}\n\nSnippet: {snippet}\n\nGenerate JSON as specified above."
    )
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)

    for _ in range(3):
        raw_output = chain.run({"summary": summary, "snippet": snippet}).strip()
        try:
            parsed = json.loads(raw_output)
            context = parsed.get("context", "").strip()
            wc = len(re.findall(r"\S+", context))
            if 80 <= wc <= 140:
                return context
        except Exception:
            pass

    return None


def _build_retriever(store: str) -> BaseRetriever:
    """Create a Chroma or Qdrant retriever, configured to return top-3 documents."""
    if store == "chroma":
        retriever = chroma_retriever()
    elif store == "qdrant":
        retriever = qdrant_retriever()
    else:
        raise ValueError(f"Unsupported store '{store}'. Choose 'chroma' or 'qdrant'.")

    # Ensure only top-3 are returned
    retriever.search_kwargs["k"] = 3
    return retriever


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_chain(
    kind: Literal["eval", "pitch", "rag"] = "eval",
    store: Literal["chroma", "qdrant"] = "chroma",
):
    """
    Returns:
      eval  → Callable({"question": summary}) → {
                 "recommendations": str,
                 "context": str,
                 "snippet": str,
                 "docs": list[str]
               }
      pitch → RetrievalQA (uses PITCH_DECK template, top-3 vector docs)
      rag   → RetrievalQA (uses RAG_WRAPPER template, top-3 vector docs)

    Args:
      kind: Type of chain:
        - "eval": project evaluation (DuckDuckGo snippet → summarize → four VC recommendations)
        - "pitch": pitch deck generation (vector retrieval + LLaMA-3)
        - "rag": generic RAG QA (vector retrieval + LLaMA-3)
      store: Vector store type for "pitch" and "rag":
        - "chroma" or "qdrant"
    """
    logger.info(f"build_chain(kind={kind}, store={store})")
    llm = _make_llm()

    # -------------------------------------------- "eval" chain
    if kind == "eval":
        retriever = _build_retriever(store)

        def _eval_run(inputs: Dict[str, str]) -> Dict[str, str]:
            summary = inputs.get("question", "").strip()
            if not summary:
                return {
                    "error": "Missing summary",
                    "recommendations": "INSUFFICIENT_CONTEXT"
                }

            snippet = _fetch_market_snippet(summary)
            if not snippet:
                return {
                    "error": "No numeric snippet found",
                    "recommendations": "INSUFFICIENT_CONTEXT"
                }

            context = _summarize_context(llm, summary, snippet)
            if not context:
                return {
                    "error": "Failed to build market context",
                    "recommendations": "INSUFFICIENT_CONTEXT"
                }

            # Generate four VC recommendations
            rec_messages = PROJECT_EVAL.format_prompt(
                question=summary,
                context=context
            ).to_messages()
            recommendations = llm.generate(rec_messages)[0].text.strip()

            # Retrieve top-3 docs for debugging/reference
            docs = retriever.get_relevant_documents(summary)
            docs_texts = [doc.page_content for doc in docs[:3]]

            return {
                "result": recommendations,
                "context": context,
                "snippet": snippet,
                "docs": docs_texts,
            }

        logger.info("Built DuckDuckGo-based eval chain.")
        return _eval_run

    # ----------------------- "pitch" or "rag" use RetrievalQA
    retriever = _build_retriever(store)
    prompt_map = {"pitch": PITCH_DECK, "rag": RAG_WRAPPER}
    prompt_template = prompt_map.get(kind)
    if prompt_template is None:
        raise ValueError(f"Unsupported chain kind '{kind}'.")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
    )
    logger.info(f"RetrievalQA '{kind}' chain ready (top-3 docs).")
    return chain
