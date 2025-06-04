#!/usr/bin/env python3
"""
src/rag/chains.py  ·  Updated for QA

Adds a new "qa" chain that:
  1. Uses SerpApi to fetch the top-3 organic snippets for the question.
  2. Truncates each snippet to 50 words and concatenates them as `context`.
  3. Calls OpenAI’s ChatGPT-4o-mini with that `context` + the original question.
  4. Returns an answer limited to 200 words, plus the fetched context.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, Literal, Optional, Any, List

import openai
import serpapi
import transformers

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseRetriever

from scripts.generate_sft_examples import CONTEXT_GEN_SYS
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
openai.api_key = os.getenv("OPENAI_API_KEY")  # make sure this is set

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


def _fetch_qa_context(question: str) -> Optional[str]:
    """
    Query SerpApi with `question`, take the top-3 organic results' snippets,
    truncate each to 50 words, and concatenate them as a single `context` string.
    Returns None if no snippets found.
    """
    try:
        res = serpapi.search({
            "q": question,
            "engine": "google",
            "api_key": os.getenv("SERPAPI_API_KEY"),
        })
    except Exception as e:
        logger.error(f"SerpApi search failed for question '{question}': {e}")
        return None

    organic = res.get("organic_results", [])
    if not organic:
        return None

    snippets: List[str] = []
    count = 0
    for result in organic:
        if count >= 3:
            break
        snippet_text = result.get("snippet", "")
        if not snippet_text:
            continue
        # Truncate to 50 words
        words = re.findall(r"\S+", snippet_text)[:50]
        truncated = " ".join(words)
        snippets.append(truncated)
        count += 1
        time.sleep(0.2)

    if not snippets:
        return None

    return "\n\n".join(snippets)


def _generate_qa_answer(question: str, context: str) -> Optional[str]:
    """
    Send a prompt to OpenAI's ChatGPT-4o-mini combining the `context` and `question`.
    Instruct the model to produce an answer in ≤200 words.
    Returns the assistant's response or None on error.
    """
    if not openai.api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return None

    prompt = (
        f"You are a helpful assistant. Use the following web snippets as context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Please answer in no more than 200 words."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise and accurate assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,  # ~ 200 words
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"OpenAI ChatCompletion failed: {e}")
        return None

    choices = resp.get("choices", [])
    if not choices:
        return None

    return choices[0]["message"]["content"].strip()


def _build_retriever(store: str) -> BaseRetriever:
    """Create a Chroma or Qdrant retriever, configured to return top-3 documents."""
    if store == "chroma":
        retriever = chroma_retriever()
    elif store == "qdrant":
        retriever = qdrant_retriever()
    else:
        raise ValueError(f"Unsupported store '{store}'. Choose 'chroma' or 'qdrant'.")
    retriever.search_kwargs["k"] = 3
    return retriever


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_chain(
    kind: Literal["eval", "pitch", "rag", "qa"] = "eval",
    store: Literal["chroma", "qdrant"] = "chroma",
):
    """
    Returns:
      eval → Callable({"question": summary}) → {
                 "result": str,
                 "context": str,
                 "snippet": str,
                 "docs": list[str]
             }
      pitch → RetrievalQA (uses PITCH_DECK template, top-3 vector docs)
      rag → RetrievalQA (uses RAG_WRAPPER template, top-3 vector docs)
      qa → Callable({"question": question}) → {
                "answer": str,
                "context": str
            }

    Args:
      kind: Type of chain:
        - "eval": project evaluation (DuckDuckGo snippet → summarize → four VC recommendations)
        - "pitch": pitch deck generation (vector retrieval + LLaMA-3)
        - "rag": generic RAG QA (vector retrieval + LLaMA-3)
        - "qa": direct QA (SerpApi snippets → OpenAI GPT-4o-mini answer)
      store: Vector store type for "pitch" and "rag" (ignored by "qa"):
        - "chroma" or "qdrant"
    """
    logger.info(f"build_chain(kind={kind}, store={store})")

    # LLaMA-3 LLM (used by eval/pitch/rag)
    llm = _make_llm()

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) eval: unchanged
    # ─────────────────────────────────────────────────────────────────────────────
    if kind == "eval":
        retriever = _build_retriever(store)

        def _eval_run(inputs: Dict[str, str]) -> Dict[str, Any]:
            summary = inputs.get("question", "").strip()
            if not summary:
                return {"result": "No summary provided", "context": "", "snippet": "", "docs": []}

            snippet = _fetch_qa_context(summary)  # reuse for QA-like snippet gathering
            if not snippet:
                return {"result": "No snippet found", "context": "", "snippet": "", "docs": []}

            context = _summarize_context(llm, summary, snippet)
            if not context:
                return {"result": "No context found", "context": "", "snippet": snippet, "docs": []}

            pv: ChatPromptValue = PROJECT_EVAL.format_prompt(question=summary, context=context)
            rec_text: str = llm.invoke(pv.to_string()).strip()

            docs_text = [d.page_content for d in retriever.get_relevant_documents(summary)]

            return {"result": rec_text, "context": context, "snippet": snippet, "docs": docs_text}

        logger.info("Built eval callable.")
        return _eval_run

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) pitch or rag: unchanged
    # ─────────────────────────────────────────────────────────────────────────────
    if kind in {"pitch", "rag"}:
        retriever = _build_retriever(store)
        prompt_map = {"pitch": PITCH_DECK, "rag": RAG_WRAPPER}
        prompt_template = prompt_map[kind]
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
        )
        logger.info(f"RetrievalQA '{kind}' chain ready (top-3 docs).")
        return chain

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) qa: New direct QA using SerpApi + OpenAI GPT-4o-mini
    # ─────────────────────────────────────────────────────────────────────────────
    if kind == "qa":
        def _qa_run(inputs: Dict[str, str]) -> Dict[str, Any]:
            question = inputs.get("question", "").strip()
            if not question:
                return {"answer": "No question provided", "context": ""}

            # 1) Fetch context from SerpApi
            context = _fetch_qa_context(question)
            if not context:
                return {"answer": "No relevant web snippets found.", "context": ""}

            # 2) Ask OpenAI
            answer = _generate_qa_answer(question, context)
            if not answer:
                return {"answer": "Failed to generate an answer.", "context": context}

            return {"answer": answer, "context": context}

        logger.info("Built QA callable.")
        return _qa_run

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Unsupported kind
    # ─────────────────────────────────────────────────────────────────────────────
    raise ValueError(f"Unsupported chain kind '{kind}'. Choose 'eval', 'pitch', 'rag', or 'qa'.")
