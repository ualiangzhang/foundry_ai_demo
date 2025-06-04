#!/usr/bin/env python3
"""
src/rag/chains.py  ·  Updated for QA and Startup Evaluation

Constructs four types of chains:
- eval  :  SerpApi → market snippet → LLaMA-3 summarization → four VC recommendations
- pitch :  Vector retrieval (top-3) → LLaMA-3 generates pitch-deck bullets
- rag   :  Vector retrieval (top-3) → generic RAG QA
- qa    :  SerpApi → top-3 web snippets → OpenAI GPT-4o-mini answer

Return values:
    eval → Callable({"question": summary}) → {
              "result": str,
              "context": str,
              "snippet": str,
              "docs": list[str]
           }
    pitch → RetrievalQA
    rag   → RetrievalQA
    qa   → Callable({"question": question}) → {
              "answer": str,
              "context": str
           }
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Dict, Literal, Optional, Any, List

import openai
import serpapi
import transformers

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import BaseRetriever
from langchain_core.prompt_values import ChatPromptValue

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
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

MARKET_KEYWORDS = [
    "Total Addressable Market 2025",
    "Compound Annual Growth Rate 2025",
    "Market Revenue 2025",
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
    """
    Query SerpApi for each keyword in MARKET_KEYWORDS combined with `summary`.
    For each search, take the first organic result's 'snippet', truncate to 50 words,
    and concatenate all such truncated snippets into a single string. If no valid
    snippet is found for any keyword, skip it. Return the combined string or None.
    """
    snippets: list[str] = []

    for kw in MARKET_KEYWORDS:
        query_text = f"{kw} {summary}"
        try:
            res = serpapi.search({
                "q": query_text,
                "engine": "google",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            })
        except Exception as e:
            logger.error(f"SerpApi search failed for '{query_text}': {e}")
            continue

        organic = res.get("organic_results", [])
        if not organic:
            continue

        first = organic[0]
        snippet_text = first.get("snippet", "")
        if not snippet_text:
            continue

        words = re.findall(r"\S+", snippet_text)[:50]
        truncated = " ".join(words)
        snippets.append(truncated)

        time.sleep(0.3)

    if not snippets:
        return None

    return " ".join(snippets)


def _summarize_context(
        llm: HuggingFacePipeline,
        summary: str,
        snippet: str,
) -> Optional[str]:
    """
    Convert snippet + summary into a market context by invoking the LLM.
    If the LLM’s output contains extra text around the JSON, extract the JSON
    block. On any error or missing structure, log the exception and return None.
    """
    template = (
        f"{CONTEXT_GEN_SYS}\n\n"
        f"Summary: {summary}\n\n"
        f"Snippet: {snippet}\n\n"
        "Generate JSON as specified above."
    )

    for _ in range(3):
        raw_output = llm.invoke(template).strip()

        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error(f"JSON braces not found or malformed in output: {raw_output}")
            continue

        json_str = raw_output[start_idx: end_idx + 1]
        try:
            parsed = json.loads(json_str)
            context = parsed.get("context", "").strip()
            if not context:
                logger.error(f"'context' key missing or empty in parsed JSON: {json_str}")
                continue
            return context

        except Exception as e:
            logger.error(f"Failed to parse JSON from LLM output: {e}; raw_output: {raw_output}")
            continue

    return None


def _fetch_qa_context(question: str) -> Optional[str]:
    """
    Query SerpApi with `question`, take the top-3 organic results' snippets,
    truncate each to 50 words, and concatenate them into a single `context` string.
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
    Ask ChatGPT-4o-mini to answer the question in ≤200 words using the provided context.
    Works with openai-python ≥1.0 (returns a ChatCompletion object).
    """
    if not openai.api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return None

    prompt = (
        "You are a knowledgeable AI assistant. Use the following web snippets purely as reference, "
        "but feel free to draw on your own general knowledge if the snippets are incomplete or unclear.\n\n"
        f"Context (web snippets):\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Provide a clear, accurate answer in no more than 200 words."
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are concise and accurate."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.8,
        )
    except Exception as e:
        logger.error(f"OpenAI chat.completions.create failed: {e}")
        return None

    # openai>=1.0 returns a ChatCompletion object → access via attributes
    if not resp.choices:
        return None

    return resp.choices[0].message.content.strip()


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
        - "eval": project evaluation (SerpApi snippet → summarize → four VC recommendations)
        - "pitch": pitch deck generation (vector retrieval + LLaMA-3)
        - "rag": generic RAG QA (vector retrieval + LLaMA-3)
        - "qa": direct QA (SerpApi snippets → OpenAI GPT-4o-mini answer)
      store: Vector store type for "pitch" and "rag" (ignored by "qa"):
        - "chroma" or "qdrant"
    """
    logger.info(f"build_chain(kind={kind}, store={store})")

    llm = _make_llm()

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) eval branch
    # ─────────────────────────────────────────────────────────────────────────────
    if kind == "eval":
        retriever = _build_retriever(store)

        def _eval_run(inputs: Dict[str, str]) -> Dict[str, Any]:
            summary = inputs.get("question", "").strip()
            if not summary:
                return {"result": "No summary provided", "context": "", "snippet": "", "docs": []}

            snippet = _fetch_market_snippet(summary)
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
    # 2) pitch or rag
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
    # 3) qa branch
    # ─────────────────────────────────────────────────────────────────────────────
    if kind == "qa":
        def _qa_run(inputs: Dict[str, str]) -> Dict[str, Any]:
            question = inputs.get("question", "").strip()
            if not question:
                return {"answer": "No question provided", "context": ""}

            context = _fetch_qa_context(question)
            if not context:
                return {"answer": "No relevant web snippets found.", "context": ""}

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
