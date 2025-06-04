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
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import openai
import serpapi
import transformers
from langchain.chains import RetrievalQA
from langchain_core.prompt_values import ChatPromptValue
from langchain.schema import BaseRetriever
from langchain_community.llms import HuggingFacePipeline

from scripts.generate_sft_examples import CONTEXT_GEN_SYS
from .model_loader import load_llama
from .prompts import PITCH_DECK, PROJECT_EVAL, RAG_WRAPPER
from .retriever import chroma_retriever, qdrant_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this environment variable is set

MARKET_KEYWORDS: List[str] = [
    "Total Addressable Market 2025",
    "Compound Annual Growth Rate 2025",
    "Market Revenue 2025",
]


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------
def _make_llm(max_new_tokens: int = 512, temperature: float = 0.2) -> HuggingFacePipeline:
    """Wrap LLaMA-3 in a HuggingFacePipeline for LangChain.

    Args:
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        HuggingFacePipeline: Configured pipeline instance for LLaMA-3.
    """
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
    """Query SerpApi for market snippets based on `summary` and keywords.

    For each entry in MARKET_KEYWORDS, performs a Google search via SerpApi
    and extracts the first organic result's snippet. Truncates each snippet
    to 50 words and concatenates them into a single string.

    Args:
        summary (str): The startup summary to augment search queries.

    Returns:
        Optional[str]: Concatenated snippets string if any found; otherwise None.
    """
    snippets: List[str] = []

    for kw in MARKET_KEYWORDS:
        query_text = f"{kw} {summary}"
        try:
            res: Dict[str, Any] = serpapi.search(
                {
                    "q": query_text,
                    "engine": "google",
                    "api_key": os.getenv("SERPAPI_API_KEY"),
                }
            )
        except Exception as e:
            logger.error(f"SerpApi search failed for '{query_text}': {e}")
            continue

        organic: List[Dict[str, Any]] = res.get("organic_results", [])
        if not organic:
            continue

        first: Dict[str, Any] = organic[0]
        snippet_text: str = first.get("snippet", "")
        if not snippet_text:
            continue

        words: List[str] = re.findall(r"\S+", snippet_text)[:50]
        truncated: str = " ".join(words)
        snippets.append(truncated)

        time.sleep(0.3)

    if not snippets:
        return None

    return " ".join(snippets)


def _summarize_context(
        llm: HuggingFacePipeline, summary: str, snippet: str
) -> Optional[str]:
    """Generate a ~100-word market context from snippet and summary via LLaMA-3.

    Sends a prompt to the LLaMA-3 pipeline using CONTEXT_GEN_SYS instructions.
    Extracts JSON with a 'context' field. Retries up to 3 times if parsing fails.

    Args:
        llm (HuggingFacePipeline): The LLaMA-3 pipeline.
        summary (str): Startup summary text.
        snippet (str): Market snippet text.

    Returns:
        Optional[str]: Generated market context if successful; otherwise None.
    """
    template: str = (
        f"{CONTEXT_GEN_SYS}\n\n"
        f"Summary: {summary}\n\n"
        f"Snippet: {snippet}\n\n"
        "Generate JSON as specified above."
    )

    for _ in range(3):
        raw_output: str = llm.invoke(template).strip()

        start_idx: int = raw_output.find("{")
        end_idx: int = raw_output.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            logger.error(f"JSON braces not found or malformed in output: {raw_output}")
            continue

        json_str: str = raw_output[start_idx: end_idx + 1]
        try:
            parsed: Dict[str, Any] = json.loads(json_str)
            context: str = parsed.get("context", "").strip()
            if not context:
                logger.error(f"'context' key missing or empty in parsed JSON: {json_str}")
                continue
            return context

        except Exception as e:
            logger.error(f"Failed to parse JSON from LLM output: {e}; raw_output: {raw_output}")
            continue

    return None


def _fetch_qa_context(question: str) -> Optional[str]:
    """Retrieve top-3 web snippets for a question via SerpApi and concatenate them.

    Args:
        question (str): The user’s question.

    Returns:
        Optional[str]: Concatenated snippets separated by double newline if found; otherwise None.
    """
    try:
        res: Dict[str, Any] = serpapi.search(
            {
                "q": question,
                "engine": "google",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
    except Exception as e:
        logger.error(f"SerpApi search failed for question '{question}': {e}")
        return None

    organic: List[Dict[str, Any]] = res.get("organic_results", [])
    if not organic:
        return None

    snippets: List[str] = []
    count: int = 0
    for result in organic:
        if count >= 3:
            break
        snippet_text: str = result.get("snippet", "")
        if not snippet_text:
            continue
        words: List[str] = re.findall(r"\S+", snippet_text)[:50]
        truncated: str = " ".join(words)
        snippets.append(truncated)
        count += 1
        time.sleep(0.2)

    if not snippets:
        return None

    return "\n\n".join(snippets)


def _generate_qa_answer(question: str, context: str) -> Optional[str]:
    """Use OpenAI GPT-4o-mini to answer a question using provided context.

    Constructs a prompt that instructs the model to use snippets as reference
    and answer in no more than 200 words. Accesses `openai.chat.completions.create`
    for openai>=1.0. Retries not implemented here (assumes caller handles errors).

    Args:
        question (str): The user’s question.
        context (str): Concatenated web snippets.

    Returns:
        Optional[str]: The answer text if successful; otherwise None.
    """
    if not openai.api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return None

    prompt: str = (
        "You are a knowledgeable AI assistant. Use the following web snippets purely as reference, "
        "but feel free to draw on your own general knowledge if the snippets are incomplete or unclear.\n\n"
        f"Context (web snippets):\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Provide a clear, accurate answer in no more than 200 words."
    )

    try:
        resp: Any = openai.chat.completions.create(
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

    if not getattr(resp, "choices", None):
        return None

    return resp.choices[0].message.content.strip()


def _build_retriever(store: str) -> BaseRetriever:
    """Create a Chroma or Qdrant retriever configured to return top-3 documents.

    Args:
        store (str): Either 'chroma' or 'qdrant'.

    Returns:
        BaseRetriever: Instantiated retriever with k=3.

    Raises:
        ValueError: If unsupported store is provided.
    """
    if store == "chroma":
        retriever: BaseRetriever = chroma_retriever()
    elif store == "qdrant":
        retriever: BaseRetriever = qdrant_retriever()
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
) -> Callable[[Dict[str, str]], Dict[str, Any]]:
    """Build and return a processing chain based on the requested kind.

    Args:
        kind (Literal["eval", "pitch", "rag", "qa"]): Type of chain to build.
          - "eval": project evaluation (SerpApi snippet → summarize → four VC recommendations)
          - "pitch": pitch deck generation (vector retrieval + LLaMA-3)
          - "rag": generic RAG QA (vector retrieval + LLaMA-3)
          - "qa": direct QA (SerpApi snippets → OpenAI GPT-4o-mini answer)
        store (Literal["chroma", "qdrant"]): Vector store type for "pitch" and "rag";
          ignored by "qa".

    Returns:
        Callable[[Dict[str, str]], Dict[str, Any]]: A function that accepts a dict
        with key "question" and returns a dict of results for the chosen chain.

    Raises:
        ValueError: If unsupported 'kind' is provided.
    """
    logger.info(f"build_chain(kind={kind}, store={store})")

    llm: HuggingFacePipeline = _make_llm()

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) eval branch
    # ─────────────────────────────────────────────────────────────────────────────
    if kind == "eval":
        retriever: BaseRetriever = _build_retriever(store)

        def _eval_run(inputs: Dict[str, str]) -> Dict[str, Any]:
            """Execute the 'eval' chain: fetch snippet, summarize context, generate recommendations."""
            summary: str = inputs.get("question", "").strip()
            if not summary:
                return {"result": "No summary provided", "context": "", "snippet": "", "docs": []}

            snippet: Optional[str] = _fetch_market_snippet(summary)
            if not snippet:
                return {"result": "No snippet found", "context": "", "snippet": "", "docs": []}

            context: Optional[str] = _summarize_context(llm, summary, snippet)
            if not context:
                return {"result": "No context found", "context": "", "snippet": snippet, "docs": []}

            pv: ChatPromptValue = PROJECT_EVAL.format_prompt(question=summary, context=context)
            rec_text: str = llm.invoke(pv.to_string()).strip()

            docs_list: List[str] = [doc.page_content for doc in retriever.get_relevant_documents(summary)]

            return {"result": rec_text, "context": context, "snippet": snippet, "docs": docs_list}

        logger.info("Built eval callable.")
        return _eval_run

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) pitch or rag
    # ─────────────────────────────────────────────────────────────────────────────
    if kind in {"pitch", "rag"}:
        retriever: BaseRetriever = _build_retriever(store)
        prompt_map: Dict[str, Any] = {"pitch": PITCH_DECK, "rag": RAG_WRAPPER}
        prompt_template: Any = prompt_map[kind]
        chain: RetrievalQA = RetrievalQA.from_chain_type(
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
            """Execute the 'qa' chain: fetch web context, call OpenAI for answer."""
            question: str = inputs.get("question", "").strip()
            if not question:
                return {"answer": "No question provided", "context": ""}

            context_str: Optional[str] = _fetch_qa_context(question)
            if not context_str:
                return {"answer": "No relevant web snippets found.", "context": ""}

            answer_text: Optional[str] = _generate_qa_answer(question, context_str)
            if not answer_text:
                return {"answer": "Failed to generate an answer.", "context": context_str}

            return {"answer": answer_text, "context": context_str}

        logger.info("Built QA callable.")
        return _qa_run

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Unsupported kind
    # ─────────────────────────────────────────────────────────────────────────────
    raise ValueError(f"Unsupported chain kind '{kind}'. Choose 'eval', 'pitch', 'rag', or 'qa'.")
