#!/usr/bin/env python3
"""
src/rag/chains.py

Defines functions to build LangChain retrieval-augmented generation (RAG) chains
for startup document evaluation and pitch deck generation using LLaMA3.

For project evaluation ("eval"), rather than using a vector store retriever,
we fetch a numeric market snippet via DuckDuckGo and build the “context”
(on-the-fly), then invoke LLaMA3 with the PROJECT_EVAL prompt (summary + context).

Functions:
    - _make_llm: Load LLaMA3 and wrap it in a HuggingFacePipeline.
    - build_chain: Construct either a “duckduckgo→LLM” chain for eval, or a
      standard RetrievalQA chain for “pitch” and “rag” kinds.
"""

import logging
import random
import re
import time
from typing import Literal, Optional

import transformers
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# Import the DuckDuckGo helper from the SFT script
from scripts.generate_sft_examples import duck_top1_snippet

from .model_loader import load_llama
from .prompts import RAG_WRAPPER, PROJECT_EVAL, PITCH_DECK
from .retriever import chroma_retriever, qdrant_retriever
from langchain.schema import BaseRetriever

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Keywords to append when searching for market data in eval
MARKET_KEYWORDS = [
    "market size 2025", "Total Addressable Market", "Compound Annual Growth Rate"
]


def _make_llm(
    max_new_tokens: int = 512,
    temperature: float = 0.2
) -> HuggingFacePipeline:
    """Load LLaMA3 model and wrap it in a HuggingFacePipeline for text generation.

    Args:
        max_new_tokens: Maximum number of tokens to generate for each call.
        temperature: Sampling temperature for generation (lower == more deterministic).

    Returns:
        A HuggingFacePipeline object that executes text generation with LLaMA3.

    Raises:
        RuntimeError: If loading the LLaMA3 model or tokenizer fails.
    """
    logger.info("Loading LLaMA3 model and tokenizer...")
    try:
        model, tokenizer = load_llama()
    except Exception as e:
        msg = f"Failed to load LLaMA3 model or tokenizer: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.1,
        )
    except Exception as e:
        msg = f"Failed to initialize HuggingFace text-generation pipeline: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Wrapping pipeline in HuggingFacePipeline for LangChain...")
    return HuggingFacePipeline(pipeline=pipeline)


def _fetch_market_context(summary: str) -> Optional[str]:
    """
    Query DuckDuckGo for a numeric snippet related to `summary`.
    Try each keyword until one yields a snippet containing at least one digit.
    Returns the first valid snippet string (≤50 words), or None if none found.
    """
    keywords = random.sample(MARKET_KEYWORDS, k=len(MARKET_KEYWORDS))
    for kw in keywords:
        query = f"{summary} {kw}"
        snippet = duck_top1_snippet(query)
        if snippet:
            # Truncate to 50 words just in case
            words = re.findall(r"\S+", snippet)[:50]
            return " ".join(words)
        time.sleep(0.5)
    return None


def build_chain(
    kind: Literal["eval", "pitch", "rag"] = "eval",
    store: Literal["chroma", "qdrant"] = "chroma"
):
    """Construct either a “duckduckgo→LLM” chain for eval, or a RetrievalQA chain.

    Args:
        kind: Type of chain to build:
            - "eval": project evaluation (uses PROJECT_EVAL prompt; context fetched via DuckDuckGo)
            - "pitch": pitch deck generation (uses PITCH_DECK prompt and vector-store retrieval)
            - "rag": generic RAG wrapper (uses RAG_WRAPPER prompt and vector-store retrieval)
        store: Which vector store to use (only applies to "pitch" and "rag"):
            - "chroma": use a local Chroma collection
            - "qdrant": use a Qdrant instance

    Returns:
        A chain object:
          - If kind="eval": returns an LLMChain that expects an input dict {"question": <summary>}.
          - If kind="pitch" or "rag": returns a RetrievalQA chain.

    Raises:
        ValueError: If `kind` or `store` arguments are invalid.
        RuntimeError: If model loading, retriever initialization, or chain creation fails.
    """
    logger.info(f"Building chain with kind='{kind}', store='{store}'...")

    # Create LLM wrapper
    try:
        llm = _make_llm()
    except Exception as e:
        msg = f"Failed to create LLM for chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    if kind == "eval":
        # For evaluation, we do not use a vector store. Instead:
        #   1. Expect input "question" = startup summary
        #   2. Fetch a numeric snippet via DuckDuckGo to form "context"
        #   3. Render the PROJECT_EVAL prompt with that summary + context
        #   4. Call LLM to get four recommendations

        # Define a custom prompt template that takes both summary and context
        prompt = PROJECT_EVAL

        def _eval_chain_run(inputs: dict) -> str:
            summary = inputs.get("question", "").strip()
            if not summary:
                return "INSUFFICIENT_CONTEXT"
            snippet = _fetch_market_context(summary)
            if not snippet:
                return "INSUFFICIENT_CONTEXT"
            # Combine summary and context into the prompt
            merged_inputs = {"question": summary, "context": snippet}
            # Format messages and run the LLM
            messages = prompt.format_prompt(**merged_inputs).to_messages()
            return llm.generate(messages)[0].text.strip()

        logger.info("Built DuckDuckGo-based eval chain.")
        return _eval_chain_run

    # For "pitch" and "rag", use a vector-store retriever + RetrievalQA
    # Select retriever
    try:
        if store == "chroma":
            retriever: BaseRetriever = chroma_retriever()
        elif store == "qdrant":
            retriever = qdrant_retriever()
        else:
            raise ValueError(f"Unsupported store '{store}'; choose 'chroma' or 'qdrant'.")
    except Exception as e:
        msg = f"Failed to initialize retriever for store '{store}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    # Select prompt template for RetrievalQA
    prompt_map = {
        "pitch": PITCH_DECK,
        "rag": RAG_WRAPPER
    }
    prompt_template = prompt_map.get(kind)
    if prompt_template is None:
        msg = f"Unsupported chain kind '{kind}'; choose 'eval', 'pitch', or 'rag'."
        logger.error(msg)
        raise ValueError(msg)

    # Build the RetrievalQA chain
    try:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )
        logger.info(f"RetrievalQA chain '{kind}' successfully built.")
        return chain
    except Exception as e:
        msg = f"Failed to instantiate RetrievalQA chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)
