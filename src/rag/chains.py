#!/usr/bin/env python3
"""
src/rag/chains.py

Defines functions to build LangChain retrieval-augmented generation (RAG) chains
for startup document evaluation and pitch deck generation using LLaMA3.

Functions:
    - _make_llm: Load LLaMA3 and wrap it in a HuggingFacePipeline.
    - build_chain: Construct a RetrievalQA chain given a task kind and vector store.

Usage:
    from src.rag.chains import build_chain
    qa_chain = build_chain(kind="eval", store="chroma")
    result = qa_chain.run("Your project summary here")
"""

import logging
from typing import Literal, Optional

import transformers
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from .model_loader import load_llama
from .retriever import chroma_retriever, qdrant_retriever
from .prompts import RAG_WRAPPER, PROJECT_EVAL, PITCH_DECK
from langchain.schema import BaseRetriever

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def build_chain(
        kind: Literal["eval", "pitch", "rag"] = "eval",
        store: Literal["chroma", "qdrant"] = "chroma"
) -> RetrievalQA:
    """Construct a RetrievalQA chain for the specified task and vector store.

    Args:
        kind: Type of chain to build:
            - "eval": project evaluation (uses PROJECT_EVAL prompt)
            - "pitch": pitch deck generation (uses PITCH_DECK prompt)
            - "rag": generic RAG wrapper (uses RAG_WRAPPER prompt)
        store: Which vector store to use for retrieval:
            - "chroma": use a local Chroma collection
            - "qdrant": use a Qdrant instance

    Returns:
        A LangChain RetrievalQA chain ready to run queries.

    Raises:
        ValueError: If `kind` or `store` arguments are invalid.
        RuntimeError: If retriever initialization or chain creation fails.
    """
    logger.info(f"Building RAG chain with kind='{kind}', store='{store}'...")

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

    # Select prompt template
    prompt_map = {
        "eval": PROJECT_EVAL,
        "pitch": PITCH_DECK,
        "rag": RAG_WRAPPER
    }
    prompt_template: Optional[str] = prompt_map.get(kind)
    if prompt_template is None:
        msg = f"Unsupported chain kind '{kind}'; choose 'eval', 'pitch', or 'rag'."
        logger.error(msg)
        raise ValueError(msg)

    # Create LLM wrapper
    try:
        llm = _make_llm()
    except Exception as e:
        msg = f"Failed to create LLM for chain: {e}"
        logger.error(msg)
        raise RuntimeError(msg)

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
