#!/usr/bin/env python3
"""
tests/test_rag.py

Demonstrates the LoRA-fine-tuned LLaMA-3 model on a single query.
Steps:
  1. Build a RetrievalQA chain to obtain a shared retriever.
  2. Log the top-3 retrieved documents.
  3. Build the LoRA inference pipeline, construct the RAG prompt using those documents,
     and log the model’s input prompt and its resulting output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from src.rag.prompts import RAG_WRAPPER

# ── Logger configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_lora_pipeline(
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> HuggingFacePipeline:
    """
    Load the LoRA-fine-tuned LLaMA-3 model and wrap it in a HuggingFacePipeline.

    Args:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.

    Returns:
        A HuggingFacePipeline instance for text generation.
    """
    try:
        model, tokenizer = load_llama(use_lora=True)
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
    except Exception as e:
        logger.error(f"Failed to load LoRA model or tokenizer: {e}")
        raise


def _run_lora_rag(
    llm_pipeline: HuggingFacePipeline,
    query: str,
    retrieved_docs: List[str],
) -> Tuple[str, str]:
    """
    Construct the RAG prompt using top-3 retrieved documents and the query,
    then invoke the LoRA LLaMA-3 pipeline.

    Args:
        llm_pipeline: A HuggingFacePipeline wrapping the LoRA model.
        query: The user’s original question/summary.
        retrieved_docs: A list of up to 3 document strings for context.

    Returns:
        A tuple of:
            - prompt_str: The full prompt text sent to the model.
            - raw_output: The model’s raw generation output.
    """
    # Join retrieved_docs with separators
    context_text = "\n\n".join(retrieved_docs)
    try:
        # Format the ChatPromptValue for RAG_WRAPPER
        prompt_value = RAG_WRAPPER.format_prompt(context=context_text, question=query)
        prompt_str = prompt_value.to_string()
        raw_output = llm_pipeline.invoke(prompt_str).strip()
        return prompt_str, raw_output
    except Exception as e:
        logger.error(f"Exception during prompt construction or LoRA invoke: {e}")
        return "", ""


def main() -> None:
    """
    1. Instantiate RetrievalQA with LoRA chain to obtain shared retriever.
    2. Retrieve and log top-3 documents for the given query.
    3. Build the LoRA inference pipeline, construct the RAG prompt, and log
       both prompt and model output.
    """
    query: str = (
        "Develop a VR fitness platform with real-time coaching features "
        "and personalized workout plans to improve user engagement."
    )

    # Step 1: Build LoRA RetrievalQA chain to extract retriever only
    try:
        lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")
        retriever: BaseRetriever = lora_rag.retriever
        logger.info("Successfully instantiated RetrievalQA chain with LoRA.")
    except Exception as e:
        logger.error(f"Failed to build RetrievalQA chain: {e}")
        return

    # Step 2: Retrieve and log top-3 documents
    try:
        docs = retriever.get_relevant_documents(query)
        retrieved_texts: List[str] = [
            doc.page_content.strip().replace("\n", " ") for doc in docs[:3]
        ]
        logger.info("STEP 1 · Top-3 Retrieved Documents:")
        for idx, doc_text in enumerate(retrieved_texts, start=1):
            logger.info(f"[{idx}] {doc_text}")
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return

    # Step 3: Build the LoRA inference pipeline and run RAG
    try:
        llm_pipeline = _build_lora_pipeline(max_new_tokens=512, temperature=0.2)
        prompt_str, model_output = _run_lora_rag(
            llm_pipeline=llm_pipeline,
            query=query,
            retrieved_docs=retrieved_texts,
        )

        logger.info("STEP 2 · LoRA Model Input Prompt:")
        if prompt_str:
            logger.info(prompt_str)
        else:
            logger.error("LoRA input prompt is empty.")

        logger.info("STEP 3 · LoRA Model Output:")
        if model_output:
            logger.info(model_output)
        else:
            logger.error("LoRA model output is empty.")
    except Exception as e:
        logger.error(f"Exception during LoRA RAG execution: {e}")


if __name__ == "__main__":
    main()
