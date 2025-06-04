#!/usr/bin/env python3
"""
tests/test_rag.py

Demonstrates the LoRA-fine-tuned LLaMA-3 model on a single query.

Steps:
  1. Build a RetrievalQA chain to obtain a shared retriever.
  2. Build the eval callable (four recommendations) using LoRA weights only.
  3. Log the top-3 retrieved documents.
  4. Log the LoRA model’s input prompt and its resulting output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import transformers
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_community.llms import HuggingFacePipeline

# ── Add project root to PYTHONPATH ───────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Local imports ────────────────────────────────────────────────────────────
from src.rag.chains import build_chain
from src.rag.model_loader import load_llama
from src.rag.prompts import PROJECT_EVAL

# ── Logger configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


def _extract_prompt_and_response(
        llm_pipeline: HuggingFacePipeline,
        summary: str,
        context: str,
) -> Tuple[str, str]:
    """Builds the PROJECT_EVAL prompt and invokes the LLM pipeline.

    Args:
        llm_pipeline (HuggingFacePipeline): A pipeline wrapping the LoRA model.
        summary (str): The startup idea summary to include as "question".
        context (str): The market context to include as "context".

    Returns:
        Tuple[str, str]:
            - The full text prompt sent to the model.
            - The raw textual output produced by the model.
    """
    try:
        # Format the ChatPromptValue for PROJECT_EVAL
        prompt_value = PROJECT_EVAL.format_prompt(question=summary, context=context)
        # Convert to a single-string prompt
        prompt_str: str = prompt_value.to_string()
        # Invoke the LLM pipeline and capture raw output
        raw_output: str = llm_pipeline.invoke(prompt_str).strip()
        return prompt_str, raw_output
    except Exception as e:
        logger.error(f"Exception during prompt construction or LLM invocation: {e}")
        return "", ""


def main() -> None:
    """Run a demonstration of RetrievalQA with a LoRA-fine-tuned LLaMA-3 model.

    1. Instantiate RetrievalQA with LoRA-based chain to obtain the retriever.
    2. Instantiate the eval callable (LoRA-based) to produce four recommendations.
    3. Retrieve and log the top-3 documents for a test query.
    4. Load the LoRA model separately and log the prompt & output for evaluation.

    Returns:
        None
    """
    # Define a test query focused on VR fitness
    query: str = (
        "Develop a VR fitness platform with real-time coaching features "
        "and personalized workout plans to improve user engagement."
    )

    # Step 1: Build LoRA RetrievalQA chain to extract retriever only
    try:
        lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")
        retriever: BaseRetriever = lora_rag.retriever  # type: ignore[attr-defined]
        logger.info("Successfully instantiated RetrievalQA chain with LoRA.")
    except Exception as e:
        logger.error(f"Failed to build RetrievalQA chain: {e}")
        return

    # Step 2: Build the eval callable using LoRA weights
    try:
        lora_eval: Callable[[Dict[str, str]], Dict[str, Any]] = build_chain(
            kind="eval",
            store="chroma",
        )
        logger.info("Successfully built LoRA eval callable.")
    except Exception as e:
        logger.error(f"Failed to build LoRA eval callable: {e}")
        return

    # Step 3: Retrieve and log top-3 documents
    try:
        top_docs: List[Any] = retriever.get_relevant_documents(query)  # type: ignore[attr-defined]
        logger.info("STEP 1 · Top-3 Retrieved Documents")
        for idx, doc in enumerate(top_docs[:3], start=1):
            # Flatten newlines for logging
            doc_text: str = doc.page_content.strip().replace("\n", " ")
            logger.info(f"[{idx}] {doc_text}")
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return

    # Step 4: Generate model input and output using LoRA eval callable
    try:
        eval_result: Dict[str, Any] = lora_eval({"question": query})
        context_text: str = eval_result.get("context", "")
        snippet_text: str = eval_result.get("snippet", "")

        # Load LoRA model and tokenizer once for prompt extraction
        try:
            model, tokenizer = load_llama(use_lora=True)  # type: ignore[arg-type]
            llm_pipeline: HuggingFacePipeline = HuggingFacePipeline(
                pipeline=transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            )
        except Exception as e:
            logger.error(f"Failed to load LoRA model or tokenizer: {e}")
            return

        prompt_text, model_output = _extract_prompt_and_response(
            llm_pipeline=llm_pipeline,
            summary=query,
            context=context_text,
        )

        # Log the prompt and final output
        logger.info("STEP 2 · LLM Model Output")
        if model_output:
            logger.info(model_output.strip())
        else:
            logger.error("Model output is empty.")
    except Exception as e:
        logger.error(f"Exception during LoRA eval execution: {e}")
        return


if __name__ == "__main__":
    main()
