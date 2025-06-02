#!/usr/bin/env python3
"""
tests/test_rag.py

Demonstrates the LoRA-fine-tuned LLaMA-3 model on a single query.
Steps:
  1. Build a RetrievalQA chain to obtain a shared retriever.
  2. Build the eval callable (four recommendations) using LoRA weights only.
  3. Output the top-3 retrieved documents.
  4. Display the LoRA model’s input prompt and its resulting output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

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
from src.rag.prompts import PROJECT_EVAL
from src.rag.chains import _build_retriever  # noqa: WPS430

# ── Logger configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_prompt_and_response(
        llm_pipeline: HuggingFacePipeline,
        summary: str,
        context: str,
) -> tuple[str, str]:
    """
    Build the input prompt for PROJECT_EVAL and invoke the LLM pipeline.

    Args:
        llm_pipeline: A HuggingFacePipeline instance wrapping the LoRA model.
        summary: The startup idea summary to include as "question".
        context: The market context to include as "context".

    Returns:
        A tuple containing:
            - The full text prompt sent to the model as a string.
            - The model's raw textual output as a string.
    """
    # Format the ChatPromptValue for PROJECT_EVAL
    prompt_value = PROJECT_EVAL.format_prompt(question=summary, context=context)
    # Convert to a single string (system + user) for the pipeline
    prompt_str = prompt_value.to_string()
    # Invoke the LLM pipeline and capture raw output
    raw_output = llm_pipeline.invoke(prompt_str).strip()
    return prompt_str, raw_output


def main() -> None:
    """
    1. Instantiate RetrievalQA with LoRA chain to obtain shared retriever.
    2. Instantiate the eval callable (LoRA-based) to produce four recommendations.
    3. Print the top-3 retrieved documents from the retriever.
    4. Generate and display the LoRA prompt and the model’s output.
    """
    # Define a test query focused on VR fitness
    query = (
        "Develop a VR fitness platform with real-time coaching features "
        "and personalized workout plans to improve user engagement."
    )

    # Step 1: Build LoRA RetrievalQA chain to extract retriever only
    lora_rag: RetrievalQA = build_chain(kind="rag", store="chroma")
    retriever: BaseRetriever = lora_rag.retriever

    # Step 2: Build the eval callable using LoRA weights
    lora_eval: Callable[[Dict[str, str]], Dict[str, Any]] = build_chain(
        kind="eval",
        store="chroma",
    )

    # Step 3: Retrieve and print top-3 documents
    top_docs = retriever.get_relevant_documents(query)
    print("\n" + "=" * 80)
    print("STEP 1 · Top-3 Retrieved Documents")
    print("=" * 80)
    for idx, doc in enumerate(top_docs[:3], start=1):
        # Flatten newlines for display
        doc_text: str = doc.page_content.strip().replace("\n", " ")
        print(f"[{idx}] {doc_text}\n")

    # Step 4: Generate model input and output using LoRA eval callable
    # First, run the eval callable to get result, context, snippet, docs
    eval_result = lora_eval({"question": query})

    # Extract context and snippet from eval_result for prompt construction
    context_text: str = eval_result.get("context", "")
    snippet_text: str = eval_result.get("snippet", "")

    # Build the LoRA input prompt and capture the raw model output
    prompt_text, model_output = _extract_prompt_and_response(
        llm_pipeline=HuggingFacePipeline(
            pipeline=transformers.pipeline(
                "text-generation",
                model=load_llama(use_lora=True)[0],
                tokenizer=load_llama(use_lora=True)[1],
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.1,
            )
        ),
        summary=query,
        context=context_text,
    )

    # Display the prompt and the final output
    print("\n" + "=" * 80)
    print("STEP 2 · LLM Model Input Prompt")
    print("=" * 80)
    print(prompt_text + "\n")

    print("\n" + "=" * 80)
    print("STEP 3 · LLM Model Output")
    print("=" * 80)
    print(model_output.strip() + "\n")


if __name__ == "__main__":
    main()
