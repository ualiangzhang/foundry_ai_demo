#!/usr/bin/env python3
"""
src/rag/prompts.py

Defines ChatPromptTemplate instances for various retrieval-augmented generation (RAG)
startup tasks using LLaMA 3. Each template structures a system+user conversation for
LangChain's ChatPromptTemplate.

Templates
---------
- RAG_WRAPPER: Generic context-question prompt for knowledge-grounded Q&A.
- PROJECT_EVAL: Matches the SFT training format (veteran VC partner giving four
  numbered recommendations).
- PITCH_DECK: Generates concise bullet points for standard pitch-deck slides.

Usage:
    from src.rag.prompts import PROJECT_EVAL
    messages = PROJECT_EVAL.format_prompt(context="…", question="…").to_messages()
"""

import logging
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── Generic RAG wrapper ────────────────────────────────
SYSTEM_STARTUP: str = (
    "You are a seasoned startup mentor. Use the CONTEXT below to answer "
    "the user's question. If the context is insufficient, say so instead of guessing."
)

try:
    RAG_WRAPPER: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_STARTUP),
        ("user", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ])
    logger.info("RAG_WRAPPER prompt template initialized successfully.")
except Exception as e:
    logger.error(f"Failed to create RAG_WRAPPER ChatPromptTemplate: {e}")
    raise

# ─────────────────────── Project-evaluation prompt (SFT-aligned) ────────────
PROJECT_EVAL_SYSTEM: str = (
    "You are a veteran VC partner. Using the provided summary and context,\n"
    "produce exactly FOUR recommendations covering Market, Product,\n"
    "Business Model, and Team. Each bullet ≤50 words, no questions.\n"
    "If context or summary is missing, reply exactly INSUFFICIENT_CONTEXT."
)
PROJECT_EVAL_USER: str = "### Startup summary\n{question}\n\n### Market context\n{context}"

try:
    PROJECT_EVAL: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", PROJECT_EVAL_SYSTEM),
        ("user", PROJECT_EVAL_USER),
    ])
    logger.info("PROJECT_EVAL prompt template initialized successfully.")
except Exception as e:
    logger.error(f"Failed to create PROJECT_EVAL ChatPromptTemplate: {e}")
    raise

# ─────────────────────── Pitch-deck generation prompt ───────────────────────
PITCH_DECK_SYSTEM: str = "You write concise bullet points for startup pitch-deck slides."
PITCH_DECK_USER: str = (
    "### Venture:\n{question}\n\n### Research snippets:\n{context}\n\n"
    "Generate slides for Problem, Solution, Market, Business Model, Team."
)

try:
    PITCH_DECK: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", PITCH_DECK_SYSTEM),
        ("user", PITCH_DECK_USER),
    ])
    logger.info("PITCH_DECK prompt template initialized successfully.")
except Exception as e:
    logger.error(f"Failed to create PITCH_DECK ChatPromptTemplate: {e}")
    raise


def get_all_prompts() -> Dict[str, ChatPromptTemplate]:
    """Return a dictionary of all available prompt templates.

    Returns:
        Dict[str, ChatPromptTemplate]: Mapping of template names to ChatPromptTemplate instances:
            - "rag": RAG_WRAPPER
            - "eval": PROJECT_EVAL
            - "pitch": PITCH_DECK
    """
    return {
        "rag": RAG_WRAPPER,
        "eval": PROJECT_EVAL,
        "pitch": PITCH_DECK,
    }
