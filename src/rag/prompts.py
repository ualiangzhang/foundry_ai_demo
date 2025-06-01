#!/usr/bin/env python3
"""
src/rag/prompts.py

Defines ChatPromptTemplate instances for various RAG-based startup tasks using
LLaMA 3.

Templates
---------
- **RAG_WRAPPER** – Generic context‑question prompt for knowledge‑grounded Q&A.
- **PROJECT_EVAL**  – Matches the SFT training format (veteran VC partner giving
  four numbered recommendations).
- **PITCH_DECK**   – Generates concise bullet points for standard pitch‑deck
  slides.

Example
-------
```python
from src.rag.prompts import PROJECT_EVAL
messages = PROJECT_EVAL.format_prompt(context="…", question="…").to_messages()
```
"""

import logging
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── Generic RAG wrapper ────────────────────────────────
SYSTEM_STARTUP = (
    "You are a seasoned startup mentor. Use the CONTEXT below to answer "
    "the user's question. If the context is insufficient, say so instead of guessing."
)

RAG_WRAPPER = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_STARTUP),
    ("user", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
])

# ─────────────────────── Project‑evaluation prompt (SFT‑aligned) ────────────
PROJECT_EVAL = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a veteran VC partner. Using ONLY the reference snippets, "
        "produce exactly FOUR numbered recommendations covering market, "
        "product, business model and team. Each bullet ≤ 50 words, no questions. "
        "If the snippets are empty, reply exactly `INSUFFICIENT_CONTEXT`.",
    ),
    (
        "user",
        "### Startup summary\n{question}\n\n### Reference snippets\n{context}",
    ),
])

# ─────────────────────── Pitch‑deck generation prompt ───────────────────────
PITCH_DECK = ChatPromptTemplate.from_messages([
    (
        "system",
        "You write concise bullet points for startup pitch‑deck slides.",
    ),
    (
        "user",
        "### Venture:\n{question}\n\n### Research snippets:\n{context}\n\n"
        "Generate slides for Problem, Solution, Market, Business Model, Team.",
    ),
])
