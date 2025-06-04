#!/usr/bin/env python3
"""
api_server.py

Minimal REST API for:
  • Startup Evaluation  → /evaluate
  • Web QA              → /qa

Both endpoints return JSON and rely on the chain builders defined in src/rag/chains.py.

Run locally:
---------------
# Activate your virtual environment first:
uvicorn api_server:app --host 0.0.0.0 --port 8000

Test endpoints:
---------------
curl -X POST http://localhost:8000/evaluate \
     -H "Content-Type: application/json" \
     -d '{"summary": "Develop a VR fitness platform ..."}'

curl -X POST http://localhost:8000/qa \
     -H "Content-Type: application/json" \
     -d '{"question": "What is CRISPR gene editing?"}'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure local package imports work
import sys
PROJECT_ROOT: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chains import build_chain  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Build chains once at startup (thread-safe, read-only afterward)
EVAL_CHAIN = build_chain(kind="eval", store="chroma")
QA_CHAIN = build_chain(kind="qa", store="chroma")

app = FastAPI(
    title="Startup Evaluator & Web QA API",
    description="Thin wrapper around local LLaMA-3 evaluation and SerpApi-powered QA.",
    version="1.0.0",
)


class EvaluateRequest(BaseModel):
    summary: str


class EvaluateResponse(BaseModel):
    result: str
    context: str
    snippet: str
    docs: list[str]


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    context: str


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    """
    Evaluate a startup idea and return market context + 4 VC recommendations.
    """
    summary = req.summary.strip()
    if not summary:
        raise HTTPException(status_code=400, detail="Summary must not be empty.")

    try:
        output = EVAL_CHAIN({"question": summary})
        return EvaluateResponse(**output)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> Dict[str, Any]:
    """
    Answer a factual question using web snippets + GPT-4o-mini.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        output = QA_CHAIN({"question": question})
        return QAResponse(**output)
    except Exception as exc:  # noqa: BLE001
        logging.exception("QA failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
