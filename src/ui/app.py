#!/usr/bin/env python3
"""
Streamlit UI for the Startup Evaluator & Web QA.

2025-06-02
* Adds a “Web QA” tab alongside the existing “Startup Evaluator” tab.
* Each tab has its own input fields and buttons.
* QA tab uses SerpApi + OpenAI GPT-4o-mini to answer questions using top-3 snippets.
* All comments and interface text are in professional, clear English.
"""

from __future__ import annotations

###############################################################################
# Early monkey‐patch so Streamlit’s watchdog doesn’t inspect PyTorch internals #
###############################################################################
import types, sys, os
_dummy = types.ModuleType("torch.classes"); _dummy.__path__ = []  # type: ignore[attr-defined]
sys.modules["torch.classes"] = _dummy
os.environ["STREAMLIT_WATCHDOG_IGNORE_DIRS"] = "torch"
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch"

###############################################################################
# Standard imports                                                            #
###############################################################################
import logging, re
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# ── FIRST Streamlit command must be `set_page_config` ────────────────────────
st.set_page_config(page_title="Evaluator & Web QA", layout="centered")

###############################################################################
# PYTHONPATH adjustment so `src.*` imports work                                #
###############################################################################
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.rag.chains import build_chain  # noqa: E402

###############################################################################
# Logger configuration                                                         #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

###############################################################################
# Utility functions                                                           #
###############################################################################
def postprocess_recommendations(raw: str) -> str:
    """
    Return only the Market-context paragraph and the four recommendation bullets.

    Steps
    -----
    1. Remove any lines starting with 'System:' or 'Human:'.
    2. Discard 'INSUFFICIENT_CONTEXT' if present.
    3. Keep everything from the first '### Market context' onward; drop the rest.
    4. If “[END]” appears after Team:, truncate and discard it and anything following.
    5. Strip trailing hashtags on the Team bullet.
    6. Escape underscores so Streamlit Markdown does not interpret them as italics.

    Parameters
    ----------
    raw : str
        Original text returned by the evaluator chain.

    Returns
    -------
    str
        Cleaned text, ready to pass to st.markdown().
    """
    # Normalize newlines
    txt = raw.replace("\r\n", "\n")

    # 1) Remove scaffolding lines
    txt = "\n".join(
        line for line in txt.splitlines()
        if not re.match(r"^\s*(System|Human)\s*:", line, flags=re.I)
    )

    # 2) Drop "INSUFFICIENT_CONTEXT"
    txt = txt.replace("INSUFFICIENT_CONTEXT", "")

    # 3) Keep from "### Market context" onward
    match = re.search(r"(?m)^###\s*Market\s+context", txt)
    if match:
        txt = txt[match.start():]
    else:
        # If "### Market context" not found, return empty
        return ""

    # 4) Truncate at first "[END]" if present
    end_index = txt.find("[END]")
    if end_index != -1:
        txt = txt[:end_index]

    # 5) Strip trailing hashtags after Team:
    txt = re.sub(
        r"(Team:\s*.+?)(?:\s*#.*)$",
        r"\1",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 6) Escape underscores to prevent Markdown italics
    txt = txt.replace("_", r"\_")

    return txt.strip()


###############################################################################
# Cache and build the “eval” and “qa” chains                                   #
###############################################################################
@st.cache_resource
def get_eval_chain() -> Any:
    """
    Build and cache the evaluation chain (VC recommendations).
    Returns a callable that expects {"question": <startup summary>}.
    """
    return build_chain(kind="eval", store="chroma")


@st.cache_resource
def get_qa_chain() -> Any:
    """
    Build and cache the web QA chain (SerpApi + OpenAI).
    Returns a callable that expects {"question": <user question>}.
    """
    return build_chain(kind="qa", store="chroma")


eval_chain: Any = get_eval_chain()
qa_chain: Any = get_qa_chain()

###############################################################################
# Streamlit UI                                                                #
###############################################################################
st.title("🚀 Startup Evaluator & Web QA")

# Create two tabs: one for startup evaluation, one for web-based QA
tabs = st.tabs(["Startup Evaluator", "Web QA"])

# ── Tab 1: Startup Evaluator -----------------------------------------------
with tabs[0]:
    st.header("Startup Evaluation")
    st.write(
        "Enter a concise startup summary. The system will fetch market context "
        "and provide four VC-style recommendations."
    )

    summary: str = st.text_area(
        label="Startup Summary",
        height=120,
        placeholder=(
            "e.g., A VR fitness platform with real-time coaching features and personalised workout plans…"
        ),
    )

    if st.button("Evaluate Startup", key="eval_button"):
        if not summary.strip():
            st.error("❗ Please enter a non-empty startup summary.")
        else:
            try:
                # Invoke the evaluation chain
                output: Dict[str, Any] = eval_chain({"question": summary})

                # 1) Three Similar Startup Examples
                st.subheader("📄 Three Similar Startup Examples")
                retrieved: List[str] = output.get("docs", [])
                if retrieved:
                    for idx, doc_text in enumerate(retrieved, start=1):
                        words: List[str] = doc_text.strip().split()
                        truncated: str = (
                            " ".join(words[:200]) + " …" if len(words) > 200 else " ".join(words)
                        )
                        with st.expander(f"Example {idx}", expanded=False):
                            st.write(truncated)
                else:
                    st.info("No similar startup examples retrieved.")

                # 2) VC Recommendations
                st.subheader("💡 VC Recommendations")
                raw_recs: str = output.get("result", "")
                cleaned_recs = postprocess_recommendations(raw_recs)
                if cleaned_recs:
                    st.markdown(cleaned_recs)
                else:
                    st.info("No recommendations generated by the model.")

            except Exception as exc:  # noqa: BLE001
                logger.exception("Evaluator chain failed: %s", exc)
                st.error(f"Evaluation failed: {exc}")

# ── Tab 2: Web QA ----------------------------------------------------------
with tabs[1]:
    st.header("Web-based Question Answering")
    st.write(
        "Enter any factual question. The system will fetch the top-3 web snippets "
        "and then use OpenAI’s ChatGPT-4o-mini to provide a concise answer (≤200 words)."
    )

    question: str = st.text_area(
        label="Your Question",
        height=120,
        placeholder="e.g., What is CRISPR gene editing?",
    )

    if st.button("Get Answer", key="qa_button"):
        if not question.strip():
            st.error("❗ Please enter a non-empty question.")
        else:
            try:
                # Invoke the QA chain
                output: Dict[str, Any] = qa_chain({"question": question})

                # Display the concatenated context from web snippets
                st.subheader("🔍 Retrieved Web Snippets Context")
                context: str = output.get("context", "")
                if context:
                    st.write(context)
                else:
                    st.info("No relevant web snippets found.")

                # Display the final answer from ChatGPT-4o-mini
                st.subheader("🤖 Answer")
                answer: str = output.get("answer", "")
                if answer:
                    st.write(answer)
                else:
                    st.info("Failed to generate an answer.")

            except Exception as exc:  # noqa: BLE001
                logger.exception("QA chain failed: %s", exc)
                st.error(f"QA failed: {exc}")
