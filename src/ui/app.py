#!/usr/bin/env python3
"""
src/ui/app.py

Streamlit UI for the Startup Evaluator & Web QA.
"""

from __future__ import annotations

###############################################################################
# Early monkey‐patch so Streamlit’s watchdog doesn’t inspect PyTorch internals #
###############################################################################
import types
import sys
import os

_dummy: types.ModuleType = types.ModuleType("torch.classes")  # type: ignore[attr-defined]
_dummy.__path__ = []  # type: ignore[attr-defined]
sys.modules["torch.classes"] = _dummy
os.environ["STREAMLIT_WATCHDOG_IGNORE_DIRS"] = "torch"
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch"

###############################################################################
# Standard imports                                                            #
###############################################################################
import logging
import re
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
logger: logging.Logger = logging.getLogger(__name__)


###############################################################################
# Utility functions                                                           #
###############################################################################
def postprocess_recommendations(raw: str) -> str:
    """
    Return only the Market-context paragraph and the four recommendation bullets.

    This function performs the following steps:
      1. Remove any lines starting with 'System:' or 'Human:' (case‐insensitive).
      2. Discard the literal 'INSUFFICIENT_CONTEXT' if present.
      3. Keep everything from the first occurrence of '### Market context' onward;
         drop any preceding text.
      4. If '[END]' appears anywhere after '### Market context', truncate at that marker
         and discard anything that follows.
      5. On the Team bullet, strip trailing hashtags (e.g. '#Finance', '#Startups').
      6. Escape underscores (_) so that Streamlit Markdown does not interpret them as italics.

    Args:
        raw (str): Original multi‐line string returned by the evaluator chain.

    Returns:
        str: Cleaned markdown‐safe string containing the market context and four bullets.
             Returns an empty string if no '### Market context' marker is found.
    """
    # Normalize newlines to '\n'
    txt: str = raw.replace("\r\n", "\n")

    # 1) Remove lines starting with 'System:' or 'Human:' (case‐insensitive)
    txt = "\n".join(
        line
        for line in txt.splitlines()
        if not re.match(r"^\s*(System|Human)\s*:", line, flags=re.IGNORECASE)
    )

    # 2) Discard 'INSUFFICIENT_CONTEXT'
    txt = txt.replace("INSUFFICIENT_CONTEXT", "")

    # 3) Keep text from '### Market context' onward
    match = re.search(r"(?m)^###\s*Market\s+context", txt)
    if match:
        txt = txt[match.start():]
    else:
        # If the marker is not found, return empty
        return ""

    # 4) Truncate at first '[END]', if present
    end_index: int = txt.find("[END]")
    if end_index != -1:
        txt = txt[:end_index]

    # 5) Strip trailing hashtags from the Team bullet
    txt = re.sub(
        r"(Team:\s*.+?)(?:\s*#.*)$",  # Capture 'Team: ...' up to the first hashtag
        r"\1",  # Keep only the part before hashtags
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

    Returns:
        Callable[[Dict[str, str]], Dict[str, Any]]:
            A function that expects a dict with key 'question' (startup summary)
            and returns a dict containing keys: 'result', 'context', 'snippet', 'docs'.
    """
    return build_chain(kind="eval", store="chroma")


@st.cache_resource
def get_qa_chain() -> Any:
    """
    Build and cache the web QA chain (SerpApi + OpenAI).

    Returns:
        Callable[[Dict[str, str]], Dict[str, Any]]:
            A function that expects a dict with key 'question' (user question)
            and returns a dict containing keys: 'answer', 'context'.
    """
    return build_chain(kind="qa", store="chroma")


eval_chain: Any = get_eval_chain()
qa_chain: Any = get_qa_chain()

###############################################################################
# Streamlit UI                                                                #
###############################################################################
st.title("🚀 Startup Evaluator & Web QA")

# Create two tabs: one for startup evaluation, one for web-based QA
tabs: List[Any] = st.tabs(["Startup Evaluator", "Web QA"])

# ── Tab 1: Startup Evaluator -----------------------------------------------
with tabs[0]:
    st.header("Startup Evaluation")
    st.write(
        "Enter a concise startup summary. The system will:\n"
        "1. Retrieve three similar examples from the local vector database.\n"
        "2. Fetch market background from the web.\n"
        "3. Use a fine-tuned LLaMA-3 8B Instruct model to summarize the market context.\n"
        "4. Use the same fine-tuned LLaMA-3 8B Instruct model to generate four VC-style recommendations."
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
                # Invoke the evaluation chain with {"question": summary}
                output: Dict[str, Any] = eval_chain({"question": summary})

                # 1) Display top-3 similar startup examples (from vector store)
                st.subheader("📄 Three Similar Startup Examples")
                retrieved: List[str] = output.get("docs", [])
                if retrieved:
                    for idx, doc_text in enumerate(retrieved, start=1):
                        words: List[str] = doc_text.strip().split()
                        truncated: str = (
                            " ".join(words[:200]) + " …"
                            if len(words) > 200
                            else " ".join(words)
                        )
                        with st.expander(f"Example {idx}", expanded=False):
                            st.write(truncated)
                else:
                    st.info("No similar startup examples retrieved.")

                # 2) Display VC recommendations
                st.subheader("💡 VC Recommendations")
                raw_recs: str = output.get("result", "")
                cleaned_recs: str = postprocess_recommendations(raw_recs)
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
                # Invoke the QA chain with {"question": question}
                output: Dict[str, Any] = qa_chain({"question": question})

                # Display concatenated context from web snippets
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
