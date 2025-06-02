#!/usr/bin/env python3
"""
Streamlit UI for the Startup Evaluator chain.

2025-06-02
* Drop all text before `### Market context` (removing System/Human scaffolding).
* Trim hashtags after “Team:” bullet.
* Escape underscores to prevent unintended Markdown italics.
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
st.set_page_config(page_title="Startup Evaluator", layout="centered")

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
    Return the Market-context paragraph and the four recommendation bullets, with:
      1. Everything before '### Market context' removed.
      2. Trailing hashtags after the Team bullet stripped.
      3. Underscores escaped to avoid Markdown italics.

    Parameters
    ----------
    raw : str
        Original text returned by the evaluator chain.

    Returns
    -------
    str
        Sanitised text, ready for st.markdown().
    """
    # Normalize newlines
    txt = raw.replace("\r\n", "\n")

    # 1) Drop everything before "### Market context"
    m = re.search(r"(?m)^###\s*Market\s+context", txt)
    if m:
        txt = txt[m.start():]
    else:
        # If for some reason "### Market context" isn't found, return empty
        return ""

    # 2) Trim trailing hashtags on the Team bullet
    txt = re.sub(
        r"(Team:\s*.+?)(?:\s*#.*)$",  # match from "Team:" up to first "#" (inclusive)
        r"\1",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 3) Escape underscores so Streamlit Markdown doesn't render them as italics
    txt = txt.replace("_", r"\_")

    return txt.strip()

###############################################################################
# Cache and build only the “eval” chain                                        #
###############################################################################
@st.cache_resource
def get_eval_chain() -> Any:
    """
    Build and cache the evaluation chain.

    Returns
    -------
    Any
        A callable that expects {"question": <startup summary>} and
        returns a dict with keys "docs", "context", and "result".
    """
    return build_chain(kind="eval", store="chroma")


eval_chain: Any = get_eval_chain()

###############################################################################
# Streamlit UI                                                                 #
###############################################################################
st.title("🚀 Startup Evaluator")

# ── Input area ---------------------------------------------------------------
summary: str = st.text_area(
    label="Enter your startup summary (idea)",
    height=120,
    placeholder=(
        "e.g., A VR fitness platform with real-time coaching features and personalised workout plans…"
    ),
)

# ── Evaluation button --------------------------------------------------------
if st.button("Evaluate Startup"):
    if not summary.strip():
        st.error("❗ Please enter a non-empty startup summary.")
    else:
        try:
            # Invoke the eval chain: returns a dict containing:
            #   "docs": List[str] (top-3 similar startup examples),
            #   "context": str (~100-word market context),
            #   "result": str (full raw output with scaffolding & recommendations)
            output: Dict[str, Any] = eval_chain({"question": summary})

            # ────────────────────────────────────────────────────────────────────
            # 1) Three Similar Startup Examples (truncate to 200 words each)
            # ────────────────────────────────────────────────────────────────────
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

            # ────────────────────────────────────────────────────────────────────
            # 2) VC Recommendations (Market-context + four bullets only)
            # ────────────────────────────────────────────────────────────────────
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
