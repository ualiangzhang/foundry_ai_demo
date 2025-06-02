#!/usr/bin/env python3
"""
Streamlit UI for the Startup Evaluator chain.

Changelog
---------
2025-06-02  • Remove Market-context display.
           • Extend post-processing to (a) drop all prompt scaffolding,
             (b) cut content preceding the first 'Market:' bullet,
             (c) strip 'INSUFFICIENT_CONTEXT'.
"""

from __future__ import annotations

###############################################################################
# Early monkey-patch to hide PyTorch internals from Streamlit’s watchdog      #
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

st.set_page_config(page_title="Startup Evaluator", layout="centered")

###############################################################################
# PYTHONPATH fix so `src.*` imports work                                      #
###############################################################################
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chains import build_chain  # noqa: E402

###############################################################################
# Logger                                                                      #
###############################################################################
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

###############################################################################
# Helpers                                                                     #
###############################################################################
def postprocess_recommendations(raw: str) -> str:
    """
    Return only the four cleaned recommendation bullets.

    1. Drop lines starting with 'System:' or 'Human:'.
    2. Remove the token 'INSUFFICIENT_CONTEXT' if present.
    3. Cut everything *before* the first 'Market:' bullet.
    4. Trim trailing hashtags on the 'Team:' bullet.
    """
    # Normalise newlines
    txt = raw.replace("\r\n", "\n")

    # 1 ▸ remove scaffolding lines
    txt = "\n".join(
        ln for ln in txt.splitlines()
        if not re.match(r"^\s*(System|Human):", ln, flags=re.I)
    )

    # 2 ▸ drop INSUFFICIENT_CONTEXT
    txt = txt.replace("INSUFFICIENT_CONTEXT", "")

    # 3 ▸ keep from first 'Market:' onward
    m = re.search(r"(?im)^\s*Market:\s*", txt)
    if m:
        txt = txt[m.start():]

    # 4 ▸ cut hashtags after Team bullet
    txt = re.sub(r"(Team:\s*.+?)(?:\s*#.*)$",
                 r"\1", txt, flags=re.I | re.DOTALL)

    return txt.strip()

###############################################################################
# Cache the evaluation chain                                                  #
###############################################################################
@st.cache_resource
def get_eval_chain() -> Any:
    return build_chain(kind="eval", store="chroma")

eval_chain: Any = get_eval_chain()

###############################################################################
# UI                                                                          #
###############################################################################
st.title("🚀 Startup Evaluator")

summary: str = st.text_area(
    "Enter your startup summary (idea)",
    height=120,
    placeholder=("e.g., A VR fitness platform with real-time coaching features "
                 "and personalised workout plans…"),
)

if st.button("Evaluate Startup"):
    if not summary.strip():
        st.error("❗ Please enter a non-empty startup summary.")
    else:
        try:
            output: Dict[str, Any] = eval_chain({"question": summary})

            # 1 ▸ Similar examples
            st.subheader("📄 Three Similar Startup Examples")
            for idx, doc in enumerate(output.get("docs", []), 1):
                words = doc.strip().split()
                snippet = " ".join(words[:200]) + (" …" if len(words) > 200 else "")
                with st.expander(f"Example {idx}", expanded=False):
                    st.write(snippet)
            if not output.get("docs"):
                st.info("No similar startup examples retrieved.")

            # 2 ▸ VC recommendations (only cleaned bullets)
            st.subheader("💡 VC Recommendations")
            recs = postprocess_recommendations(output.get("result", ""))
            if recs:
                st.markdown(recs)
            else:
                st.info("No recommendations generated by the model.")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluator chain failed: %s", exc)
            st.error(f"Evaluation failed: {exc}")
