# src/ui/app.py

"""Streamlit UI for three LLM chains: Evaluator, Pitch-deck, Generic RAG."""

from __future__ import annotations

###############################################################################
# Early monkey-patch: provide a dummy torch.classes module so Streamlit’s
# file-watcher never touches the lazy C++ proxy inside PyTorch.                #
###############################################################################
import types
import sys
import os

# Create a fake torch.classes module with an empty __path__
dummy_torch_classes = types.ModuleType("torch.classes")
dummy_torch_classes.__path__ = []  # type: ignore[attr-defined]
sys.modules["torch.classes"] = dummy_torch_classes

# Tell Streamlit’s watchdog to ignore anything under "torch"
os.environ["STREAMLIT_WATCHDOG_IGNORE_DIRS"] = "torch"
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch"

###############################################################################
# Standard imports                                                             #
###############################################################################
import logging
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import transformers
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# ── Ensure project root is on PYTHONPATH so that src modules can be imported ──
# If this file is at <project_root>/src/ui/app.py, then:
#   Path(__file__).parent           -> <project_root>/src/ui
#   Path(__file__).parent.parent    -> <project_root>/src
#   Path(__file__).parent.parent.parent -> <project_root>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chains import build_chain  # noqa: E402
from src.rag.model_loader import load_llama  # noqa: E402

###############################################################################
# Logger configuration                                                         #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

###############################################################################
# Build & cache the three chains                                               #
###############################################################################
@st.cache_resource
def get_chains() -> Dict[str, Any]:
    """
    Return cached instances of the eval, pitch, and rag chains.

    - "eval": Callable that runs DuckDuckGo + LLaMA-3 summarization → VC recommendations.
    - "pitch": RetrievalQA chain for pitch-deck bullet generation.
    - "rag": RetrievalQA chain for generic RAG QA.
    """
    return {
        "eval": build_chain(kind="eval", store="chroma"),
        "pitch": build_chain(kind="pitch", store="chroma"),
        "rag": build_chain(kind="rag", store="chroma"),
    }


chains: Dict[str, Any] = get_chains()

###############################################################################
# Streamlit UI                                                                 #
###############################################################################
tab1, tab2, tab3 = st.tabs(["Evaluator", "Pitch-deck", "Generic RAG"])

# ────────────────────────────── Tab 1: Evaluator ────────────────────────────
with tab1:
    st.header("Startup Evaluator")
    summary: str = st.text_area("Enter your startup summary (idea)", height=100)

    if st.button("Evaluate Startup"):
        if not summary.strip():
            st.error("Please enter a non-empty summary.")
        else:
            try:
                result: Dict[str, Any] = chains["eval"]({"question": summary})

                # 1. Display top-3 retrieved documents, if any
                st.subheader("Top-3 Retrieved Documents")
                docs: List[str] = result.get("docs", [])
                if docs:
                    for i, doc in enumerate(docs, start=1):
                        # Replace newline characters with spaces
                        safe_text: str = doc.strip().replace("\n", " ")
                        st.markdown("**Doc {}:** {}".format(i, safe_text))
                else:
                    st.info("No documents retrieved for this summary.")

                # 2. Display the DuckDuckGo snippet (numeric fact), if available
                st.subheader("Market Snippet")
                snippet: str = result.get("snippet", "")
                if snippet:
                    st.write(snippet)
                else:
                    st.info("No snippet with numeric data found.")

                # 3. Display the ~100-word market context, if generated
                st.subheader("Market Context (~100 words)")
                context: str = result.get("context", "")
                if context:
                    st.write(context)
                else:
                    st.info("No context generated by the model.")

                # 4. Display the four VC recommendations
                st.subheader("VC Recommendations")
                recommendations: str = result.get("result", "")
                if recommendations:
                    st.write(recommendations)
                else:
                    st.info("No recommendations generated by the model.")

            except Exception as exc:
                logger.exception("Evaluator failed: %s", exc)
                st.error(f"Evaluation failed: {exc}")

# ─────────────────────────── Tab 2: Pitch-deck ──────────────────────────────
with tab2:
    st.header("Pitch-deck Generator")
    deck_summary: str = st.text_area("Startup summary for deck generation", height=100)

    if st.button("Generate Pitch-deck"):
        if not deck_summary.strip():
            st.error("Please enter a non-empty summary.")
        else:
            try:
                chain: RetrievalQA = chains["pitch"]
                try:
                    deck_output: str = chain.run(deck_summary)  # type: ignore[attr-defined]
                except Exception:
                    deck_output = chain({"query": deck_summary})  # type: ignore[attr-defined]

                st.subheader("Pitch-deck Bullets")
                if deck_output:
                    st.write(deck_output)
                else:
                    st.info("No output generated for pitch deck.")

            except Exception as exc:
                logger.exception("Pitch-deck failed: %s", exc)
                st.error(f"Pitch-deck generation failed: {exc}")

# ─────────────────────────── Tab 3: Generic RAG ──────────────────────────────
with tab3:
    st.header("Generic RAG QA")
    question: str = st.text_input("Ask any question about your startup or market")

    if st.button("Get Answer"):
        if not question.strip():
            st.error("Please enter a non-empty question.")
        else:
            try:
                chain: RetrievalQA = chains["rag"]
                try:
                    answer: str = chain.run(question)  # type: ignore[attr-defined]
                except Exception:
                    answer = chain({"query": question})  # type: ignore[attr-defined]

                st.subheader("Answer")
                if answer:
                    st.write(answer)
                else:
                    st.info("No answer returned by the model.")

            except Exception as exc:
                logger.exception("RAG failed: %s", exc)
                st.error(f"RAG query failed: {exc}")
