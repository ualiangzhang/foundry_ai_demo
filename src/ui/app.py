# src/ui/app.py
import streamlit as st
from src.rag.chains import build_chain

@st.cache_resource
def chains():
    return dict(
        eval=build_chain("eval"),
        pitch=build_chain("pitch"),
        rag=build_chain("rag")
    )

tab1, tab2, tab3 = st.tabs(["Evaluator", "Pitch-deck", "Generic RAG"])

with tab1:
    summ = st.text_area("Startup summary")
    if st.button("Evaluate"):
        st.write(chains()["eval"](summ)["result"])

with tab2:
    summ = st.text_area("Startup summary for deck generation")
    if st.button("Generate slides"):
        st.write(chains()["pitch"](summ)["result"])

with tab3:
    q = st.text_input("Ask any question")
    if st.button("Ask"):
        st.write(chains()["rag"](q)["result"])
