# src/rag/chains.py
import transformers
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from .model_loader import load_llama
from .retriever import chroma_retriever, qdrant_retriever
from .prompts import RAG_WRAPPER, PROJECT_EVAL, PITCH_DECK

def _make_llm(max_new_tokens=512, temperature=0.2):
    model, tok = load_llama()
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)

def build_chain(kind: str = "eval", store: str = "chroma"):
    retriever = chroma_retriever() if store == "chroma" else qdrant_retriever()
    prompt = {"eval": PROJECT_EVAL,
              "pitch": PITCH_DECK,
              "rag": RAG_WRAPPER}[kind]

    return RetrievalQA.from_chain_type(
        llm=_make_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
