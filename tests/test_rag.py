# tests/test_rag.py

import sys
from pathlib import Path

# ─── Add project root (one level up from tests/) to Python path ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Core imports ─────────────────────────────────────────────────────────────
from src.rag.chains import build_chain  # gives us RetrievalQA with LoRA merged
from src.rag.model_loader import load_llama
from src.rag.prompts import PROJECT_EVAL
import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

def make_base_chain(retriever, max_new_tokens=512, temperature=0.0):
    """
    Load the *base* (no-LoRA) Llama-3 model and wrap it in a RetrievalQA chain.
    """
    # 1) Load base model + tokenizer without merging any LoRA
    base_model, base_tok = load_llama(use_lora=False)

    # 2) Build a HuggingFace pipeline for text-generation
    pipe = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=base_tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,           # deterministic for comparison
        repetition_penalty=1.1,
    )
    base_llm = HuggingFacePipeline(pipeline=pipe)

    # 3) Build a RetrievalQA chain wrapping our base_llm and the same retriever
    base_chain = RetrievalQA.from_chain_type(
        llm=base_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROJECT_EVAL}
    )
    return base_chain

def make_lora_chain(store="chroma"):
    """
    Load the LoRA-merged Llama-3 + create a RetrievalQA chain (identical to training).
    """
    return build_chain(kind="eval", store=store)

if __name__ == "__main__":
    # ─── Query to evaluate ────────────────────────────────────────────────────
    query = (
        "Our startup produces mushroom-based leather. "
        "Could you critique our go-to-market plan?"
    )

    # ─── 1) Build the LoRA-fine-tuned chain (retriever + LoRA-merged LLM) ──────
    lora_chain = make_lora_chain(store="chroma")

    # ─── 2) Extract its Chroma retriever so we can reuse it for the base model ─
    retriever = lora_chain.retriever

    # ─── 3) Build a “base” chain that uses the exact same retriever but no LoRA ─
    base_chain = make_base_chain(retriever)

    # ─── 4) Fetch the top-3 retrieved documents (same for both base and LoRA) ───
    top_docs = retriever.get_relevant_documents(query)
    # We’ll only print the first 3 snippets (they’re ordered by relevance)
    top_docs = top_docs[:3]

    # ─── 5) Generate answers from both models ───────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Show the top-3 retrieved snippets (identical for both models)")
    print("=" * 80)
    for i, doc in enumerate(top_docs, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"Snippet {i}: {snippet}\n")

    print("\n" + "=" * 80)
    print("STEP 2: Base LLaMA-3 (no LoRA) response")
    print("=" * 80)
    base_result = base_chain({"query": query})
    # RetrievalQA stores the answer under “result”
    print(base_result["result"].strip())

    print("\n" + "=" * 80)
    print("STEP 3: LoRA-Fine-Tuned LLaMA-3 response")
    print("=" * 80)
    lora_result = lora_chain({"query": query})
    print(lora_result["result"].strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")
