# tests/test_rag.py

import sys
from pathlib import Path

# ─── Add project root (one level up from tests/) to Python path ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Core imports ─────────────────────────────────────────────────────────────
from src.rag.chains import build_chain  # RetrievalQA chain with LoRA merged by default
from src.rag.model_loader import load_llama
from src.rag.prompts import PROJECT_EVAL

import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

def make_base_chain(retriever, max_new_tokens=512, temperature=0.0):
    """
    Load the base (no-LoRA) LLaMA-3 model and wrap it in a RetrievalQA chain.
    """
    # 1) Load base model + tokenizer without LoRA
    base_model, base_tok = load_llama(use_lora=False)

    # 2) Build a HuggingFace pipeline for deterministic text generation
    pipe = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=base_tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.1
    )
    base_llm = HuggingFacePipeline(pipeline=pipe)

    # 3) Build a RetrievalQA chain using the same retriever and prompt
    base_chain = RetrievalQA.from_chain_type(
        llm=base_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROJECT_EVAL}
    )
    return base_chain

def make_lora_chain(store="chroma"):
    """
    Load the LoRA-merged LLaMA-3 and create a RetrievalQA chain exactly as in training.
    """
    # build_chain() uses load_llama(use_lora=True) internally
    return build_chain(kind="eval", store=store)

if __name__ == "__main__":
    # ─── Query to compare ────────────────────────────────────────────────────────
    query = (
        "Our startup produces mushroom-based leather. "
        "Could you critique our go-to-market plan?"
    )

    # ─── 1) Build the LoRA-fine-tuned chain ─────────────────────────────────────
    lora_chain = make_lora_chain(store="chroma")

    # ─── 2) Extract the retriever from the LoRA chain ───────────────────────────
    retriever = lora_chain.retriever

    # ─── 3) Build the "base" chain using the same retriever but no LoRA ──────────
    base_chain = make_base_chain(retriever)

    # ─── 4) Retrieve the top-3 docs (identical for both models) ─────────────────
    top_docs = retriever.get_relevant_documents(query)[:3]

    # ─── 5) Print the top-3 retrieved snippets once ──────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Top-3 retrieved snippets (shared by both models)")
    print("=" * 80)
    for i, doc in enumerate(top_docs, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"Snippet {i}: {snippet}\n")

    # ─── 6) Generate and print the base-model response ───────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: Base LLaMA-3 (no LoRA) response")
    print("=" * 80)
    base_result = base_chain({"query": query})
    print(base_result["result"].strip())

    # ─── 7) Generate and print the LoRA-fine-tuned response ──────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: LoRA-Fine-Tuned LLaMA-3 response")
    print("=" * 80)
    lora_result = lora_chain({"query": query})
    print(lora_result["result"].strip())

    print("\n" + "=" * 80)
    print("Comparison complete.\n")
