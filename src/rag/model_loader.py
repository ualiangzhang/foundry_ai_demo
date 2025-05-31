# src/rag/model_loader.py
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM

BASE_DIR   = Path("models/base/Meta-Llama-3-8B-Instruct")
ADAPT_DIR  = Path("models/adapters/llama3_lora")

def load_llama(device_map="auto", four_bit=True):
    base = str(BASE_DIR.resolve())
    adapt = str(ADAPT_DIR.resolve())

    # ---- FAST tokenizer: AutoTokenizer with use_fast=True --------------
    tok = AutoTokenizer.from_pretrained(
        base,
        use_fast=True,   # <-- forces LlamaTokenizerFast
        legacy=False     # suppresses legacy warning
    )

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map)
    if four_bit:
        model_kwargs["load_in_4bit"] = True

    model = LlamaForCausalLM.from_pretrained(base, **model_kwargs)

    if ADAPT_DIR.exists():
        model = PeftModel.from_pretrained(model, adapt)
        model.merge_and_unload()
        print("✓  LoRA merged")

    model.eval()
    return model, tok
