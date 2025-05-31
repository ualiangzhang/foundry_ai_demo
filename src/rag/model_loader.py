# src/rag/model_loader.py
from pathlib import Path
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

BASE = Path("models/base/Meta-Llama-3-8B-Instruct")
ADAPT = Path("models/adapters/llama3_lora")          # LoRA

def load_llama(device_map="auto", four_bit=True):
    tok = LlamaTokenizer.from_pretrained(BASE)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map)
    if four_bit:
        kwargs |= dict(load_in_4bit=True)
    model = LlamaForCausalLM.from_pretrained(BASE, **kwargs)

    if ADAPT.exists():
        model = PeftModel.from_pretrained(model, ADAPT)
        model.merge_and_unload()                     # merges LoRA into base
        print("✓  LoRA merged")

    model.eval()
    return model, tok
