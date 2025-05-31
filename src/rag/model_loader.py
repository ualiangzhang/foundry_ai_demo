# src/rag/model_loader.py

from pathlib import Path
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

# Use plain strings when passing to HF .from_pretrained()
BASE_DIR = Path("models/base/Meta-Llama-3-8B-Instruct")
ADAPT_DIR = Path("models/adapters/llama3_lora")

def load_llama(device_map="auto", four_bit=True):
    # Cast Path to str before feeding into HuggingFace
    base_str  = str(BASE_DIR.resolve())
    adapt_str = str(ADAPT_DIR.resolve())

    # 1) Load tokenizer from local directory
    tok = LlamaTokenizer.from_pretrained(base_str)

    # 2) Load model (4-bit or full) from same directory
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map
    }
    if four_bit:
        model_kwargs["load_in_4bit"] = True

    model = LlamaForCausalLM.from_pretrained(base_str, **model_kwargs)

    # 3) If LoRA adapter exists, merge it
    if ADAPT_DIR.exists():
        model = PeftModel.from_pretrained(model, adapt_str)
        model.merge_and_unload()
        print("✓  LoRA merged")

    model.eval()
    return model, tok
