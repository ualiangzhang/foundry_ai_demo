# src/rag/model_loader.py

from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

# ─── Path to base and adapter directories ────────────────────────────────────
BASE_DIR  = Path("models/base/Meta-Llama-3-8B-Instruct")
ADAPT_DIR = Path("models/adapters/llama3_lora")

def load_llama(
    device_map="auto",
    four_bit=True,
    use_lora=True
):
    """
    Load LLaMA-3 (8B) model and tokenizer.
    If `use_lora=True` and LoRA adapter exists, merge LoRA weights.

    Args:
      device_map (str or dict): how to map layers to GPUs/CPUs (e.g. "auto").
      four_bit (bool): whether to load in 4-bit quantization via BitsAndBytesConfig.
      use_lora (bool): if False, skip applying the LoRA adapter.

    Returns:
      model (transformers.PreTrainedModel), tok (transformers.PreTrainedTokenizerFast)
    """
    base_path  = str(BASE_DIR.resolve())
    adapt_path = str(ADAPT_DIR.resolve())

    # 1) Fast tokenizer (no SentencePiece C++ call)
    tok = AutoTokenizer.from_pretrained(
        base_path,
        use_fast=True,
        legacy=False
    )

    # 2) Build a BitsAndBytesConfig if we want 4-bit
    model_kwargs = {"device_map": device_map}
    if four_bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["quantization_config"] = bnb_cfg
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # 3) Load the base LLaMA-3 model
    model = LlamaForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        **model_kwargs
    )

    # 4) If use_lora=True and adapter folder exists, merge LoRA
    if use_lora and ADAPT_DIR.exists():
        adapter = PeftModel.from_pretrained(model, adapt_path)
        adapter.merge_and_unload()
        print("✓  LoRA merged")

    model.eval()
    return model, tok
