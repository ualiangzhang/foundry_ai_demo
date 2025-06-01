#!/usr/bin/env python3
"""
src/rag/model_loader.py

Provides functions to load the LLaMA-3 (8B) base model and optionally merge
a LoRA adapter for fine-tuning. Supports 4-bit quantization via BitsAndBytesConfig.

Functions:
    - load_llama: Load the base LLaMA-3 model and tokenizer, optionally merging LoRA weights.

Usage:
    from src.rag.model_loader import load_llama
    model, tokenizer = load_llama(device_map="auto", four_bit=True, use_lora=True)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Path to base and adapter directories ────────────────────────────────────
BASE_DIR: Path = Path("models/base/Meta-Llama-3-8B-Instruct")
ADAPT_DIR: Path = Path("models/adapters/llama3_lora")


def load_llama(
    device_map: Union[str, dict] = "auto",
    four_bit: bool = True,
    use_lora: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Load the LLaMA-3 (8B) model and tokenizer. Optionally apply a LoRA adapter.

    This function performs the following steps:
      1. Load a fast tokenizer for the LLaMA-3 model from BASE_DIR.
      2. Configure quantization: 4-bit via BitsAndBytesConfig if four_bit=True,
         otherwise load in bfloat16.
      3. Load the base LLaMA-3 model with the specified device_map.
      4. If use_lora=True and the adapter directory exists, load and merge the LoRA adapter.

    Args:
        device_map: How to map model layers to devices. Can be "auto" or a dictionary
                    mapping layer names to device IDs.
        four_bit: Whether to load the model in 4-bit quantized mode. If False,
                  model is loaded in bfloat16 (bf16).
        use_lora: If True and a LoRA adapter is found in ADAPT_DIR, merge its weights
                  into the base model.

    Returns:
        A tuple (model, tokenizer):
          - model: A transformers.PreTrainedModel (LlamaForCausalLM) set to evaluation mode.
          - tokenizer: A transformers.PreTrainedTokenizerFast for tokenization.

    Raises:
        FileNotFoundError: If BASE_DIR does not exist or missing model files.
        RuntimeError: If loading model/tokenizer or merging LoRA fails.
    """
    base_path = BASE_DIR.resolve()
    adapt_path = ADAPT_DIR.resolve()

    # 1) Verify base directory exists
    if not base_path.exists():
        msg = f"Base model directory '{base_path}' not found."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # 2) Load tokenizer
    try:
        logger.info(f"Loading tokenizer from '{base_path}'...")
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            str(base_path),
            use_fast=True,
            legacy=False
        )
    except Exception as e:
        msg = f"Failed to load tokenizer from '{base_path}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    # 3) Prepare model loading kwargs
    model_kwargs: dict = {"device_map": device_map}
    if four_bit:
        # Use 4-bit quantization with bfloat16 compute dtype
        logger.info("Configuring model for 4-bit quantization (bnb)...")
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["quantization_config"] = bnb_cfg
        except Exception as e:
            msg = f"Failed to create BitsAndBytesConfig for 4-bit: {e}"
            logger.error(msg)
            raise RuntimeError(msg)
    else:
        # Use bfloat16 without quantization
        logger.info("Configuring model to use torch_dtype=bfloat16...")
        model_kwargs["torch_dtype"] = torch.bfloat16

    # 4) Load base LLaMA-3 model
    try:
        logger.info(f"Loading LLaMA-3 base model from '{base_path}'...")
        model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            str(base_path),
            trust_remote_code=True,
            **model_kwargs
        )
    except Exception as e:
        msg = f"Failed to load LLaMA-3 from '{base_path}': {e}"
        logger.error(msg)
        raise RuntimeError(msg)

    # 5) Optionally merge LoRA adapter
    if use_lora:
        if adapt_path.exists():
            try:
                logger.info(f"Loading LoRA adapter from '{adapt_path}' and merging...")
                adapter_model = PeftModel.from_pretrained(model, str(adapt_path))
                adapter_model.merge_and_unload()
                logger.info("✓  LoRA weights merged successfully.")
            except Exception as e:
                msg = f"Failed to load or merge LoRA adapter from '{adapt_path}': {e}"
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            logger.warning(f"LoRA adapter directory '{adapt_path}' not found; skipping LoRA merge.")

    # 6) Set model to evaluation mode
    model.eval()
    return model, tokenizer
