"""Shared model loading utilities for base model and LoRA adapters."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import LORA_CONFIG, MODEL_NAME, QUANT_CONFIG


def get_quant_config() -> BitsAndBytesConfig:
    """Build BitsAndBytesConfig from project defaults."""
    return BitsAndBytesConfig(
        load_in_4bit=QUANT_CONFIG["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=QUANT_CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=QUANT_CONFIG["bnb_4bit_quant_type"],
    )


def load_tokenizer(model_name: str = MODEL_NAME):
    """Load tokenizer and set pad token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    quant_config: BitsAndBytesConfig | None = None,
) -> AutoModelForCausalLM:
    """Load the base model with quantization."""
    if quant_config is None:
        quant_config = get_quant_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def load_adapter(
    base_model: AutoModelForCausalLM,
    adapter_path: str | Path,
) -> PeftModel:
    """Load a single LoRA adapter on top of the base model."""
    return PeftModel.from_pretrained(base_model, str(adapter_path))


def attach_lora(
    model: AutoModelForCausalLM,
    lora_config: dict | None = None,
    for_training: bool = True,
) -> PeftModel:
    """Attach a new LoRA adapter to the base model.

    Args:
        model: Base model (or model already wrapped in PeftModel).
        lora_config: Override default LoRA config.
        for_training: If True, prepare model for kbit training and enable grad checkpointing.
    """
    if for_training:
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

    from peft import LoraConfig

    cfg = lora_config or LORA_CONFIG
    peft_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        task_type=cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)

    if for_training:
        model.print_trainable_parameters()

    return model


def load_model_for_inference(
    adapter_path: str | Path,
    model_name: str = MODEL_NAME,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load base model + adapter ready for inference."""
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name)
    model = load_adapter(base_model, adapter_path)
    model.eval()
    return model, tokenizer
