"""Centralized configuration for Robin LoRA pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Model
MODEL_NAME = "microsoft/phi-2"
DEFAULT_ADAPTER = ADAPTERS_DIR / "general_adapter"

# Adapter registry — add new task adapters here as they are trained
ADAPTER_REGISTRY: dict[str, dict] = {
    "general": {
        "path": ADAPTERS_DIR / "general_adapter",
        "description": "General-purpose adapter, always active",
        "trained": True,
    },
    "code": {
        "path": ADAPTERS_DIR / "code_adapter",
        "description": "Code generation and debugging",
        "trained": False,
    },
}

# Quantization
QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
}

# LoRA
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["Wqkv", "out_proj", "fc1", "fc2"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Training
TRAIN_CONFIG = {
    "max_length": 512,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "eval_steps": 200,
    "save_steps": 500,
    "save_total_limit": 2,
    "eval_strategy": "steps",
    "dataloader_num_workers": 2,
    "remove_unused_columns": False,
    "report_to": "none",
}

# Inference
INFERENCE_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# RAG
RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "index_dir": PROJECT_ROOT / "rag_store",
    "top_k": 5,
    "similarity_threshold": 0.3,
}

# Dataset
DATASET_CONFIG = {
    "source": "HuggingFaceH4/ultrachat_200k",
    "split": "train_sft",
    "num_examples": 8000,
    "num_eval": 200,
    "seed": 42,
}
