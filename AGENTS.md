# Robin LoRA — Dynamic LoRA Adapter Pipeline with RAG

## Project Overview

Fine-tune an open-source base model (Phi-2.7B) using LoRA adapters for different tasks. A general-purpose LoRA adapter is always attached to the base model. Task-specific adapters are loaded alongside it depending on the task at hand. When the active task changes, the model's output is indexed into a RAG pipeline, and retrieved context is fed into the model when the new task adapter loads.

## Tech Stack

- **Base model**: Phi-2.7B (HuggingFace)
- **LoRA fine-tuning**: PEFT + transformers + bitsandbytes
- **RAG**: sentence-transformers (embeddings), FAISS/Chroma (vector store)
- **Runtime**: Python 3.10+, PyTorch, GPU with 12GB+ VRAM

## Architecture

1. **Base model** — Phi-2.7B, never modified.
2. **General-purpose adapter** — LoRA, always active, never merged into base weights.
3. **Task-specific adapters** — One per task, trained with the general adapter attached and active.
4. **RAG pipeline** — Embedding model → vector store → retrieval → prompt construction.
5. **Switching** — Manual adapter switch logic.

## Inference Flow

Base + general + task adapter (multi-adapter via PEFT) → generate → on task switch: index output into RAG, load new task adapter, inject retrieved context into next input.

## Key Design Decisions

- General adapter is **never merged** — stays as a separate, active, updatable LoRA adapter.
- Task adapters are trained **with the general adapter loaded and active** so training matches runtime behavior.
- No automatic routing/classifier initially — manual switching only.

## Project Structure (Planned)

```
Robin_LoRA/
├── AGENTS.md
├── lora_adapter_pipeline.md
├── src/
│   ├── model.py              # Base model + adapter loading
│   ├── train.py              # Training loop for adapters
│   ├── inference.py          # Multi-adapter inference
│   ├── rag.py                # RAG pipeline (embeddings, vector store, retrieval)
│   ├── switch.py             # Manual adapter switching logic
│   └── config.py             # Configuration (paths, hyperparams, etc.)
├── adapters/                 # Saved LoRA adapter weights
│   ├── general/
│   ├── task_1/
│   └── task_2/
├── data/                     # Training datasets
├── notebooks/                # Exploration / experiments
└── requirements.txt
```

## Conventions

- Follow existing patterns in the codebase (PEFT `add_adapter`/`set_adapter` API).
- Keep the general adapter and task adapters as separate PEFT adapters — never merge.
- Docstrings on public functions. Minimal inline comments.
- Type hints on all function signatures.
- Config goes in `src/config.py` (paths, model name, hyperparams).
- Run `ruff check` and `pyright` before committing.

## Implementation Plan

1. **Environment setup** — `requirements.txt`, install deps, verify GPU.
2. **Base model loading** — `src/model.py`: load Phi-2.7B quantized, add general LoRA adapter.
3. **Training** — `src/train.py`: training script for general adapter + task adapters.
4. **RAG pipeline** — `src/rag.py`: embedding, indexing, retrieval, context injection.
5. **Inference + switching** — `src/inference.py` + `src/switch.py`: multi-adapter inference and manual switch.
6. **Integration** — End-to-end: load model → train adapters → run inference → switch tasks → RAG retrieval.
