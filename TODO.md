# Robin LoRA — Implementation Checklist

## Infrastructure
- [x] Create `requirements.txt` (transformers, peft, bitsandbytes, gradio, datasets, torch, sentence-transformers, faiss-gpu, ruff, pyright)
- [x] Create `src/` directory and move scripts into it
- [x] Create `src/config.py` — centralized paths, model name, hyperparameters
- [x] Refactor shared model loading code into `src/model.py` (quantization config, base model, tokenizer, adapter loading)
- [x] Set up `ruff check` and `pyright` linting
- [x] Create `notebooks/` directory for exploration

## Training
- [ ] Refactor `train_general_adapter.py` → `src/train.py` using shared config/model
- [ ] General adapter training with multi-adapter support (load general adapter first, then task adapter on top)
- [ ] Task-specific adapter training loop
- [ ] At least one task-specific adapter trained and saved

## Inference
- [x] Refactor `inference.py` → `src/inference.py` using shared model loading
- [x] Multi-adapter inference (general + task adapter via PEFT `add_adapter`/`set_adapter`)
- [x] Dynamic adapter loading at runtime without reloading base model

## Adapter Switching
- [x] Create `src/switch.py` — manual adapter switch logic
- [x] On switch: index previous model output into RAG vector store
- [x] On switch: load new task adapter, inject retrieved context into next input

## RAG Pipeline
- [x] Create `src/rag.py`
- [x] Embedding model setup (sentence-transformers)
- [x] Vector store setup (FAISS or Chroma)
- [x] Indexing: embed and store model outputs on task switch
- [x] Retrieval: embed query, retrieve relevant past context
- [x] Context injection: feed retrieved context into prompt for new task

## Integration
- [ ] End-to-end flow: load model → train general adapter → train task adapters → inference → switch tasks → RAG retrieval
