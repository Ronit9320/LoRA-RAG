# Dynamic LoRA Adapter Pipeline with RAG

## Idea

Fine-tune an open-source base model (Phi-2.7B) using LoRA adapters for different tasks. A general-purpose LoRA adapter is always attached to the base model. Task-specific adapters are loaded alongside it depending on the task at hand.

When the active task changes, the model's output is indexed into a RAG pipeline. That output becomes retrievable context, which is fed into the model when the new task adapter is loaded — so the next adapter's input includes relevant context from before the switch.

## Architecture

- **Base model**: Phi-2.7B
- **General-purpose adapter**: LoRA adapter, always active, never merged into base weights (kept swappable/updatable)
- **Task-specific adapters**: LoRA adapters, one per task, trained with the general adapter attached and active (not merged) so they learn on top of the actual combined behavior
- **Adapter switching**: manual, based on task requirements — no automatic routing/classifier initially
- **RAG pipeline**:
  - Embedding model for indexing outputs
  - Vector store for retrieval
  - Retrieval + prompt construction logic to inject retrieved context into the next input

## Inference Flow

1. Base model + general adapter + active task adapter all loaded simultaneously (multi-adapter, not merged)
2. Model produces output
3. If task/adapter needs to change:
   - Output is indexed into RAG store
   - New task adapter is loaded (general adapter stays attached)
   - Retrieved context from RAG is injected into the next input
4. Model generates next output with new adapter + retrieved context

## Key Technical Requirements

1. **Base model**: Phi-2.7B (HuggingFace)
2. **LoRA fine-tuning**: PEFT + transformers + bitsandbytes (quantized training)
3. **Training data**: task-specific datasets per adapter, plus general-purpose dataset
4. **Adapter management**: PEFT's `add_adapter`, multiple adapters active simultaneously via `set_adapter`
5. **RAG components**: sentence-transformers (embeddings), FAISS/Chroma (vector store)
6. **Manual switch logic**: simple script/interface to select active adapter
7. **Compute**: GPU with 12GB+ VRAM (sufficient for Phi-2.7B + LoRA + quantization)

## Key Design Decision

General adapter is NOT merged into the base model. It stays as a separate, active, updatable LoRA adapter at all times. Task adapters are trained with the general adapter loaded and active, so training reflects the actual combined runtime behavior — without permanently baking the general adapter into the base weights.
