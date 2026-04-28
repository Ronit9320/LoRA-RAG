import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-1.7B"
DATA_FILE = "training_data.json"
OUTPUT_DIR = "./robin-lora"
MAX_SEQ_LEN = 512
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── LOAD & FORMAT DATA ────────────────────────────────────────────────────────


def format_conversation(example):
    """
    Convert conversations array into Qwen3 chat format:
    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    ...
    <|im_end|>
    """
    text = ""
    for msg in example["conversations"]:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    return {"text": text}


print("Loading training data...")
with open(DATA_FILE, "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_conversation)
print(f"Dataset size: {len(dataset)} examples")
print(f"Sample:\n{dataset[0]['text'][:300]}\n")

# ── LOAD MODEL & TOKENIZER ────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
model.enable_input_require_grads()

# ── CONFIGURE LoRA ────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── TRAINING ARGUMENTS ────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    warmup_steps=5,
    lr_scheduler_type="cosine",
    report_to="none",
    optim="paged_adamw_8bit",
)

# ── TRAIN ─────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()

# ── SAVE ──────────────────────────────────────────────────────────────────────

print("Saving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
