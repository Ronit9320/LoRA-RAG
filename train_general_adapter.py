import os
import json
import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "microsoft/phi-2"
adapter_save_path = "adapters/general_adapter"
train_data_path = "data/general_adapter_train/train.jsonl"
eval_data_path = "data/general_adapter_train/eval.jsonl"
checkpoint_dir = "checkpoints"
max_length = 512

os.makedirs("adapters", exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def load_jsonl(path):
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    lines.sort(key=lambda x: x["prompt"])
    return lines

train_data = load_jsonl(train_data_path)
eval_data = load_jsonl(eval_data_path)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_ids = self.tokenizer.encode(item["prompt"])
        prompt_len = len(prompt_ids)

        full_text = item["prompt"] + item["response"]
        full_ids = self.tokenizer.encode(full_text)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = full_ids.copy()
            mask_end = min(prompt_len, self.max_length)
            for i in range(mask_end):
                labels[i] = -100
        else:
            labels = full_ids.copy()
            for i in range(prompt_len):
                labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

train_dataset = TextDataset(train_data, tokenizer, max_length)
eval_dataset = TextDataset(eval_data, tokenizer, max_length)

def collate_fn(features):
    input_ids = pad_sequence(
        [f["input_ids"] for f in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = pad_sequence(
        [f["labels"] for f in features],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    bf16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

checkpoints = sorted(
    glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1]),
)
resume = checkpoints[-1] if checkpoints else None

if resume:
    print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
else:
    print("Starting fresh training")
    trainer.train()

model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

print(f"Adapter saved to {adapter_save_path}")
