"""Prepare training dataset from HuggingFace ultrachat_200k."""

import json
import os
from random import Random

from datasets import load_dataset

from src.config import DATA_DIR, DATASET_CONFIG

output_dir = str(DATA_DIR / "general_adapter_train")
os.makedirs(output_dir, exist_ok=True)

ds = load_dataset(DATASET_CONFIG["source"], split=DATASET_CONFIG["split"])

rng = Random(DATASET_CONFIG["seed"])
indices = list(range(len(ds)))
rng.shuffle(indices)
indices = indices[: DATASET_CONFIG["num_examples"]]

num_train = DATASET_CONFIG["num_examples"] - DATASET_CONFIG["num_eval"]

train_processed = []
eval_processed = []
for i, idx in enumerate(indices):
    messages = ds[idx]["messages"]
    user_msg = None
    asst_msg = None
    for m in messages:
        if m["role"] == "user" and user_msg is None:
            user_msg = m["content"].strip()
        elif m["role"] == "assistant" and asst_msg is None and user_msg is not None:
            asst_msg = m["content"].strip()
            break
    if user_msg and asst_msg:
        item = {
            "prompt": f"Instruct: {user_msg}\nOutput:",
            "response": f" {asst_msg}",
        }
        if i < num_train:
            train_processed.append(item)
        else:
            eval_processed.append(item)

with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
    for item in train_processed:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "eval.jsonl"), "w") as f:
    for item in eval_processed:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(train_processed)} train + {len(eval_processed)} eval examples to {output_dir}/")
