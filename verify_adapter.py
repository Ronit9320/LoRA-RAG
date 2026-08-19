"""Quick verification of the trained adapter."""

import torch

from src.config import DEFAULT_ADAPTER
from src.model import load_adapter, load_base_model, load_tokenizer

tokenizer = load_tokenizer()
base_model = load_base_model()
model = load_adapter(base_model, str(DEFAULT_ADAPTER))

prompts = [
    "Instruct: Explain the difference between gradient descent and stochastic gradient descent, including when you would prefer one over the other.\nOutput:",
    "Instruct: Write a Python function that implements a binary search and explain its time complexity.\nOutput:",
    "Instruct: A train leaves Station A at 60 km/h. Another train leaves Station B (200 km away) at 40 km/h heading toward it. They start at the same time. When and where do they meet?\nOutput:",
    "Instruct: Compare and contrast supervised, unsupervised, and reinforcement learning. Give a real-world use case for each.\nOutput:",
    "Instruct: The following is a proof attempt: 'All horses are the same color. Proof by induction: Base case — one horse is trivially the same color as itself. Inductive step — assume any n horses are the same color. For n+1 horses, remove one, the remaining n are the same color; remove a different one, the remaining n are the same color; therefore all n+1 are the same color.' Is this proof valid? If not, identify the flaw.\nOutput:",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 60)
    print(result)
    print("=" * 60)
