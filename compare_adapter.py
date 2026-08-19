"""Side-by-side comparison of base model vs adapter-enhanced output."""

import torch

from src.config import DEFAULT_ADAPTER
from src.model import load_adapter, load_base_model, load_tokenizer

adapter_path = str(DEFAULT_ADAPTER)

tokenizer = load_tokenizer()
base = load_base_model()
model = load_adapter(base, adapter_path)

prompts = [
    "Explain the difference between gradient descent and stochastic gradient descent.",
    "Write a Python function that implements a binary search.",
    "A train leaves Station A at 60 km/h. Another train leaves Station B (200 km away) at 40 km/h heading toward it. They start at the same time. When and where do they meet?",
    "Compare supervised, unsupervised, and reinforcement learning with a real-world use case for each.",
    "The following is a proof attempt: 'All horses are the same color. Proof by induction: Base case — one horse is trivially the same color as itself. Inductive step — assume any n horses are the same color. For n+1 horses, remove one, the remaining n are the same color; remove a different one, the remaining n are the same color; therefore all n+1 are the same color.' Is this proof valid? If not, identify the flaw.",
]

for name, enabled in [("WITHOUT adapter", False), ("WITH adapter", True)]:
    if enabled:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    for prompt in prompts:
        formatted = f"Instruct: {prompt}\nOutput:"
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print("---")
        print(result)
        print()
