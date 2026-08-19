"""Smoke test for base model loading (no adapter)."""

from src.model import load_base_model, load_tokenizer

tokenizer = load_tokenizer()
model = load_base_model()

prompt = "hi"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with __import__("torch").no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
