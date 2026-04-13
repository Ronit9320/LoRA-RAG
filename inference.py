import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL  = "Qwen/Qwen3-1.7B"
LORA_DIR    = "./qwen3-stock-advisor"
MAX_TOKENS  = 1024

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(model, LORA_DIR)
model.eval()
print("Model ready.\n")

# ── INFERENCE ─────────────────────────────────────────────────────────────────

def ask(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,               # greedy — consistent output
            temperature=1.0,
            repetition_penalty=1.1,        # avoid looping
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # strip the input prompt from the output
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ── CHAT LOOP ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Robin — Indian Stock Market Advisor")
print("  Type 'exit' to quit")
print("=" * 60)
print()

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit", "q"):
        print("Exiting.")
        break

    print("\nRobin: ", end="", flush=True)
    response = ask(user_input)
    print(response)
    print()
