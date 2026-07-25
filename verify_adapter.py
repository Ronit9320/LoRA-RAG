import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

model_name = "microsoft/phi-2"
adapter_path = "adapters/general_adapter"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)

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
