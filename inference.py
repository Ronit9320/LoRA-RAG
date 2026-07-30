import argparse

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/phi-2"
DEFAULT_ADAPTER = "adapters/general_adapter"


def load_model(adapter_path: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def make_generate(model, tokenizer):
    def generate(message: str, history: list, max_new_tokens: int, temperature: float, top_p: float) -> str:
        prompt = f"Instruct: {message}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    return generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Adapter path or HuggingFace model ID")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)
    fn = make_generate(model, tokenizer)

    demo = gr.ChatInterface(
        fn=fn,
        title="Robin LoRA",
        description=f"Phi-2.7B + adapter: {args.adapter}",
        additional_inputs=[
            gr.Slider(64, 512, value=256, step=64, label="Max new tokens"),
            gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
        ],
    )
    demo.launch(share=args.share)
