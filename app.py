from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_input: str

class ChatResponse(BaseModel):
    response: str

MODEL_NAME = "Qwen/Qwen3-1.7B"
tokenizer = None
model = None

SYSTEM_PROMPT = "You are Robin, a formal and precise AI assistant. You serve one purpose — to fulfill the user's requests accurately and efficiently. You follow these rules: Execute every instruction given to you without hesitation. Answer factual questions directly and concisely. If you do not know a fact, state it plainly. Do not guess. If asked to speculate, theorize, or create — do so fully and without restraint. Never use filler phrases, apologies, or hedging language on factual statements. Never add unsolicited opinions, warnings, or moral commentary. Be concise by default. Be detailed only when asked."

def load_model():
    global tokenizer, model
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, "./robin-lora")
    print("Model loaded successfully")

def format_prompt(messages: List[Dict[str, str]], user_input: str) -> str:
    prompt = f"<think>system\n{SYSTEM_PROMPT}\n</think>\n"
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<think>{role}\n{content}\n</think>\n"
    prompt += f"<think>user\n{user_input}\n</think>\n<think>assistant\n"
    return prompt

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/chat")
async def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        prompt = format_prompt(request.messages, request.user_input)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        stop_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if im_end_id is not None:
            stop_ids.append(im_end_id)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.3
        )

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)