from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import chromadb
from sentence_transformers import SentenceTransformer
from ddgs import DDGS

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_input: str

class ChatResponse(BaseModel):
    response: str

MODEL_NAME = "Qwen/Qwen3-1.7B"
tokenizer = None
model = None
chroma_client = None
collection = None
embedder = None

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

def init_rag():
    global chroma_client, collection, embedder
    chroma_client = chromadb.PersistentClient(path="./robin-memory")
    collection = chroma_client.get_or_create_collection(name="robin_knowledge")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("RAG pipeline initialized")

def format_prompt(messages: List[Dict[str, str]], user_input: str, context: str = None) -> str:
    prompt = f"<think>system\n{SYSTEM_PROMPT}\n</think>\n"
    if context:
        prompt += f"<think>context\n{context}\n</think>\n"
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<think>{role}\n{content}\n</think>\n"
    prompt += f"<think>user\n{user_input}\n</think>\n<think>assistant\n"
    return prompt

def search_memory(query: str, n=3) -> list[str]:
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    if results and results["documents"]:
        return results["documents"][0]
    return []

def web_search_and_store(query: str) -> list[str]:
    queries = [query, f"what is {query}", f"{query} 2026"]
    all_results = []
    
    with DDGS() as ddgs:
        for q in queries:
            if len(all_results) >= 3:
                break
            try:
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    body = r.get("body", "").strip()
                    if body and body not in all_results:
                        all_results.append(body)
            except Exception as e:
                print(f"Search failed for '{q}': {e}")
                continue
    
    if all_results:
        embeddings = embedder.encode(all_results).tolist()
        ids = [f"{hash(q + str(i))}" for i in range(len(all_results))]
        collection.add(documents=all_results, embeddings=embeddings, ids=ids)
    
    return all_results[:3]

def robin_doesnt_know(response: str) -> bool:
    triggers = [
        "i do not have access",
        "i cannot answer",
        "i do not know",
        "no information",
        "unknown"
    ]
    return any(t in response.lower() for t in triggers)

def always_search(query: str) -> bool:
    triggers = [
        "time", "date", "today", "now", "current", "latest",
        "news", "price", "weather", "score", "who is", "who won"
    ]
    return any(t in query.lower() for t in triggers)

@app.on_event("startup")
async def startup():
    load_model()
    init_rag()

@app.get("/")
async def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat")
async def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Pass 1: Generate without context first
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
        print(f"Pass 1 response: {response_text}")
        print(f"Doesnt know: {robin_doesnt_know(response_text)}")

        # Pass 2: If Robin doesn't know, search and regenerate
        if robin_doesnt_know(response_text) or always_search(request.user_input):
            print(f"Robin doesn't know. Searching: {request.user_input}")

            # Check ChromaDB first
            rag_results = search_memory(request.user_input, n=3)

            if rag_results:
                context = "\n\n".join(rag_results)
                print(f"Found {len(rag_results)} memory results")
            else:
                # Fall back to web search
                web_results = web_search_and_store(request.user_input)
                context = "\n\n".join(web_results) if web_results else None
                print(f"Found {len(web_results)} web results" if web_results else "No results found")

            if context:
                # Regenerate with context
                prompt = format_prompt(request.messages, request.user_input, context=context)
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

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