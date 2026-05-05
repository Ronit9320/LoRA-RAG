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
from datetime import datetime
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

SYSTEM_PROMPT = f"You are Robin, a straightforward AI assistant. The current date is {datetime.now().strftime('%B %d, %Y')}. You do not hallucinate. If you do not know something, say so directly. You have access to a search tool — when you do not know something, web search results will be provided to you in the context. Always use the provided context to answer the user's question. Do not say you cannot access information if context is provided."

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
        # Step 1: Check ChromaDB for relevant context
        rag_results = search_memory(request.user_input, n=3)
        context = None
        
        # Step 2: If no memory results, search web
        if not rag_results:
            print(f"Memory empty for: {request.user_input}, searching web...")
            web_results = web_search_and_store(request.user_input)
            if web_results:
                context = "\n\n".join(web_results)
                print(f"Found {len(web_results)} web results")
            else:
                print("No web results found")
        else:
            context = "\n\n".join(rag_results)
            print(f"Found {len(rag_results)} memory results")
        
        # Step 3: Generate with context (if any)
        prompt = format_prompt(request.messages, request.user_input, context=context)
        
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
        
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response_text = response_text.strip()
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)