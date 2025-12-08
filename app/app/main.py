# app/main.py
import os
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

from app.models import GenerationRequest, GenerationResponse

# --- 1. Configuration and Global State ---
API_KEY = os.environ.get("API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "distilgpt2")

if not API_KEY:
    # Service cannot run without a key
    raise ValueError("API_KEY environment variable is not set. Cannot start service.")

# Globals for the lazy-loaded model (Singleton Pattern)
_LLM_MODEL: Optional[AutoModelForCausalLM] = None
_LLM_TOKENIZER: Optional[AutoTokenizer] = None
_LOCK = threading.Lock() # Ensures thread-safe initialization

app = FastAPI(
    title="Containerized LLM Serving API",
    description="A robust, production-ready REST service for a Hugging Face LLM.",
    version="1.0.0"
)

# --- 2. Model Loading Function ---
def load_llm_model():
    """Synchronous function to load the model into memory only once."""
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is None:
        with _LOCK:
            # Double check inside the lock
            if _LLM_MODEL is None:
                print(f"[{time.ctime()}] ⏳ Loading LLM: {LLM_MODEL_NAME}...")
                try:
                    # Use GPU if available, otherwise fall back to CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    _LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
                    _LLM_MODEL = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME).to(device)
                    print(f"[{time.ctime()}] ✅ LLM loaded successfully on device: {device}.")
                except Exception as e:
                    print(f"[{time.ctime()}] ❌ Error loading model: {e}")
                    raise

# --- 3. Dependency for API Key Authentication ---
async def verify_api_key(x_api_key: str = Header(..., description="Your secret API key.")):
    """Authenticates the request based on the X-API-Key header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    return True

# --- 4. API Endpoints ---

@app.get("/health", status_code=200)
def health_check():
    """Core requirement: Health check endpoint to ensure the service is running."""
    return {"status": "ok", "message": "LLM Service is up and running!"}

@app.post("/generate", response_model=GenerationResponse, dependencies=[Depends(verify_api_key)])
async def generate_text(request: GenerationRequest):
    """
    Core requirement: Generates text using the LLM.
    The CPU-bound inference is offloaded to a thread pool for concurrency.
    """
    # Lazy Load Model on the first request (I/O bound task)
    if _LLM_MODEL is None:
        # FastAPI's loop.run_in_executor (aliased by asyncio.to_thread)
        # runs the synchronous load_llm_model in a separate thread,
        # preventing the main event loop from being blocked by I/O (disk read).
        await asyncio.to_thread(load_llm_model)

    if _LLM_MODEL is None or _LLM_TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Check server logs.")

    # --- Synchronous Inference Logic (CPU-bound) ---
    def model_inference():
        """This function executes in a separate thread and contains the heavy, blocking work."""
        # Tokenize and move to device
        inputs = _LLM_TOKENIZER(request.prompt, return_tensors="pt", truncation=True)
        device = _LLM_MODEL.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate text
        output_tokens = _LLM_MODEL.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=_LLM_TOKENIZER.eos_token_id
        )

        # Decode and clean up
        generated_text = _LLM_TOKENIZER.decode(output_tokens[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()

        return generated_text

    try:
        # Core Requirement: Offload CPU-bound task to thread pool
        generated_text = await asyncio.to_thread(model_inference)
        
        return GenerationResponse(generated_text=generated_text)
        
    except Exception as e:
        print(f"[{time.ctime()}] 🚨 Inference Error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during generation.")