"""
main.py
=======
FastAPI backend for ASHA-AI.

Endpoints:
  POST /triage        — symptom → triage decision (English or Kannada)
  POST /medcheck      — medicine image → dosage explanation
  POST /voice         — audio file → transcribed Kannada text
  GET  /health        — health check

Run: uvicorn main:app --reload --port 8000

pip install fastapi uvicorn chromadb sentence-transformers
    ollama  (install separately: https://ollama.ai)
    then: ollama pull gemma4:4b   (or your fine-tuned model)
"""

import os
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import httpx  # for Ollama API

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ASHA-AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ChromaDB (offline) ───────────────────────────────────────────────────────
CHROMA_DIR = Path("data/chroma_db")
embed_fn   = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
chroma     = chromadb.PersistentClient(path=str(CHROMA_DIR))
kb         = chroma.get_or_create_collection("asha_knowledge", embedding_function=embed_fn)

# ─── Ollama config (local Gemma 4) ────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:4b")    # your fine-tuned model

SYSTEM_PROMPT = (
    "You are ASHA-AI, a trusted health assistant for ASHA (Accredited Social Health Activist) "
    "workers in rural India. You provide clear, actionable triage guidance based on IMNCI, "
    "ASHA training manuals, and WHO primary healthcare protocols. "
    "Use simple, direct language. Always prioritise safety. When uncertain, advise referral. "
    "Respond in the same language the user uses (Kannada or English)."
)

# ─── Request/Response models ──────────────────────────────────────────────────
class TriageRequest(BaseModel):
    query: str                  # symptom description (Kannada or English)
    language: str = "en"        # "en" | "kn" (Kannada)

class TriageResponse(BaseModel):
    answer: str
    sources: list[str]
    language: str
    confidence: str             # "high" | "medium" | "refer_immediately"

class MedCheckRequest(BaseModel):
    image_base64: str           # base64 encoded medicine label photo
    question: Optional[str] = "What is this medicine and what is the dosage?"

# ─── Helper: call local Ollama ────────────────────────────────────────────────
async def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": prompt},
                ],
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

# ─── Helper: retrieve context from ChromaDB ───────────────────────────────────
def retrieve_context(query: str, n: int = 3) -> tuple[str, list[str]]:
    results = kb.query(query_texts=[query], n_results=n)
    docs    = results["documents"][0]
    sources = [m.get("title", m.get("source", "ASHA Manual"))
               for m in results["metadatas"][0]]
    context = "\n\n".join([f"[{src}] {doc}" for src, doc in zip(sources, docs)])
    return context, sources

# ─── Helper: detect if answer needs immediate referral ───────────────────────
EMERGENCY_KEYWORDS = [
    "emergency", "call 108", "immediate referral", "do not delay",
    "life-threatening", "ತಕ್ಷಣ", "ತುರ್ತು",  # Kannada emergency terms
]

def assess_confidence(answer: str) -> str:
    answer_lower = answer.lower()
    if any(k in answer_lower for k in EMERGENCY_KEYWORDS):
        return "refer_immediately"
    if "refer" in answer_lower or "phc" in answer_lower:
        return "medium"
    return "high"

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "kb_size": kb.count()}


@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest):
    # 1. Retrieve relevant ASHA manual context
    context, sources = retrieve_context(req.query)

    # 2. Build augmented prompt
    lang_instruction = (
        "Respond in Kannada (ಕನ್ನಡ)." if req.language == "kn"
        else "Respond in clear, simple English."
    )
    prompt = (
        f"Relevant guidelines:\n{context}\n\n"
        f"ASHA worker's question: {req.query}\n\n"
        f"{lang_instruction}\n"
        "Give a numbered, step-by-step response. Start with the most urgent action."
    )

    # 3. Call local Gemma 4
    answer = await call_ollama(prompt)

    return TriageResponse(
        answer=answer,
        sources=sources,
        language=req.language,
        confidence=assess_confidence(answer),
    )


@app.post("/medcheck")
async def medcheck(req: MedCheckRequest):
    """
    Takes a base64 medicine label image and returns dosage info.
    Uses Gemma 4's multimodal capability via Ollama.
    """
    prompt = (
        f"This is a photo of a medicine label from India. "
        f"Question: {req.question}\n"
        "Explain in simple terms what this medicine is for, "
        "the recommended dose, and any warnings. "
        "If you cannot read the label clearly, say so."
    )

    # Gemma 4 multimodal via Ollama
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [req.image_base64],
                }],
                "options": {"temperature": 0.1, "num_predict": 256},
            },
        )
        resp.raise_for_status()
        answer = resp.json()["message"]["content"]

    return {"answer": answer, "confidence": assess_confidence(answer)}


@app.post("/voice")
async def voice_to_text(audio: UploadFile = File(...)):
    """
    Transcribes Kannada/Hindi audio using local Whisper.
    Returns transcribed text ready to send to /triage.

    Install: pip install openai-whisper
    Or use whisper.cpp for fully offline edge use.
    """
    try:
        import whisper
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Whisper not installed. Run: pip install openai-whisper"
        )

    # Save uploaded audio temporarily
    tmp_path = Path("/tmp/asha_audio.wav")
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    model = whisper.load_model("small")   # ~460MB — works offline
    result = model.transcribe(str(tmp_path), language=None)  # auto-detect language

    return {
        "text": result["text"],
        "language": result.get("language", "unknown"),
        "ready_for_triage": True,
    }
