"""
chatbot_api.py — ARIA Chatbot Service
Nexus Auctions — Standalone FastAPI chatbot
"""

import os
import re
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from groq import Groq

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARIA — Nexus Chatbot API",
    description="RAG-powered auction support chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chatbot",
    "chroma_db"
)

# ✅ FIXED MODEL NAME
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

# ── Groq client (GLOBAL FIX) ─────────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

# ── Globals ───────────────────────────────────────────────────────────────────

_ready = False
_collection = None
_sbert = None
_rag_error = None

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA, the customer support assistant for Nexus Auctions.
Help users with bidding, payments, shipping, account issues, and item questions.
Be concise and professional.
"""

# ── RAG INIT ──────────────────────────────────────────────────────────────────

def init_rag():
    global _ready, _collection, _sbert, _rag_error

    if _ready:
        return True

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(CHROMA_DIR):
            _rag_error = f"ChromaDB not found at {CHROMA_DIR}"
            print(f"[ARIA] {_rag_error}")
            return False

        client_db = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client_db.get_collection("nexus_knowledge")

        _sbert = SentenceTransformer("all-MiniLM-L6-v2")

        _ready = True
        _rag_error = None
        print(f"[ARIA] Ready. {_collection.count()} docs loaded.")
        return True

    except Exception as e:
        _rag_error = str(e)
        print(f"[ARIA] Init failed: {_rag_error}")
        return False


def retrieve(query, n=3):
    emb = _sbert.encode([query]).tolist()
    result = _collection.query(query_embeddings=emb, n_results=n)
    return result.get("documents", [[]])[0]

# ── Load RAG on startup ───────────────────────────────────────────────────────

init_rag()

# ── Request Model ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "GUEST"
    history: Optional[List[ChatMessage]] = []
    fail_count: Optional[int] = 0

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "ARIA Chatbot API",
        "model": GROQ_MODEL,
        "provider": "Groq"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": _ready,
        "docs": _collection.count() if _ready else 0,
        "rag_error": _rag_error,
        "model": GROQ_MODEL
    }

@app.post("/chat")
def chat(req: ChatRequest):

    if not _ready:
        init_rag()

    context = ""

    try:
        chunks = retrieve(req.message)
        if chunks:
            context = "\n".join(chunks)
    except Exception as e:
        print(f"[ARIA] retrieval error: {e}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in (req.history or [])[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    user_input = f"Context:\n{context}\n\nQuestion: {req.message}" if context else req.message

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=200,
        )

        return {
            "response": response.choices[0].message.content,
            "intent": "general",
            "escalate": False
        }

    except Exception as e:
        return {
            "response": f"Sorry, I cannot reach the AI model. ({e})",
            "intent": "error",
            "escalate": False
        }
