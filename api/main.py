import os
import sys
"""
Nexus Recommendation System — FastAPI Service
Run with: python -m uvicorn main:app --reload --port 8000

Dependencies (all already installed from notebooks):
  pip install fastapi uvicorn pandas numpy scikit-learn sentence-transformers scipy
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Recommendation API",
    description="Hybrid recommendation system for the Nexus auction platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Artifacts ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

print("Loading artifacts...")

catalog = pd.read_csv(f"{DATA_DIR}/item_catalog.csv")
train   = pd.read_csv(f"{DATA_DIR}/train.csv")

with open(f"{DATA_DIR}/content_data.pkl", "rb") as f:
    content_data = pickle.load(f)

with open(f"{DATA_DIR}/cf_data.pkl", "rb") as f:
    cf_data = pickle.load(f)

with open(f"{DATA_DIR}/hybrid_model_config.pkl", "rb") as f:
    hybrid_config = pickle.load(f)

# ── Unpack Content Artifacts ───────────────────────────────────────────────────

sbert_emb      = content_data["sbert_embeddings"]
cb_item_to_idx = content_data["item_to_idx"]

# ── Unpack CF Artifacts (scipy SVD) ───────────────────────────────────────────

user_factors    = cf_data["user_factors"]
item_factors    = cf_data["item_factors"]
user_to_idx     = cf_data["user_to_idx"]
item_to_idx     = cf_data["item_to_idx"]
idx_to_item     = cf_data["idx_to_item"]
user_seen_items = cf_data["user_seen_items"]
all_item_ids    = cf_data["all_item_ids"]

# ── Popularity Map ─────────────────────────────────────────────────────────────

catalog["popularity_norm"] = catalog["popularity"] / catalog["popularity"].max()
pop_map = dict(zip(catalog["item_id"], catalog["popularity_norm"]))

ALPHA = hybrid_config.get("best_alpha", 0.5)
BETA  = hybrid_config.get("best_beta",  0.3)
GAMMA = hybrid_config.get("best_gamma", 0.2)

print(f"Loaded. Weights: α={ALPHA}, β={BETA}, γ={GAMMA}")
print(f"Catalog: {len(catalog):,} items | Known users: {len(user_to_idx):,}")

# ── Helper Functions ───────────────────────────────────────────────────────────

def get_cf_scores(user_id: str, candidates: list) -> dict:
    if user_id not in user_to_idx:
        return {item: 0.0 for item in candidates}

    u_idx      = user_to_idx[user_id]
    u_vector   = user_factors[u_idx]
    cand_items = [iid for iid in candidates if iid in item_to_idx]
    cand_idxs  = np.array([item_to_idx[iid] for iid in cand_items])

    if len(cand_idxs) == 0:
        return {item: 0.0 for item in candidates}

    raw    = item_factors[cand_idxs] @ u_vector
    min_s, max_s = raw.min(), raw.max()
    rng    = max_s - min_s if max_s != min_s else 1.0
    normed = (raw - min_s) / rng

    score_map = dict(zip(cand_items, normed.tolist()))
    return {item: score_map.get(item, 0.0) for item in candidates}


def get_content_scores(user_id: str, candidates: list) -> dict:
    history = train[train["user_id"] == user_id]
    if history.empty:
        return {item: 0.0 for item in candidates}

    profile = np.zeros(sbert_emb.shape[1])
    total_w = 0.0
    for _, row in history.iterrows():
        if row["item_id"] in cb_item_to_idx:
            idx      = cb_item_to_idx[row["item_id"]]
            profile += sbert_emb[idx] * row["interaction_weight"]
            total_w += row["interaction_weight"]

    if total_w == 0:
        return {item: 0.0 for item in candidates}

    profile = (profile / total_w).reshape(1, -1)
    scores  = {}
    for iid in candidates:
        if iid in cb_item_to_idx:
            idx = cb_item_to_idx[iid]
            scores[iid] = float(cosine_similarity(profile, sbert_emb[idx].reshape(1, -1))[0][0])
        else:
            scores[iid] = 0.0
    return scores


def hybrid_recommend(user_id: str, top_n: int = 10) -> list:
    seen       = user_seen_items.get(user_id, set())
    candidates = [iid for iid in catalog["item_id"] if iid not in seen]

    cf_scores      = get_cf_scores(user_id, candidates)
    content_scores = get_content_scores(user_id, candidates)

    results = []
    for iid in candidates:
        cf_s  = cf_scores.get(iid, 0.0)
        cb_s  = content_scores.get(iid, 0.0)
        pop_s = pop_map.get(iid, 0.0)
        final = ALPHA * cf_s + BETA * cb_s + GAMMA * pop_s
        results.append({
            "item_id":       iid,
            "score":         round(final, 4),
            "cf_score":      round(cf_s, 4),
            "content_score": round(cb_s, 4),
            "pop_score":     round(pop_s, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top     = results[:top_n]
    cat_map = catalog.set_index("item_id")[["item_title", "avg_price", "popularity"]].to_dict("index")

    for r in top:
        info         = cat_map.get(r["item_id"], {})
        r["item_title"] = str(info.get("item_title", "Unknown"))
        r["avg_price"]  = round(float(info.get("avg_price", 0)), 2)
        r["popularity"] = int(info.get("popularity", 0))

    return top


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "Nexus Recommendation API", "version": "1.0.0",
            "docs": "/docs", "endpoints": ["/recommend", "/popular", "/similar/{item_id}", "/users", "/health"]}


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": len(catalog),
            "known_users": len(user_to_idx), "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA}}


@app.get("/recommend")
def recommend(user_id: str = Query(...), top_n: int = Query(10, ge=1, le=50)):
    """Personalized hybrid recommendations for a user."""
    try:
        recs = hybrid_recommend(user_id, top_n=top_n)
        return {"user_id": user_id, "cold_start": user_id not in user_to_idx,
                "count": len(recs), "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/popular")
def popular(top_n: int = Query(10, ge=1, le=50)):
    """Most popular items — fallback for cold start."""
    top = catalog.nlargest(top_n, "popularity")[["item_id","item_title","avg_price","popularity"]].copy()
    top["popularity"] = top["popularity"].astype(int)
    top["avg_price"]  = top["avg_price"].astype(float)
    return {"count": len(top), "items": top.to_dict("records")}


@app.get("/similar/{item_id}")
def similar_items(item_id: str, top_n: int = Query(10, ge=1, le=50)):
    """Items similar to a given item using SBERT cosine similarity."""
    item_id = item_id.upper()
    if item_id not in cb_item_to_idx:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    idx    = cb_item_to_idx[item_id]
    vec    = sbert_emb[idx].reshape(1, -1)
    scores = cosine_similarity(vec, sbert_emb).flatten()
    scores[idx] = 0
    top_idx = np.argsort(scores)[::-1][:top_n]
    inv_map = {v: k for k, v in cb_item_to_idx.items()}
    cat_map = catalog.set_index("item_id").to_dict("index")

    results = [{"item_id": inv_map[i], "item_title": cat_map.get(inv_map[i], {}).get("item_title",""),
                "avg_price": round(float(cat_map.get(inv_map[i],{}).get("avg_price",0)),2),
                "similarity": round(float(scores[i]),4)} for i in top_idx]

    return {"seed_item": {"item_id": item_id, "item_title": cat_map.get(item_id,{}).get("item_title","")},
            "count": len(results), "similar": results}


@app.get("/users")
def list_users(limit: int = Query(20, ge=1, le=200)):
    users = [str(u) for u in list(user_to_idx.keys())[:limit]]
    return {"count": len(users), "users": users}


@app.get("/items")
def list_items(limit: int = Query(20, ge=1, le=200)):
    items = catalog.head(limit)[["item_id","item_title","avg_price","popularity"]].copy()
    items["popularity"] = items["popularity"].astype(int)
    items["avg_price"]  = items["avg_price"].astype(float)
    return {"count": len(items), "items": items.to_dict("records")}


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

import re, random
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    message:    str
    user_id:    Optional[str] = "GUEST"
    history:    Optional[List[ChatMessage]] = []
    fail_count: Optional[int] = 0

# Lazy loaded
_chat_ready = False
_collection = None
_chat_sbert = None
OLLAMA_MODEL = "llama3.2"

ALLOWED_TOPICS = [
    "bid","auction","item","price","pay","ship","deliver","account","register",
    "login","password","sell","consign","return","refund","dispute","complain",
    "authentic","condition","recommend","suggest","find","search","help","hello",
    "hi","hey","watch","jewel","antique","art","furniture","coin","book","invoice",
    "receipt","collect","track","reserve","premium","estimate","lot","winner",
    "lost","suspend","agent","human","customer","service","support",
    "thank","thanks","ok","okay","great","got","sure","yes","no","please",
    "sorry","appreciate","understood","noted","cool","good","alright","cheers",
    "how","what","when","where","going","doing","are","you","can","need","want",
    "have","will","would","could","should","any","is","my","it","do","i",
]

INAPPROPRIATE_WORDS = [
    "fuck","shit","bitch","bastard","idiot","cunt","porn","nude","naked","racist","terrorist",
]

FRUSTRATION_SIGNALS = [
    r"not help",r"useless",r"terrible",r"awful",r"rubbish",r"speak.*human",
    r"real person",r"agent",r"customer service",r"manager",r"connect me",
    r"live chat",r"talk.*someone",r"don't understand",
]

INTENT_PATTERNS = {
    "recommend":  [r"recommend",r"suggest",r"what should",r"show me",r"find me",r"looking for"],
    "price":      [r"how much",r"price",r"cost",r"worth",r"value",r"sell for"],
    "how_to_bid": [r"how.*bid",r"how.*auction",r"place.*bid",r"how.*buy",r"how.*work"],
    "payment":    [r"pay",r"invoice",r"receipt",r"refund",r"charge"],
    "shipping":   [r"ship",r"deliver",r"collect",r"track",r"arriv"],
    "account":    [r"account",r"login",r"password",r"register",r"suspend"],
    "complaint":  [r"complain",r"dispute",r"not.*described",r"broken",r"missing",r"fraud"],
    "agent":      [r"human",r"agent",r"person",r"customer service",r"manager",r"live chat",r"connect me"],
    "greeting":   [r"^hi$",r"^hello$",r"^hey$",r"good morning",r"good afternoon",r"^hi",r"^hello"],
}

GREETING_REPLY   = "Hello! Welcome to Nexus Auctions. I'm ARIA, your auction assistant. I can help you with bidding, payments, shipping, account issues, and finding items. What can I help you with today?"
ESCALATE_REPLY   = "I understand. Let me connect you with a member of our customer service team who can help you further."
INAPPROPRIATE_REPLY = "I'm not able to engage with that kind of message. Please keep our conversation respectful and auction-related."
OFF_TOPIC_REPLIES = [
    "I'm here to help with Nexus auction questions only — bidding, payments, shipping, and account support. For anything else I'm afraid I'm not the right assistant!",
    "That's outside what I can help with. I specialise in Nexus auction support. Is there anything auction-related I can assist with?",
    "I'm only able to help with Nexus Auctions topics. If you have a question about bidding, an item, payment, or your account, I'm all yours!",
]

SYSTEM_PROMPT = """You are ARIA, the customer support assistant for Nexus Auctions, a premium online auction house.
Help users with bidding, payments, shipping, account issues, item questions, and disputes.
Answer ONLY using the provided context. Be warm, professional, concise (2-4 sentences).
Use British English and GBP for prices. Never invent policies or contact details."""


def _detect_intent(text):
    t = text.strip().lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, t): return intent
    return "general"

def _is_on_topic(text):
    return any(topic in text.lower() for topic in ALLOWED_TOPICS)

def _is_frustrated(text):
    return any(re.search(p, text.lower()) for p in FRUSTRATION_SIGNALS)

def _is_inappropriate(text):
    words = re.findall(r"[a-z]+", text.lower())
    return any(w in INAPPROPRIATE_WORDS for w in words)

def _init_chat():
    global _chat_ready, _collection, _chat_sbert
    if _chat_ready: return True
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"]       = "1"
        CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chatbot", "chroma_db")
        if not os.path.exists(CHROMA_DIR):
            print(f"[ARIA] ChromaDB not found at {CHROMA_DIR}")
            return False
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("nexus_knowledge")
        _chat_sbert = SentenceTransformer("all-MiniLM-L6-v2")
        _chat_ready = True
        print(f"[ARIA] Ready. {_collection.count()} documents loaded.")
        return True
    except Exception as e:
        print(f"[ARIA] Init failed: {e}")
        return False

def _retrieve(query, n=3):
    emb = _chat_sbert.encode([query]).tolist()
    return _collection.query(query_embeddings=emb, n_results=n)["documents"][0]


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if _is_inappropriate(req.message):
        return {"response": INAPPROPRIATE_REPLY, "intent": "blocked", "escalate": False}

    intent = _detect_intent(req.message)

    if intent == "greeting":
        return {"response": GREETING_REPLY, "intent": "greeting", "escalate": False}

    if intent == "agent" or _is_frustrated(req.message) or (req.fail_count or 0) >= 2:
        return {"response": ESCALATE_REPLY, "intent": "escalate", "escalate": True}

    # Short acknowledgements — just confirm and invite next question
    ACK_WORDS = {"great","ok","okay","thanks","thank","sure","noted","understood","alright","cheers","cool","perfect","got it","nice","good","k","yep","yup"}
    if req.message.strip().lower() in ACK_WORDS or len(req.message.strip().split()) <= 2 and _detect_intent(req.message) == "general" and any(w in req.message.lower() for w in ACK_WORDS):
        ack_replies = [
            "Great! Is there anything else I can help you with?",
            "Of course! Feel free to ask if you have any other questions.",
            "Happy to help! Anything else you would like to know?",
        ]
        return {"response": random.choice(ack_replies), "intent": "ack", "escalate": False}

    if intent == "general" and not _is_on_topic(req.message):
        return {"response": random.choice(OFF_TOPIC_REPLIES), "intent": "off_topic", "escalate": False}

    # Short acknowledgements — no need to call Ollama
    ACK_WORDS = ["thank","thanks","great","ok","okay","good","noted","understood","cheers","alright","perfect","sure","got it","appreciate"]
    if any(req.message.strip().lower() == w or req.message.strip().lower() == w + "!" for w in ACK_WORDS):
        return {"response": "You're welcome! Is there anything else I can help you with?", "intent": "acknowledgement", "escalate": False}

    if not _init_chat():
        return {"response": "I'm not fully set up yet. Please run notebook 05 first.", "intent": "error", "escalate": False}

    context_parts = []
    try:
        chunks = _retrieve(req.message)
        if chunks: context_parts.append("KNOWLEDGE:\n" + "\n".join(chunks))
    except Exception as e:
        print(f"[ARIA] Retrieval error: {e}")

    if intent == "recommend":
        try:
            recs = hybrid_recommend(req.user_id, top_n=3)
            if recs:
                lines = [f"{i+1}. {str(r.get('item_title','')).title()} — £{float(r.get('avg_price',0)):.2f}" for i,r in enumerate(recs[:3])]
                context_parts.append("PERSONALISED ITEMS:\n" + "\n".join(lines))
        except: pass

    context  = "\n\n".join(context_parts)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (req.history or [])[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.message}" if context else req.message})

    try:
        import ollama as _ollama
        resp = _ollama.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": 0.5, "num_predict": 200})
        return {"response": resp["message"]["content"], "intent": intent, "escalate": False}
    except Exception as e:
        return {"response": f"Sorry, I cannot reach the AI model. Make sure Ollama is running. ({e})", "intent": "error", "escalate": False}
