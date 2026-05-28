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

# ── PostgreSQL Connection ─────────────────────────────────────────────────────
# Replace placeholders with real credentials from backend team
# Or set as environment variables on Railway

DB_HOST     = os.getenv("DB_HOST",     "YOUR_DB_HOST")  # ask backend team for host
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "Nexus")
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def load_from_db():
    """
    Load interactions and item catalog directly from Nexus PostgreSQL database.
    Falls back to static files if DB connection fails.
    """
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DB_URL)

        # Load item catalog from item + category tables
        catalog_query = """
            SELECT 
                i.id::text          AS item_id,
                i.name              AS item_title,
                i.starting_price    AS avg_price,
                c.name              AS category,
                0                   AS popularity,
                0                   AS n_interactions
            FROM item i
            LEFT JOIN category c ON i.category_id = c.id
        """

        # Load interactions from bid + auction tables
        # bid = interaction_weight 3 (strongest signal)
        interactions_query = """
            SELECT
                b.bidder_id::text       AS user_id,
                a.item_id::text         AS item_id,
                i.name                  AS item_title,
                3                       AS interaction_weight,
                b.bid_time              AS interaction_timestamp,
                b.bid_amount            AS price
            FROM bid b
            JOIN auction a ON b.auction_id = a.id
            JOIN item i    ON a.item_id    = i.id
        """

        catalog = pd.read_sql(catalog_query, engine)
        train   = pd.read_sql(interactions_query, engine)

        # Calculate popularity from interaction counts
        popularity = train.groupby("item_id")["interaction_weight"].sum().reset_index()
        popularity.columns = ["item_id", "popularity"]
        catalog = catalog.merge(popularity, on="item_id", how="left", suffixes=("", "_new"))
        if "popularity_new" in catalog.columns:
            catalog["popularity"] = catalog["popularity_new"].fillna(0)
            catalog.drop(columns=["popularity_new"], inplace=True)
        else:
            catalog["popularity"] = catalog["popularity"].fillna(0)

        print(f"DB loaded: {len(catalog):,} items | {len(train):,} interactions")
        return catalog, train

    except Exception as e:
        print(f"DB connection failed: {e}")
        print("Falling back to static files...")
        return None, None


print("Loading artifacts...")

# Try DB first, fall back to static files
catalog, train = load_from_db()

if catalog is None or train is None:
    catalog = pd.read_csv(f"{DATA_DIR}/item_catalog.csv")
    train   = pd.read_csv(f"{DATA_DIR}/train.csv")
    print(f"Static files loaded: {len(catalog):,} items | {len(train):,} interactions")

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
