import os
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Recommendation API",
    description="Database-driven recommendation system for the Nexus auction platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database Config ───────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. Example: "
        "postgresql://postgres:postgres@postgres:5432/nexus"
    )

engine = create_engine(DATABASE_URL)


# ── Recommendation Weights ────────────────────────────────────────────────────

VIEW_WEIGHT = 1
BID_WEIGHT = 4

ALPHA = 0.55   # collaborative score
BETA = 0.30    # content score
GAMMA = 0.15   # popularity score


# ── Global Data Cache ─────────────────────────────────────────────────────────

catalog = pd.DataFrame()
train = pd.DataFrame()
tfidf_matrix = None
item_to_content_idx = {}
user_seen_items = {}
item_users = {}
pop_map = {}


# ── Database Loading ──────────────────────────────────────────────────────────

def load_from_db():
    """
    Load item catalog and interactions directly from PostgreSQL.
    No CSV fallback. No pickle files.
    """

    catalog_query = """
        SELECT
            i.id::text AS item_id,
            i.name AS item_title,
            COALESCE(i.description, '') AS description,
            COALESCE(c.name, '') AS category,
            COALESCE(i.starting_price, 0) AS avg_price
        FROM item i
        LEFT JOIN category c ON i.category_id = c.id;
    """

    view_query = """
        SELECT
            ui.user_id::text AS user_id,
            ui.item_id::text AS item_id,
            i.name AS item_title,
            1 AS interaction_weight,
            ui.created_at AS interaction_timestamp,
            COALESCE(i.starting_price, 0) AS price
        FROM user_interactions ui
        JOIN item i ON ui.item_id::text = i.id::text
        WHERE ui.interaction_type = 'view';
    """

    bid_query = """
        SELECT
            b.bidder_id::text AS user_id,
            a.item_id::text AS item_id,
            i.name AS item_title,
            4 AS interaction_weight,
            b.bid_time AS interaction_timestamp,
            b.bid_amount AS price
        FROM bid b
        JOIN auction a ON b.auction_id = a.id
        JOIN item i ON a.item_id = i.id;
    """

    db_catalog = pd.read_sql(catalog_query, engine)

    views = pd.read_sql(view_query, engine)
    bids = pd.read_sql(bid_query, engine)

    db_train = pd.concat([views, bids], ignore_index=True)

    if db_catalog.empty:
        raise RuntimeError("No items found in database.")

    # If there are no interactions yet, keep empty train but API can still return popular/latest items.
    if db_train.empty:
        db_train = pd.DataFrame(
            columns=[
                "user_id",
                "item_id",
                "item_title",
                "interaction_weight",
                "interaction_timestamp",
                "price",
            ]
        )

    popularity = (
        db_train.groupby("item_id")["interaction_weight"]
        .sum()
        .reset_index()
        .rename(columns={"interaction_weight": "popularity"})
    )

    db_catalog = db_catalog.merge(popularity, on="item_id", how="left")
    db_catalog["popularity"] = db_catalog["popularity"].fillna(0)

    return db_catalog, db_train


def rebuild_indexes():
    """
    Build in-memory structures from latest DB data.
    """

    global tfidf_matrix
    global item_to_content_idx
    global user_seen_items
    global item_users
    global pop_map

    if catalog.empty:
        raise RuntimeError("Catalog is empty.")

    # Content-based text
    content_text = (
        catalog["item_title"].fillna("") + " " +
        catalog["description"].fillna("") + " " +
        catalog["category"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(content_text)

    item_to_content_idx = {
        item_id: idx
        for idx, item_id in enumerate(catalog["item_id"].tolist())
    }

    # User seen items
    seen = defaultdict(set)
    item_user_weights = defaultdict(dict)

    for _, row in train.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        weight = float(row["interaction_weight"])

        seen[user_id].add(item_id)
        item_user_weights[item_id][user_id] = item_user_weights[item_id].get(user_id, 0) + weight

    user_seen_items = dict(seen)
    item_users = dict(item_user_weights)

    max_pop = catalog["popularity"].max()
    if max_pop and max_pop > 0:
        catalog["popularity_norm"] = catalog["popularity"] / max_pop
    else:
        catalog["popularity_norm"] = 0.0

    pop_map = dict(zip(catalog["item_id"], catalog["popularity_norm"]))


def refresh_data():
    """
    Reload DB data and rebuild recommendation indexes.
    """

    global catalog
    global train

    catalog, train = load_from_db()
    rebuild_indexes()

    print(f"DB loaded: {len(catalog):,} items | {len(train):,} interactions")


# Load once at startup
refresh_data()


# ── Recommendation Helpers ────────────────────────────────────────────────────

def normalize_scores(score_dict):
    if not score_dict:
        return {}

    values = np.array(list(score_dict.values()), dtype=float)

    min_v = values.min()
    max_v = values.max()

    if max_v == min_v:
        return {k: 0.0 for k in score_dict.keys()}

    return {
        k: float((v - min_v) / (max_v - min_v))
        for k, v in score_dict.items()
    }


def get_cf_scores(user_id: str, candidates: list) -> dict:
    """
    Simple dynamic item-based collaborative filtering from DB interactions.

    Logic:
    - Look at items the user has interacted with.
    - Find other items interacted with by similar users.
    - Score candidates by overlap strength.
    """

    user_id = str(user_id)

    if user_id not in user_seen_items:
        return {item: 0.0 for item in candidates}

    user_history = train[train["user_id"].astype(str) == user_id]

    if user_history.empty:
        return {item: 0.0 for item in candidates}

    raw_scores = defaultdict(float)

    for _, hist_row in user_history.iterrows():
        hist_item = str(hist_row["item_id"])
        hist_weight = float(hist_row["interaction_weight"])

        users_for_hist_item = item_users.get(hist_item, {})

        for candidate in candidates:
            candidate = str(candidate)
            users_for_candidate = item_users.get(candidate, {})

            if not users_for_candidate:
                continue

            common_users = set(users_for_hist_item.keys()) & set(users_for_candidate.keys())

            if not common_users:
                continue

            similarity = len(common_users) / max(
                len(users_for_hist_item),
                len(users_for_candidate),
                1
            )

            raw_scores[candidate] += similarity * hist_weight

    normed = normalize_scores(raw_scores)

    return {item: normed.get(item, 0.0) for item in candidates}


def get_content_scores(user_id: str, candidates: list) -> dict:
    """
    Content-based filtering using TF-IDF similarity.
    Builds a user profile from items the user interacted with.
    """

    user_id = str(user_id)

    history = train[train["user_id"].astype(str) == user_id]

    if history.empty:
        return {item: 0.0 for item in candidates}

    profile = None
    total_weight = 0.0

    for _, row in history.iterrows():
        item_id = str(row["item_id"])
        weight = float(row["interaction_weight"])

        if item_id not in item_to_content_idx:
            continue

        idx = item_to_content_idx[item_id]
        item_vec = tfidf_matrix[idx] * weight

        if profile is None:
            profile = item_vec
        else:
            profile = profile + item_vec

        total_weight += weight

    if profile is None or total_weight == 0:
        return {item: 0.0 for item in candidates}

    profile = profile / total_weight

    candidate_indices = []
    candidate_ids = []

    for item_id in candidates:
        item_id = str(item_id)
        if item_id in item_to_content_idx:
            candidate_ids.append(item_id)
            candidate_indices.append(item_to_content_idx[item_id])

    if not candidate_indices:
        return {item: 0.0 for item in candidates}

    candidate_matrix = tfidf_matrix[candidate_indices]
    raw = cosine_similarity(profile, candidate_matrix).flatten()

    score_map = dict(zip(candidate_ids, raw.tolist()))

    return {item: float(score_map.get(item, 0.0)) for item in candidates}


def hybrid_recommend(user_id: str, top_n: int = 10) -> list:
    user_id = str(user_id)

    seen = user_seen_items.get(user_id, set())
    candidates = [
        str(iid)
        for iid in catalog["item_id"].astype(str).tolist()
        if str(iid) not in seen
    ]

    if not candidates:
        return []

    cf_scores = get_cf_scores(user_id, candidates)
    content_scores = get_content_scores(user_id, candidates)

    results = []

    for iid in candidates:
        cf_s = cf_scores.get(iid, 0.0)
        cb_s = content_scores.get(iid, 0.0)
        pop_s = pop_map.get(iid, 0.0)

        final = ALPHA * cf_s + BETA * cb_s + GAMMA * pop_s

        results.append({
            "item_id": iid,
            "score": round(float(final), 4),
            "cf_score": round(float(cf_s), 4),
            "content_score": round(float(cb_s), 4),
            "pop_score": round(float(pop_s), 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    top = results[:top_n]

    cat_map = catalog.set_index("item_id")[
        ["item_title", "avg_price", "category", "popularity"]
    ].to_dict("index")

    for r in top:
        info = cat_map.get(r["item_id"], {})
        r["item_title"] = str(info.get("item_title", "Unknown"))
        r["category"] = str(info.get("category", ""))
        r["avg_price"] = round(float(info.get("avg_price", 0)), 2)
        r["popularity"] = int(info.get("popularity", 0))

    return top


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Nexus Recommendation API",
        "version": "2.0.0",
        "mode": "database-only",
        "docs": "/docs",
        "endpoints": [
            "/recommend",
            "/popular",
            "/similar/{item_id}",
            "/users",
            "/items",
            "/refresh",
            "/health",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "database-only",
        "catalog_size": len(catalog),
        "interaction_count": len(train),
        "known_users": len(user_seen_items),
        "weights": {
            "view": VIEW_WEIGHT,
            "bid": BID_WEIGHT,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
        },
    }


@app.post("/refresh")
def refresh():
    """
    Manually reload latest DB data.
    Use this after inserting new seed data or new interactions.
    """
    try:
        refresh_data()
        return {
            "status": "refreshed",
            "catalog_size": len(catalog),
            "interaction_count": len(train),
            "known_users": len(user_seen_items),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
def recommend(
    user_id: str = Query(...),
    top_n: int = Query(10, ge=1, le=50),
):
    try:
        recs = hybrid_recommend(user_id, top_n=top_n)

        return {
            "user_id": user_id,
            "cold_start": str(user_id) not in user_seen_items,
            "count": len(recs),
            "recommendations": recs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/popular")
def popular(top_n: int = Query(10, ge=1, le=50)):
    top = catalog.nlargest(top_n, "popularity")[
        ["item_id", "item_title", "category", "avg_price", "popularity"]
    ].copy()

    top["popularity"] = top["popularity"].astype(int)
    top["avg_price"] = top["avg_price"].astype(float)

    return {
        "count": len(top),
        "items": top.to_dict("records"),
    }


@app.get("/similar/{item_id}")
def similar_items(
    item_id: str,
    top_n: int = Query(10, ge=1, le=50),
):
    item_id = str(item_id)

    if item_id not in item_to_content_idx:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")

    idx = item_to_content_idx[item_id]
    vec = tfidf_matrix[idx]
    scores = cosine_similarity(vec, tfidf_matrix).flatten()

    scores[idx] = 0

    top_idx = np.argsort(scores)[::-1][:top_n]

    inv_map = {v: k for k, v in item_to_content_idx.items()}
    cat_map = catalog.set_index("item_id").to_dict("index")

    results = []

    for i in top_idx:
        similar_item_id = inv_map[i]
        info = cat_map.get(similar_item_id, {})

        results.append({
            "item_id": similar_item_id,
            "item_title": str(info.get("item_title", "")),
            "category": str(info.get("category", "")),
            "avg_price": round(float(info.get("avg_price", 0)), 2),
            "similarity": round(float(scores[i]), 4),
        })

    return {
        "seed_item": {
            "item_id": item_id,
            "item_title": cat_map.get(item_id, {}).get("item_title", ""),
        },
        "count": len(results),
        "similar": results,
    }


@app.get("/users")
def list_users(limit: int = Query(20, ge=1, le=200)):
    users = list(user_seen_items.keys())[:limit]
    return {
        "count": len(users),
        "users": users,
    }


@app.get("/items")
def list_items(limit: int = Query(20, ge=1, le=200)):
    items = catalog.head(limit)[
        ["item_id", "item_title", "category", "avg_price", "popularity"]
    ].copy()

    items["popularity"] = items["popularity"].astype(int)
    items["avg_price"] = items["avg_price"].astype(float)

    return {
        "count": len(items),
        "items": items.to_dict("records"),
    }
