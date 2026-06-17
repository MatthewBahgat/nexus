import json
import os
import base64
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# App Setup

app = FastAPI(
    title="Nexus Recommendation API",
    description="API-driven recommendation system for the Nexus auction platform",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_admin_secret(secret: str | None) -> tuple[str | None, str | None, str | None]:
    if not secret:
        return None, None, None

    secret = secret.strip()

    try:
        parsed = json.loads(secret)
        if isinstance(parsed, dict):
            token = parsed.get("accessToken") or parsed.get("access_token") or parsed.get("token")
            email = parsed.get("email") or parsed.get("username")
            password = parsed.get("password")
            return token, email, password
    except json.JSONDecodeError:
        pass

    if ":" in secret and "@" in secret.split(":", 1)[0]:
        email, password = secret.split(":", 1)
        return None, email.strip(), password.strip()

    return secret, None, None


# Upstream API Config

ADMIN_SECRET = os.getenv("ADMIN") or os.getenv("NEXUS_ADMIN")
ADMIN_TOKEN, ADMIN_EMAIL, ADMIN_PASSWORD = parse_admin_secret(ADMIN_SECRET)
NEXUS_API_BASE_URL = os.getenv("NEXUS_API_BASE_URL", "https://nexus.tidygram.site").rstrip("/")
NEXUS_API_TOKEN = os.getenv("NEXUS_API_TOKEN") or os.getenv("API_BEARER_TOKEN") or ADMIN_TOKEN
NEXUS_API_EMAIL = os.getenv("NEXUS_API_EMAIL") or os.getenv("NEXUS_ADMIN_EMAIL") or ADMIN_EMAIL
NEXUS_API_PASSWORD = os.getenv("NEXUS_API_PASSWORD") or os.getenv("NEXUS_ADMIN_PASSWORD") or ADMIN_PASSWORD
NEXUS_API_TIMEOUT = float(os.getenv("NEXUS_API_TIMEOUT", "20"))
NEXUS_PAGE_LIMIT = int(os.getenv("NEXUS_PAGE_LIMIT", "100"))
NEXUS_INTERACTION_LIMIT = int(os.getenv("NEXUS_INTERACTION_LIMIT", "500"))
NEXUS_AUTO_REFRESH_MINUTES = float(os.getenv("NEXUS_AUTO_REFRESH_MINUTES", "5"))
NEXUS_TOKEN_REFRESH_SKEW_SECONDS = int(os.getenv("NEXUS_TOKEN_REFRESH_SKEW_SECONDS", "60"))


# Recommendation Weights

VIEW_WEIGHT = 1
BID_WEIGHT = 4

ALPHA = 0.55   # default collaborative score
BETA = 0.30    # default content score
GAMMA = 0.15   # default popularity score


# Global Data Cache

catalog = pd.DataFrame()
train = pd.DataFrame()
tfidf_matrix = None
item_to_content_idx = {}
user_seen_items = {}
item_users = {}
pop_map = {}
refresh_lock = threading.Lock()
auto_refresh_started = False
auth_lock = threading.Lock()
auth_token = NEXUS_API_TOKEN
auth_token_exp = 0.0


# API Loading

def jwt_expiry(token: str | None) -> float:
    if not token:
        return 0.0

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        return float(json.loads(decoded).get("exp", 0))
    except Exception:
        return 0.0


auth_token_exp = jwt_expiry(auth_token)


def token_is_valid(token: str | None, exp: float) -> bool:
    if not token:
        return False

    # Non-JWT tokens are accepted because we cannot inspect their expiry.
    if exp <= 0:
        return True

    return time.time() < (exp - NEXUS_TOKEN_REFRESH_SKEW_SECONDS)


def api_post(path: str, body: dict, authenticated: bool = False) -> dict:
    url = urljoin(f"{NEXUS_API_BASE_URL}/", path.lstrip("/"))
    encoded = json.dumps(body).encode("utf-8")
    headers = api_headers(authenticated=authenticated)
    headers["Content-Type"] = "application/json"
    request = Request(url, data=encoded, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=NEXUS_API_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed with {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {path} failed: {exc.reason}") from exc


def get_access_token() -> str | None:
    global auth_token
    global auth_token_exp

    with auth_lock:
        if token_is_valid(auth_token, auth_token_exp):
            return auth_token

        if not NEXUS_API_EMAIL or not NEXUS_API_PASSWORD:
            return auth_token

        payload = api_post(
            "/auth/login",
            {
                "email": NEXUS_API_EMAIL,
                "password": NEXUS_API_PASSWORD,
            },
            authenticated=False,
        )
        token = payload.get("accessToken") or payload.get("access_token")

        if not token:
            raise RuntimeError("Auth login did not return an access token.")

        auth_token = token
        auth_token_exp = jwt_expiry(token)
        return auth_token


def get_access_token_safe() -> str | None:
    try:
        return get_access_token()
    except Exception as exc:
        print(f"Auth token unavailable: {exc}")
        return None


def api_headers(authenticated: bool = False) -> dict:
    headers = {
        "Accept": "application/json",
        "User-Agent": "nexus-recommender/2.1",
    }

    if authenticated:
        token = get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    return headers


def api_get(path: str, params: dict | None = None, authenticated: bool = False) -> dict:
    query = f"?{urlencode(params or {})}" if params else ""
    url = urljoin(f"{NEXUS_API_BASE_URL}/", path.lstrip("/")) + query
    request = Request(url, headers=api_headers(authenticated=authenticated), method="GET")

    try:
        with urlopen(request, timeout=NEXUS_API_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {path} failed with {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {path} failed: {exc.reason}") from exc


def fetch_paginated(
    path: str,
    *,
    limit: int = NEXUS_PAGE_LIMIT,
    authenticated: bool = False,
    extra_params: dict | None = None,
) -> list[dict]:
    records = []
    page = 1

    while True:
        params = {"page": page, "limit": limit}
        if extra_params:
            params.update(extra_params)

        payload = api_get(path, params=params, authenticated=authenticated)
        data = payload.get("data") or payload.get("items") or []
        records.extend(data)

        meta = payload.get("meta") or {}
        total_pages = int(meta.get("totalPages") or meta.get("total_pages") or page)

        if page >= total_pages or not data:
            break

        page += 1

    return records


def first_value(source: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_timestamp(value: Any) -> Any:
    if value:
        return value

    return datetime.now(timezone.utc).isoformat()


def load_categories() -> dict:
    categories = fetch_paginated("/categories")
    return {str(category.get("id")): category for category in categories}


def item_to_catalog_row(item: dict, categories: dict) -> dict:
    category_id = first_value(item, "categoryId", "category_id")
    category = item.get("category") or categories.get(str(category_id), {})

    return {
        "item_id": str(item.get("id")),
        "item_title": str(first_value(item, "name", "title", default="")),
        "description": str(first_value(item, "description", default="") or ""),
        "category": str(first_value(category, "name", default="") or ""),
        "avg_price": as_float(first_value(item, "startingPrice", "starting_price", "buyNowPrice", "buy_now_price")),
        "popularity": 0.0,
    }


def load_catalog_from_api() -> pd.DataFrame:
    categories = load_categories()
    items = fetch_paginated("/items")

    if not items:
        raise RuntimeError("No items returned by the Nexus API.")

    rows = [item_to_catalog_row(item, categories) for item in items]
    db_catalog = pd.DataFrame(rows).drop_duplicates(subset=["item_id"], keep="last")

    if db_catalog.empty:
        raise RuntimeError("No usable items returned by the Nexus API.")

    return db_catalog


def interaction_to_train_row(interaction: dict) -> dict | None:
    user_id = first_value(interaction, "userId", "user_id", "customerId", "customer_id")
    item_id = first_value(interaction, "itemId", "item_id")

    item = interaction.get("item") or {}
    auction = interaction.get("auction") or {}

    if item_id is None:
        item_id = first_value(item, "id")
    if item_id is None:
        item_id = first_value(auction, "itemId", "item_id")

    if user_id is None or item_id is None:
        return None

    interaction_type = str(
        first_value(interaction, "interactionType", "interaction_type", "type", default="view")
    ).lower()
    default_weight = BID_WEIGHT if "bid" in interaction_type else VIEW_WEIGHT
    weight = as_float(
        first_value(
            interaction,
            "interactionWeight",
            "interaction_weight",
            "weight",
            "score",
            default=default_weight,
        ),
        default_weight,
    )

    return {
        "user_id": str(user_id),
        "item_id": str(item_id),
        "item_title": str(first_value(interaction, "itemTitle", "item_title", default=None) or first_value(item, "name", "title", default="")),
        "interaction_weight": weight,
        "interaction_timestamp": normalize_timestamp(
            first_value(
                interaction,
                "interactionTimestamp",
                "interaction_timestamp",
                "createdAt",
                "created_at",
                "timestamp",
            )
        ),
        "price": as_float(first_value(interaction, "price", "amount", "bidAmount", "bid_amount")),
    }


def load_interactions_from_api() -> pd.DataFrame:
    columns = [
        "user_id",
        "item_id",
        "item_title",
        "interaction_weight",
        "interaction_timestamp",
        "price",
    ]

    if not get_access_token_safe():
        return pd.DataFrame(columns=columns)

    interactions = fetch_paginated(
        "/recommender-interactions",
        limit=NEXUS_INTERACTION_LIMIT,
        authenticated=True,
    )
    rows = []

    for interaction in interactions:
        row = interaction_to_train_row(interaction)
        if row:
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def load_auction_popularity() -> dict:
    popularity = defaultdict(float)

    try:
        auctions = fetch_paginated("/auctions")
    except RuntimeError:
        return {}

    for auction in auctions:
        item_id = first_value(auction, "itemId", "item_id")
        if item_id is None:
            continue

        status = str(first_value(auction, "status", default="")).lower()
        status_boost = {
            "live": 3.0,
            "ended": 2.0,
            "scheduled": 1.0,
        }.get(status, 0.5)
        highest_price = as_float(first_value(auction, "currentHighestPrice", "current_highest_price"))
        reserve_price = as_float(first_value(auction, "reservePrice", "reserve_price"))

        popularity[str(item_id)] += status_boost + max(highest_price, reserve_price, 0.0) / 1000.0

    return dict(popularity)


def load_from_api():
    """
    Load item catalog and recommendation interactions from the deployed Nexus API.
    Public catalog data comes from /items and /categories. Personalized interaction
    data comes from /recommender-interactions when NEXUS_API_TOKEN is provided.
    """

    db_catalog = load_catalog_from_api()
    db_train = load_interactions_from_api()

    if not db_train.empty:
        popularity = (
            db_train.groupby("item_id")["interaction_weight"]
            .sum()
            .reset_index()
            .rename(columns={"interaction_weight": "popularity"})
        )

        db_catalog = db_catalog.drop(columns=["popularity"], errors="ignore")
        db_catalog = db_catalog.merge(popularity, on="item_id", how="left")
        db_catalog["popularity"] = db_catalog["popularity"].fillna(0.0)
    else:
        auction_popularity = load_auction_popularity()
        db_catalog["popularity"] = (
            db_catalog["item_id"].astype(str).map(auction_popularity).fillna(0.0)
        )

    return db_catalog, db_train


def rebuild_indexes():
    """
    Build in-memory structures from latest API data.
    """

    global tfidf_matrix
    global item_to_content_idx
    global user_seen_items
    global item_users
    global pop_map

    if catalog.empty:
        raise RuntimeError("Catalog is empty.")

    content_text = (
        catalog["item_title"].fillna("") + " " +
        catalog["description"].fillna("") + " " +
        catalog["category"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(content_text)

    item_to_content_idx = {
        item_id: idx
        for idx, item_id in enumerate(catalog["item_id"].astype(str).tolist())
    }

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

    pop_map = dict(zip(catalog["item_id"].astype(str), catalog["popularity_norm"]))


def refresh_data():
    """
    Reload API data and rebuild recommendation indexes.
    """

    global catalog
    global train

    with refresh_lock:
        catalog, train = load_from_api()
        rebuild_indexes()

    print(f"API loaded: {len(catalog):,} items | {len(train):,} interactions")


# Load once at startup
refresh_data()


def auto_refresh_loop():
    interval_seconds = max(NEXUS_AUTO_REFRESH_MINUTES, 0) * 60

    if interval_seconds <= 0:
        print("Auto refresh disabled.")
        return

    while True:
        time.sleep(interval_seconds)

        try:
            refresh_data()
            print(f"Auto refresh complete. Next refresh in {NEXUS_AUTO_REFRESH_MINUTES:g} minutes.")
        except Exception as exc:
            print(f"Auto refresh failed: {exc}")


@app.on_event("startup")
def start_auto_refresh():
    global auto_refresh_started

    if auto_refresh_started or NEXUS_AUTO_REFRESH_MINUTES <= 0:
        return

    auto_refresh_started = True
    thread = threading.Thread(target=auto_refresh_loop, daemon=True)
    thread.start()
    print(f"Auto refresh enabled every {NEXUS_AUTO_REFRESH_MINUTES:g} minutes.")


# Recommendation Helpers

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


def get_dynamic_weights(user_id: str) -> dict:
    """
    Shift the blend smoothly by how much behavior we know for this user.
    More history raises collaborative confidence; sparse users lean on content
    and popularity until there is enough personal signal.
    """

    user_id = str(user_id)
    history_count = len(user_seen_items.get(user_id, set()))

    if history_count == 0:
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 1.0,
            "history_count": history_count,
            "interaction_count": 0,
            "total_interaction_weight": 0.0,
            "confidence": 0.0,
            "profile": "cold_start",
        }

    user_history = train[train["user_id"].astype(str) == user_id]
    interaction_count = len(user_history)
    total_weight = float(user_history["interaction_weight"].sum()) if not user_history.empty else 0.0

    item_confidence = 1.0 - np.exp(-history_count / 6.0)
    event_confidence = 1.0 - np.exp(-interaction_count / 12.0)
    weight_confidence = 1.0 - np.exp(-total_weight / 20.0)
    confidence = float(
        0.60 * item_confidence +
        0.25 * event_confidence +
        0.15 * weight_confidence
    )

    alpha = 0.05 + 0.60 * confidence
    beta = 0.55 - 0.25 * confidence
    gamma = 1.0 - alpha - beta

    if confidence < 0.25:
        profile = "new_user"
    elif confidence < 0.65:
        profile = "warming_up"
    else:
        profile = "active_user"

    return {
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "gamma": round(float(gamma), 4),
        "history_count": history_count,
        "interaction_count": interaction_count,
        "total_interaction_weight": round(total_weight, 4),
        "confidence": round(confidence, 4),
        "profile": profile,
    }


def get_cf_scores(user_id: str, candidates: list) -> dict:
    """
    Simple dynamic item-based collaborative filtering from API interactions.
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
                1,
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
    weights = get_dynamic_weights(user_id)

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

        final = (
            weights["alpha"] * cf_s +
            weights["beta"] * cb_s +
            weights["gamma"] * pop_s
        )

        results.append({
            "item_id": iid,
            "score": round(float(final), 4),
            "cf_score": round(float(cf_s), 4),
            "content_score": round(float(cb_s), 4),
            "pop_score": round(float(pop_s), 4),
            "weights": {
                "alpha": weights["alpha"],
                "beta": weights["beta"],
                "gamma": weights["gamma"],
            },
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
        r["popularity"] = round(float(info.get("popularity", 0)), 4)

    return top


# Routes

@app.get("/")
def root():
    return {
        "service": "Nexus Recommendation API",
        "version": "2.1.0",
        "mode": "api",
        "upstream_api": NEXUS_API_BASE_URL,
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
        "mode": "api",
        "upstream_api": NEXUS_API_BASE_URL,
        "catalog_size": len(catalog),
        "interaction_count": len(train),
        "known_users": len(user_seen_items),
        "authenticated_interactions": bool(get_access_token_safe()),
        "auth_mode": "token" if NEXUS_API_TOKEN else "login" if NEXUS_API_EMAIL else "none",
        "admin_secret_configured": bool(ADMIN_SECRET),
        "auto_refresh_minutes": NEXUS_AUTO_REFRESH_MINUTES,
        "default_weights": {
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
    Manually reload latest API data.
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
        weights = get_dynamic_weights(user_id)
        recs = hybrid_recommend(user_id, top_n=top_n)

        return {
            "user_id": user_id,
            "cold_start": str(user_id) not in user_seen_items,
            "profile": weights["profile"],
            "history_count": weights["history_count"],
            "interaction_count": weights["interaction_count"],
            "total_interaction_weight": weights["total_interaction_weight"],
            "confidence": weights["confidence"],
            "weights": {
                "alpha": weights["alpha"],
                "beta": weights["beta"],
                "gamma": weights["gamma"],
            },
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

    top["popularity"] = top["popularity"].astype(float)
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

    items["popularity"] = items["popularity"].astype(float)
    items["avg_price"] = items["avg_price"].astype(float)

    return {
        "count": len(items),
        "items": items.to_dict("records"),
    }
