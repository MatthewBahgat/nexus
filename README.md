# Nexus Recommendation System

Hybrid recommendation system combining Collaborative Filtering (SVD),
Content-Based Filtering (SBERT), and Popularity scoring.

---

## Folder Structure

```
nexus_recommender/
├── notebooks/
│   ├── 01_data_pipeline.ipynb       ← Download & clean data
│   ├── 02_content_based.ipynb       ← TF-IDF + SBERT content filtering
│   ├── 03_collaborative.ipynb       ← SVD collaborative filtering
│   └── 04_hybrid_and_evaluation.ipynb ← Hybrid model + Precision@K / NDCG@K
├── api/
│   ├── main.py                      ← FastAPI service
│   └── demo.html                    ← Demo website (open in browser)
├── data/
│   ├── raw/                         ← Downloaded dataset goes here (auto)
│   └── processed/                   ← Generated .csv and .pkl files go here (auto)
└── requirements.txt
```

---

## Step 1 — Install Dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

---

## Step 2 — Run the Notebooks IN ORDER

Open Jupyter:

```bash
jupyter notebook
```

Then run each notebook **top to bottom**, one at a time:

1. `01_data_pipeline.ipynb`
   - Downloads the Online Retail II dataset (~20MB)
   - Cleans data and saves train/test split
   - ✅ Done when you see: "Data pipeline complete. Run notebook 02 next."

2. `02_content_based.ipynb`
   - Downloads SBERT model (~90MB, first run only)
   - Builds TF-IDF and SBERT item embeddings
   - ✅ Done when you see: "Saved content_data.pkl"

3. `03_collaborative.ipynb`
   - Trains SVD matrix factorization (no extra installs needed)
   - ✅ Done when you see: "Saved cf_data.pkl"

4. `04_hybrid_and_evaluation.ipynb`
   - Combines all components into the hybrid model
   - Runs Precision@K and NDCG@K evaluation
   - ✅ Done when you see: "Recommendation system complete!"

---

## Step 3 — Start the API

Open a NEW terminal, go into the `api/` folder:

```bash
cd api
python -m uvicorn main:app --reload --port 8000
```

You should see:
```
Loading artifacts...
Loaded. Weights: α=0.5, β=0.3, γ=0.2
Catalog: ~4,000 items | Known users: ~5,800
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

## Step 4 — Open the Demo

Open `api/demo.html` in your browser (just double-click it).

The green dot in the top right means the API is connected.

**Demo modes:**
- 🎯 Personalized — enter a user ID or click "Random user"
- 🔥 Popular Items — top items by interaction score
- 🔗 Similar Items — enter an item ID (e.g. `85123A`)

Auto-generated API docs: http://localhost:8000/docs

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'surprise'` | Re-run notebook 03 — old cf_data.pkl needs to be regenerated |
| `uvicorn not recognized` | Use `python -m uvicorn main:app --reload --port 8000` |
| `FileNotFoundError: cf_data.pkl` | Run notebooks 01→04 first before starting the API |
| SBERT download slow | Normal on first run — it caches after that |
