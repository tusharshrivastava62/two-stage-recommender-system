from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from scipy.sparse import csr_matrix

app = FastAPI(title="Two-Stage Recommender", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROC = ROOT / "data" / "processed"
ML_DIR = ROOT / "data" / "ml-1m" / "ml-1m"  # extracted folder


# Globals loaded at startup
_als = None
_ranker = None
_feature_cols: List[str] = []
_train_mat: csr_matrix | None = None

_train_items_by_user: Dict[int, set] = {}
_item_pop: Dict[int, int] = {}
_user_act: Dict[int, int] = {}
_item_genre_feats: pd.DataFrame | None = None  # item_id + genre_*
_genre_cols: List[str] = []


def build_user_item_matrix(train: pd.DataFrame, u2i, it2i) -> csr_matrix:
    rows = train["user_id"].map(u2i).to_numpy()
    cols = train["item_id"].map(it2i).to_numpy()
    vals = np.ones(len(train), dtype=np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(len(u2i), len(it2i)))


def load_movies(ml_dir: Path) -> pd.DataFrame:
    movies_path = ml_dir / "movies.dat"
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        dtype={"item_id": int, "title": str, "genres": str},
        encoding="latin-1",
    )
    return movies


def make_genre_vocab(movies: pd.DataFrame) -> List[str]:
    vocab = set()
    for g in movies["genres"].fillna("").tolist():
        for part in g.split("|"):
            if part:
                vocab.add(part)
    return sorted(vocab)


def genre_onehot(movies: pd.DataFrame, vocab: List[str]) -> pd.DataFrame:
    vocab_index = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(movies), len(vocab)), dtype=np.int8)

    for r, g in enumerate(movies["genres"].fillna("").tolist()):
        for part in g.split("|"):
            if part in vocab_index:
                mat[r, vocab_index[part]] = 1

    out = pd.DataFrame(mat, columns=[f"genre_{g}" for g in vocab])
    out.insert(0, "item_id", movies["item_id"].values)
    return out


@app.on_event("startup")
def load_artifacts():
    global _als, _ranker, _feature_cols, _train_mat
    global _train_items_by_user, _item_pop, _user_act, _item_genre_feats, _genre_cols

    # Load models
    _als = joblib.load(ART / "als_retrieval.joblib")
    _ranker = joblib.load(ART / "lgbm_ranker.joblib")  # {"model":..., "feature_cols":...}
    _feature_cols = list(_ranker["feature_cols"])

    # Load train interactions (for filtering + features)
    train = pd.read_parquet(PROC / "train.parquet")

    u2i = _als["u2i"]
    it2i = _als["it2i"]

    _train_mat = build_user_item_matrix(train, u2i, it2i)
    _train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    _item_pop = train["item_id"].value_counts().to_dict()
    _user_act = train["user_id"].value_counts().to_dict()

    # Genre features (must match training)
    movies = load_movies(ML_DIR)
    vocab = make_genre_vocab(movies)  # should be 18 for ML-1M
    _item_genre_feats = genre_onehot(movies, vocab)
    _genre_cols = [c for c in _item_genre_feats.columns if c.startswith("genre_")]

    # Safety check: ensure ranker expects these genre columns (it should)
    # If it doesn't, we still proceed; we will build exactly _feature_cols later.


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    n_candidates: int = Query(200, ge=50, le=500),
):
    """
    Two-stage inference:
    1) ALS retrieval generates candidates
    2) Build features
    3) LightGBM ranker reranks candidates
    4) Return top-k item_ids
    """
    als_model = _als["model"]
    u2i = _als["u2i"]
    it2i = _als["it2i"]
    i2item = _als["i2item"]

    if user_id not in u2i:
        return {"user_id": user_id, "k": k, "items": [], "note": "unknown user_id"}

    uid = u2i[user_id]

    seen = _train_items_by_user.get(user_id, set())
    seen_idx = [it2i[it] for it in seen if it in it2i]

    ids, scores = als_model.recommend(
        userid=uid,
        user_items=_train_mat[uid],
        N=n_candidates,
        filter_items=seen_idx,
        recalculate_user=True,
    )

    # Build candidate table (item_id + als_score)
    cand_items = []
    cand_scores = []
    for idx, sc in zip(ids, scores):
        if idx in i2item:
            cand_items.append(int(i2item[idx]))
            cand_scores.append(float(sc))

    if not cand_items:
        return {"user_id": user_id, "k": k, "items": []}

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "item_id": cand_items,
            "als_score": cand_scores,
            "item_pop": [int(_item_pop.get(it, 0)) for it in cand_items],
            "user_act": int(_user_act.get(user_id, 0)),
        }
    )

    # Join genre one-hots
    df = df.merge(_item_genre_feats, on="item_id", how="left")
    df.fillna(0, inplace=True)

    # Build feature matrix in EXACT order expected by the ranker
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0  # safety: missing feature -> 0

    X = df[_feature_cols]
    preds = _ranker["model"].predict(X)

    df["pred"] = preds
    df = df.sort_values("pred", ascending=False)

    top_items = df["item_id"].head(k).tolist()

    return {
        "user_id": user_id,
        "k": k,
        "items": [int(x) for x in top_items],
        "meta": {
            "candidates": len(cand_items),
            "features": len(_feature_cols),
        },
    }
