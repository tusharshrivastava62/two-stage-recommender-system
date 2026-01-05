from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from metrics import recall_at_k, ndcg_at_k


def make_id_maps(df: pd.DataFrame):
    """
    Build contiguous index mappings from raw user_id/item_id (train only).
    """
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())

    u2i = {u: i for i, u in enumerate(users)}
    it2i = {it: i for i, it in enumerate(items)}
    i2item = {i: it for it, i in it2i.items()}  # inverse mapping for item indices

    return u2i, it2i, i2item


def build_user_item_matrix(df: pd.DataFrame, u2i, it2i) -> csr_matrix:
    """
    Build a user-item implicit feedback sparse matrix.
    """
    rows = df["user_id"].map(u2i).to_numpy()
    cols = df["item_id"].map(it2i).to_numpy()
    vals = df["implicit"].to_numpy(dtype=np.float32)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(u2i), len(it2i)))
    return mat


def truth_dict(df_holdout: pd.DataFrame) -> dict[int, set[int]]:
    """
    For each user, holdout truth items (usually 1 item in our split).
    """
    out: dict[int, set[int]] = {}
    for u, g in df_holdout.groupby("user_id"):
        out[u] = set(g["item_id"].tolist())
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    # Build mappings on TRAIN only (realistic)
    u2i, it2i, i2item = make_id_maps(train)

    # Build train sparse matrix
    train_mat = build_user_item_matrix(train, u2i, it2i)

    # implicit ALS expects item-user matrix for efficient fitting
    item_user = train_mat.T.tocsr()

    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=20,
        random_state=42,
    )
    model.fit(item_user)

    # Track already seen items per user for filtering
    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    def recommend(user_ids: list[int], k: int = 100) -> list[list[int]]:
        """
        Recommend top-k items per user, filtering train-seen items.
        Robust to any unexpected item indices.
        """
        recs: list[list[int]] = []

        for u in user_ids:
            if u not in u2i:
                recs.append([])
                continue

            uid = u2i[u]

            # Filter items the user already interacted with in train
            seen = train_items_by_user.get(u, set())
            seen_idx = [it2i[it] for it in seen if it in it2i]

            ids, _scores = model.recommend(
                userid=uid,
                user_items=train_mat[uid],
                N=k,
                filter_items=seen_idx,
                recalculate_user=True,
            )

            # ✅ Robust mapping: skip any indices not in i2item
            clean: list[int] = []
            for idx in ids:
                if idx in i2item:
                    clean.append(i2item[idx])

            recs.append(clean)

        return recs

    def eval_split(name: str, holdout: pd.DataFrame):
        td = truth_dict(holdout)

        # Evaluate only on users seen in train
        users = [u for u in td.keys() if u in u2i]
        truth = [td[u] for u in users]

        recs = recommend(users, k=100)

        print(f"\n{name} (ALS retrieval)")
        print(f"Users evaluated: {len(users)}")
        print(f"Recall@20: {recall_at_k(recs, truth, 20):.4f}   NDCG@20: {ndcg_at_k(recs, truth, 20):.4f}")
        print(f"Recall@50: {recall_at_k(recs, truth, 50):.4f}   NDCG@50: {ndcg_at_k(recs, truth, 50):.4f}")

    eval_split("VAL", val)
    eval_split("TEST", test)

    # Save model + mappings for later API use
    import joblib

    joblib.dump(
        {"model": model, "u2i": u2i, "it2i": it2i, "i2item": i2item},
        art / "als_retrieval.joblib",
    )
    print("\nSaved:", art / "als_retrieval.joblib")


if __name__ == "__main__":
    main()
