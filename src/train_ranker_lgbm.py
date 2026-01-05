from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from metrics import recall_at_k, ndcg_at_k


def make_recs_from_scores(df: pd.DataFrame, k: int) -> tuple[list[list[int]], list[set[int]]]:
    recs = []
    truth = []
    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("pred", ascending=False)
        recs.append(g["item_id"].head(k).tolist())
        truth.append(set(g.loc[g["label"] == 1, "item_id"].tolist()))
    return recs, truth


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "ranker"
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(data_dir / "val_ranker.parquet")
    test = pd.read_parquet(data_dir / "test_ranker.parquet")

    # Features: everything except these
    drop_cols = {"label", "user_id", "item_id", "qid"}
    feature_cols = [c for c in val.columns if c not in drop_cols]

    X_val = val[feature_cols]
    y_val = val["label"]
    group_val = val.groupby("qid").size().to_numpy()

    X_test = test[feature_cols]
    y_test = test["label"]
    group_test = test.groupby("qid").size().to_numpy()

    # Train on VAL data (simple for now; later we can build train-split ranker data too)
    # In practice you'd train on train->val, and test on test.
    train_data = lgb.Dataset(X_val, label=y_val, group=group_val)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=500,
    )

    # Predict
    val["pred"] = model.predict(X_val)
    test["pred"] = model.predict(X_test)

    # Evaluate ranking quality
    for name, df in [("VAL", val), ("TEST", test)]:
        recs, truth = make_recs_from_scores(df, k=100)
        print(f"\n{name} (LightGBM ranker)")
        print(f"Recall@20: {recall_at_k(recs, truth, 20):.4f}   NDCG@20: {ndcg_at_k(recs, truth, 20):.4f}")
        print(f"Recall@50: {recall_at_k(recs, truth, 50):.4f}   NDCG@50: {ndcg_at_k(recs, truth, 50):.4f}")

    # Save
    joblib.dump({"model": model, "feature_cols": feature_cols}, art / "lgbm_ranker.joblib")
    print("\nSaved:", art / "lgbm_ranker.joblib")


if __name__ == "__main__":
    main()
