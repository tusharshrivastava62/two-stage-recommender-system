from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix


def load_movies(ml_dir: Path) -> pd.DataFrame:
    # movies.dat format: MovieID::Title::Genres
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


def make_genre_vocab(movies: pd.DataFrame) -> list[str]:
    vocab = set()
    for g in movies["genres"].fillna("").tolist():
        for part in g.split("|"):
            if part:
                vocab.add(part)
    return sorted(vocab)


def genre_onehot(movies: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    vocab_index = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(movies), len(vocab)), dtype=np.int8)

    for r, g in enumerate(movies["genres"].fillna("").tolist()):
        for part in g.split("|"):
            if part in vocab_index:
                mat[r, vocab_index[part]] = 1

    out = pd.DataFrame(mat, columns=[f"genre_{g}" for g in vocab])
    out.insert(0, "item_id", movies["item_id"].values)
    return out


def build_user_item_matrix(train: pd.DataFrame, u2i, it2i) -> csr_matrix:
    """
    Rebuild the same user-item CSR matrix shape that ALS expects at inference.
    (We build it from TRAIN only to avoid leakage.)
    """
    rows = train["user_id"].map(u2i).to_numpy()
    cols = train["item_id"].map(it2i).to_numpy()
    vals = np.ones(len(train), dtype=np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(len(u2i), len(it2i)))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    ml_dir = root / "data" / "ml-1m" / "ml-1m"  # extracted folder
    art = root / "artifacts"
    out_dir = root / "data" / "ranker"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load interactions
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    # Load ALS model bundle
    bundle = joblib.load(art / "als_retrieval.joblib")
    model = bundle["model"]
    u2i = bundle["u2i"]
    it2i = bundle["it2i"]
    i2item = bundle["i2item"]

    # ✅ Build the CSR matrix required by implicit.recommend
    train_mat = build_user_item_matrix(train, u2i, it2i)

    # Feature lookups
    item_pop = train["item_id"].value_counts().to_dict()  # popularity
    user_act = train["user_id"].value_counts().to_dict()  # activity
    train_items_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()

    # Genre features
    movies = load_movies(ml_dir)
    vocab = make_genre_vocab(movies)
    item_genres = genre_onehot(movies, vocab)  # one-hot by item_id

    # Truth mappings (val/test: 1 held-out item per user)
    val_truth = val.groupby("user_id")["item_id"].apply(lambda x: set(x.tolist())).to_dict()
    test_truth = test.groupby("user_id")["item_id"].apply(lambda x: set(x.tolist())).to_dict()

    rng = np.random.default_rng(42)

    def build_split(name: str, truth: dict[int, set[int]], n_candidates: int = 200, n_neg: int = 50):
        rows = []
        users = [u for u in truth.keys() if u in u2i]

        # Random negative pool from TRAIN items only (keeps it realistic)
        all_train_items = np.array(list(it2i.keys()), dtype=int)

        for u in users:
            uid = u2i[u]
            seen = train_items_by_user.get(u, set())
            seen_idx = [it2i[it] for it in seen if it in it2i]

            # ✅ ALS candidates + scores (must pass user_items as CSR row)
            ids, scores = model.recommend(
                userid=uid,
                user_items=train_mat[uid],
                N=n_candidates,
                filter_items=seen_idx,
                recalculate_user=True,
            )

            # Convert ALS internal indices -> raw item_id (robust)
            cand_items = []
            cand_scores = []
            for idx, sc in zip(ids, scores):
                if idx in i2item:
                    cand_items.append(i2item[idx])
                    cand_scores.append(float(sc))

            pos_items = truth[u]

            # Negatives: sample items not seen and not positive
            # (fast mask-based sampling)
            # If pool is too small, fallback to smaller sample.
            mask = np.ones(len(all_train_items), dtype=bool)
            if seen:
                # remove seen
                seen_arr = np.array(list(seen), dtype=int)
                mask &= ~np.isin(all_train_items, seen_arr)
            if pos_items:
                pos_arr = np.array(list(pos_items), dtype=int)
                mask &= ~np.isin(all_train_items, pos_arr)
            neg_pool = all_train_items[mask]

            if len(neg_pool) > 0:
                neg_items = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False).tolist()
            else:
                neg_items = []

            # Labeled candidate rows
            for it, sc in zip(cand_items, cand_scores):
                rows.append(
                    {
                        "user_id": u,
                        "item_id": it,
                        "label": 1 if it in pos_items else 0,
                        "als_score": sc,
                        "item_pop": item_pop.get(it, 0),
                        "user_act": user_act.get(u, 0),
                    }
                )

            # Extra negatives
            for it in neg_items:
                rows.append(
                    {
                        "user_id": u,
                        "item_id": int(it),
                        "label": 0,
                        "als_score": 0.0,
                        "item_pop": item_pop.get(int(it), 0),
                        "user_act": user_act.get(u, 0),
                    }
                )

        df = pd.DataFrame(rows)

        # Join genre one-hots
        df = df.merge(item_genres, on="item_id", how="left")
        df.fillna(0, inplace=True)

        # Group/query id for ranking (group by user)
        df["qid"] = df["user_id"]

        out_path = out_dir / f"{name.lower()}_ranker.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {name} ranker data -> {out_path}  rows={len(df)} users={len(users)}")

    build_split("VAL", val_truth)
    build_split("TEST", test_truth)


if __name__ == "__main__":
    main()
