from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    extract_dir = data_dir / "ml-1m"

    if not extract_dir.exists():
        if not zip_path.exists():
            print(f"Downloading MovieLens 1M -> {zip_path}")
            urlretrieve(ML_1M_URL, zip_path)
        print(f"Extracting -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir / "ml-1m"


def load_ratings(ml_dir: Path) -> pd.DataFrame:
    # ratings.dat format: UserID::MovieID::Rating::Timestamp
    ratings_path = ml_dir / "ratings.dat"
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def time_split_per_user(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user: last interaction -> test, previous -> val, rest -> train.
    """
    g = df.groupby("user_id", group_keys=False)
    test = g.tail(1)
    val = g.apply(lambda x: x.iloc[-2:-1])
    train = df.drop(index=test.index).drop(index=val.index)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    ml_dir = download_movielens_1m(data_dir)
    df = load_ratings(ml_dir)

    # Convert to implicit feedback (common in industry retrieval)
    df["implicit"] = 1.0

    train, val, test = time_split_per_user(df)

    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    print("Saved:")
    print(" ", out_dir / "train.parquet", len(train))
    print(" ", out_dir / "val.parquet", len(val))
    print(" ", out_dir / "test.parquet", len(test))


if __name__ == "__main__":
    main()
