from __future__ import annotations
import numpy as np


def recall_at_k(recs: list[list[int]], truth: list[set[int]], k: int) -> float:
    vals = []
    for r, t in zip(recs, truth):
        if not t:
            continue
        topk = set(r[:k])
        vals.append(len(topk & t) / len(t))
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(recs: list[list[int]], truth: list[set[int]], k: int) -> float:
    vals = []
    for r, t in zip(recs, truth):
        if not t:
            continue

        dcg = 0.0
        for i, item in enumerate(r[:k], start=1):
            if item in t:
                dcg += 1.0 / np.log2(i + 1)

        ideal_hits = min(len(t), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
        vals.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(vals)) if vals else 0.0
