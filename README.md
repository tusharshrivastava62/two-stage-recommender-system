# Two-Stage Recommender System (Production-Ready)

This repository implements a **production-style two-stage recommender system** commonly used in large-scale machine learning systems (e.g., Netflix, YouTube, Amazon).

The recommendation pipeline is split into:

1. **Candidate Retrieval** (recall-focused)
2. **Learning-to-Rank** (relevance-focused)

This architecture improves **scalability, latency, and ranking quality** compared to single-stage recommenders.

---

## System Architecture

### Stage 1: Candidate Retrieval (ALS)
- Implicit feedback matrix factorization (ALS)
- Learns latent user and item embeddings
- Retrieves a small candidate set from a large item corpus
- Optimized for **high recall** and **low latency**

### Stage 2: Ranking (LightGBM)
- Gradient Boosted Decision Tree ranker
- Trained on user–item interaction features
- Optimized using learning-to-rank objectives
- Produces final ranked recommendations

---

## Project Goals

- Demonstrate **industry-standard recommender system design**
- Separate recall-oriented and relevance-oriented modeling
- Serve recommendations via a **real-time inference API**
- Focus on **ML system design**, not just offline modeling

---

## Data & Feature Engineering

- Implicit user–item interaction signals
- ALS latent vectors (user & item embeddings)
- Ranking features include:
  - ALS similarity scores
  - User/item interaction statistics
  - Popularity-based signals

> Training data, processed datasets, and trained model artifacts are intentionally excluded from Git and generated locally.

---

## Training Pipeline

1. Data preprocessing and interaction matrix creation  
2. Train ALS retrieval model  
3. Generate candidate items per user  
4. Build ranking feature dataset  
5. Train LightGBM ranker  
6. Persist trained models for inference  

---

## Online Inference API

The system exposes a REST API using **FastAPI**.

### Example Request
```bash
curl "http://localhost:8000/recommend?user_id=1&k=10"
