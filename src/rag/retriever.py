# src/rag/retriever.py
"""
Retriever with explicit recomputed semantic similarity.

- Loads FAISS index (as before)
- Uses FAISS to fetch candidate docs (fetch_k)
- Re-embeds candidate texts with MiniLM and computes cosine(sim)
- Computes lexical overlap score
- Computes combined score = alpha*semantic + (1-alpha)*lexical
- Returns top_k entries and logs retrievals
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
import numpy as np
from numpy.linalg import norm

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# Paths
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_index"
LOGS_DIR = CURRENT_DIR / "logs"
RETRIEVAL_LOG = LOGS_DIR / "retrieval_logs.jsonl"
os.makedirs(LOGS_DIR, exist_ok=True)

# Embedding model same as ingestion
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return [a.tolist() for a in arr]

    def embed_query(self, text: str) -> List[float]:
        arr = self.model.encode([text], batch_size=1, show_progress_bar=False)
        return arr[0].tolist()

    def __call__(self, text):
        if isinstance(text, (list, tuple)):
            return self.embed_documents(list(text))
        return self.embed_query(text)

# helpers
def _cosine_sim(a: List[float], b: List[float]) -> float:
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    norm_a = norm(arr_a)
    norm_b = norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))

def _lexical_score(query: str, doc_text: str) -> float:
    q_tokens = {t.lower().strip(".,;:!?\"'()[]") for t in query.split() if t.strip()}
    d_tokens = {t.lower().strip(".,;:!?\"'()[]") for t in doc_text.split() if t.strip()}
    if not q_tokens:
        return 0.0
    overlap = q_tokens.intersection(d_tokens)
    return len(overlap) / len(q_tokens)

def _load_faiss_store(embeddings):
    if not FAISS_DIR.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_DIR}. Run ingestion first.")
    store = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return store

def retrieve_policy(
    query: str,
    fetch_k: int = 10,
    top_k: int = 3,
    alpha: float = 0.85,
) -> Dict[str, Any]:
    """
    Returns:
    {
      "query": str,
      "final": [ {text, metadata, sem_sim, lexical, combined} ],
      "candidates": [...],
      "confidence": float
    }
    """
    embeddings = MiniLMEmbeddings()
    store = _load_faiss_store(embeddings)

    # 1) Use FAISS to get candidate documents (no reliance on its numeric score)
    try:
        docs = store.similarity_search(query, k=fetch_k)
    except Exception:
        try:
            cand_with_scores = store.similarity_search_with_score(query, k=fetch_k)
            docs = [d for d, _ in cand_with_scores]
        except Exception:
            docs = []

    if not docs:
        return {"query": query, "final": [], "candidates": [], "confidence": 0.0}

    # 2) Recompute semantic similarity properly:
    # embed query and candidate texts with the same model
    query_emb = embeddings.embed_query(query)
    candidate_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    candidate_embs = embeddings.embed_documents(candidate_texts)

    sem_sims = [_cosine_sim(query_emb, emb) for emb in candidate_embs]  # values in [-1,1]
    # map to [0,1]
    sem_sims_mapped = [max(0.0, (s + 1.0) / 2.0) for s in sem_sims]

    # 3) Lexical scores
    lex_scores = [_lexical_score(query, txt) for txt in candidate_texts]

    # 4) normalize semantic & lexical to 0..1
    def _norm(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return []
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return [1.0 for _ in arr]
        return ((arr - mn) / (mx - mn)).tolist()

    sem_norm = _norm(sem_sims_mapped)
    lex_norm = _norm(lex_scores)

    combined = [alpha * s + (1 - alpha) * l for s, l in zip(sem_norm, lex_norm)]

    entries = []
    for d, s_raw, s_n, l_n, c in zip(docs, sem_sims_mapped, sem_norm, lex_norm, combined):
        text = d.page_content if hasattr(d, "page_content") else str(d)
        metadata = d.metadata if hasattr(d, "metadata") else {}
        entries.append({
            "text": text,
            "metadata": metadata,
            "semantic_raw": float(s_raw),
            "semantic_norm": float(s_n),
            "lexical_norm": float(l_n),
            "combined": float(c),
        })

    entries.sort(key=lambda x: x["combined"], reverse=True)
    final = entries[:top_k]
    confidence = final[0]["combined"] if final else 0.0

    # log
    try:
        log = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "fetch_k": fetch_k,
            "top_k": top_k,
            "alpha": alpha,
            "final_doc_ids": [f.get("metadata", {}).get("doc_id") for f in final],
            "confidence": float(confidence),
        }
        with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
    except Exception:
        pass

    return {"query": query, "final": final, "candidates": entries, "confidence": float(confidence)}

# quick CLI
if __name__ == "__main__":
    q = "How long do refunds take according to policy?"
    out = retrieve_policy(q, fetch_k=10, top_k=3, alpha=0.85)
    print("Query:", out["query"])
    print("Confidence:", out["confidence"])
    print("\nTop results:")
    for i, r in enumerate(out["final"], 1):
        print(f"\n== Result {i} (combined={r['combined']:.3f}) ==")
        print("doc_id:", r["metadata"].get("doc_id"))
        print(r["text"][:800].strip())
