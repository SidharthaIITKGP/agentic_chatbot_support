# src/rag/embeddings.py
from typing import List
from sentence_transformers import SentenceTransformer

# Load MiniLM once (fast & local)
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts using MiniLM.
    Returns a list of embeddings (each embedding is a Python list of floats).
    """
    if not texts:
        return []
    embeddings = _MODEL.encode(texts, batch_size=32, show_progress_bar=False)
    # Convert numpy arrays → python lists for vectorstore compatibility
    return [emb.tolist() for emb in embeddings]

def embed_query(text: str) -> List[float]:
    """
    Convenience wrapper to embed a single query string.
    """
    return embed_texts([text])[0]
