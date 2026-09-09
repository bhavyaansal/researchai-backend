"""
Lightweight text embedder and semantic similarity calculator.
Uses character n-gram TF-IDF vectors for fast in-process similarity computation without external model downloads.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute character n-gram TF-IDF cosine similarity between two texts."""
    if not text_a or not text_b:
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vec.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return max(0.0, min(1.0, float(sim)))
    except Exception:
        return 0.0


def embed_text(text: str) -> np.ndarray:
    """Encode a single string into a TF-IDF vector."""
    if not text:
        return np.zeros(100)
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        return vec.fit_transform([text]).toarray()[0]
    except Exception:
        return np.zeros(100)


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Encode a list of strings into TF-IDF vectors."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        matrix = vec.fit_transform(texts).toarray()
        return [row for row in matrix]
    except Exception:
        return [np.zeros(100) for _ in texts]

