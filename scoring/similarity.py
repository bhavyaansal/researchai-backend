"""
Similarity scoring:
- Lexical: RapidFuzz ratio (character-level, catches near-exact copying)
- Semantic: cosine similarity on embedding vectors (catches paraphrasing)
- Combined: weighted average of both, per config weights
"""
import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from config import settings


def lexical_score(text_a: str, text_b: str) -> float:
    """
    Character-level similarity ratio between two strings, normalized to 0-1.
    RapidFuzz returns 0-100, so we divide by 100.
    """
    return fuzz.ratio(text_a, text_b) / 100.0


def semantic_score(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Cosine similarity between two embedding vectors, normalized to 0-1.
    Embeddings come from sentence-transformers, range -1 to 1; clamp to 0-1.
    """
    sim = cosine_similarity([embedding_a], [embedding_b])[0][0]
    return max(0.0, min(1.0, float(sim)))


def combined_score(lex_score: float, sem_score: float) -> float:
    """Weighted combination of lexical and semantic scores."""
    return (
        lex_score * settings.LEXICAL_WEIGHT
        + sem_score * settings.SEMANTIC_WEIGHT
    )


def global_document_score(
    span_scores: list[float],
    paragraph_lengths: list[int],
    total_document_length: int = 0,
) -> float:
    """
    Compute the overall document similarity score.
    Weighted by span lengths and relative to total_document_length if provided.
    If total_document_length is not provided, falls back to sum of span lengths.
    If there are no flagged spans or text length is 0, score is 0.0.
    """
    if not span_scores or not paragraph_lengths:
        return 0.0

    weighted_sum = sum(s * l for s, l in zip(span_scores, paragraph_lengths))
    denom = total_document_length if total_document_length > 0 else sum(paragraph_lengths)
    if denom == 0:
        return 0.0

    return max(0.0, min(1.0, weighted_sum / denom))

