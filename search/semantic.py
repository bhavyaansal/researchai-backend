"""
Simplified semantic search using TF-IDF instead of ChromaDB.
ChromaDB causes silent crashes on Railway free tier (512MB RAM).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db.models import SessionLocal, SourceDocument

_vectorizer = None
_source_texts = []
_source_meta = []


def _build_index():
    global _vectorizer, _source_texts, _source_meta
    db = SessionLocal()
    try:
        sources = db.query(SourceDocument).filter(
            SourceDocument.is_user_upload == 0
        ).all()
        if not sources:
            return
        _source_texts = [s.sentence_text for s in sources]
        _source_meta = [{"title": s.source_title, "text": s.sentence_text} for s in sources]
        _vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
        _vectorizer.fit(_source_texts)
    finally:
        db.close()


def index_source_documents(sources: list[dict]):
    """Called during seeding — just rebuilds the in-memory index."""
    _build_index()


def search_semantic(query_text: str, top_k: int = 5) -> list[dict]:
    global _vectorizer, _source_texts, _source_meta
    if _vectorizer is None:
        _build_index()
    if not _source_texts:
        return []
    try:
        query_vec = _vectorizer.transform([query_text])
        corpus_vecs = _vectorizer.transform(_source_texts)
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append({
                    "source_title": _source_meta[i]["title"],
                    "source_text": _source_meta[i]["text"],
                    "score": float(scores[i]),
                })
        return results
    except Exception:
        return []