"""
Lexical search using rank-bm25 - free, pure Python, no server needed.
This replaces Elasticsearch in the free stack.

NOTE: BM25 index is rebuilt from the SourceDocument table each time this module
loads. For a small/medium corpus this is fast. For a large corpus, you'd want
to persist the index instead of rebuilding it every run.
"""
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from db.models import SourceDocument


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Good enough for BM25."""
    return text.lower().split()


def build_bm25_index(db_session):
    """
    Build a BM25 index from all source documents in the database.
    Returns (bm25_index, source_docs_list) so callers can map scores back to text.
    """
    sources = db_session.query(SourceDocument).filter(
        SourceDocument.is_user_upload == 0
    ).all()
    if not sources:
        return None, []

    tokenized_corpus = [_tokenize(s.sentence_text) for s in sources]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, sources


def search_lexical(query_text: str, bm25_index, sources: list, top_k: int = 5) -> list[dict]:
    """
    Search the BM25 index and token set similarity for the most similar source sentences to query_text.
    Returns top_k matches as [{"source_title":..., "source_text":..., "score": 0-1}, ...]
    """
    if bm25_index is None or not sources:
        return []

    tokenized_query = _tokenize(query_text)
    raw_scores = bm25_index.get_scores(tokenized_query)

    sorted_raw = sorted(raw_scores, reverse=True)
    top_scores = [s for s in sorted_raw[:3] if s > 0]
    if not top_scores:
        return []  # no real BM25 signal at all — nothing matched

    ceiling = max(top_scores)
    MIN_MEANINGFUL_RAW = 0.5  # raw scores below this are pure noise

    results = []
    for i, raw in enumerate(raw_scores):
        if raw < MIN_MEANINGFUL_RAW:
            continue

        base_score = float(min(raw / ceiling, 1.0))

        if base_score > 0.65:
            fuzzy_score = fuzz.token_set_ratio(
                query_text.lower(),
                sources[i].sentence_text.lower()
            ) / 100.0
            # Blend: 70% BM25, 30% fuzzy
            final_score = 0.7 * base_score + 0.3 * fuzzy_score
        else:
            final_score = base_score

        results.append({
            "source_title": sources[i].source_title,
            "source_text": sources[i].sentence_text,
            "score": round(final_score, 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
