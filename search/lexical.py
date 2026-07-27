"""
Lexical search using rank-bm25 - free, pure Python, no server needed.
This replaces Elasticsearch in the free stack.

NOTE: BM25 index is rebuilt from the SourceDocument table each time this module
loads. For a small/medium corpus this is fast. For a large corpus, you'd want
to persist the index instead of rebuilding it every run.
"""
from rank_bm25 import BM25Okapi
from db.models import SourceDocument


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Good enough for BM25."""
    return text.lower().split()


def build_bm25_index(db_session):
    """
    Build a BM25 index from all source documents in the database.
    Returns (bm25_index, source_docs_list) so callers can map scores back to text.
    """
    sources = db_session.query(SourceDocument).all()
    if not sources:
        return None, []

    tokenized_corpus = [_tokenize(s.sentence_text) for s in sources]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, sources


def search_lexical(query_text: str, bm25_index, sources: list, top_k: int = 5) -> list[dict]:
    """
    Search the BM25 index for the most similar source sentences to query_text.
    Returns top_k matches as [{"source_title":..., "source_text":..., "score": 0-1}, ...]
    """
    if bm25_index is None or not sources:
        return []

    tokenized_query = _tokenize(query_text)
    raw_scores = bm25_index.get_scores(tokenized_query)

    # BM25 scores are unbounded. Max-score normalization alone is misleading when
    # the best raw score is still weak (e.g. only stopwords overlap) - it would
    # inflate that weak match to a perfect 1.0. Use a fixed reference ceiling
    # instead, based on typical BM25 scores for genuinely matching short sentences.
    # Anything below this floor is treated as noise, not a real match.
    BM25_REFERENCE_CEILING = 8.0  # tune this against your real corpus
    MIN_MEANINGFUL_SCORE = 1.0    # raw scores below this are considered no-match

    scored = [
        {
            "source_title": sources[i].source_title,
            "source_text": sources[i].sentence_text,
            "score": (
                float(min(raw_scores[i] / BM25_REFERENCE_CEILING, 1.0))
                if raw_scores[i] >= MIN_MEANINGFUL_SCORE
                else 0.0
            ),
        }
        for i in range(len(sources))
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
