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

#-----------------------------
    # Use dynamic ceiling: max of either the actual corpus max OR a minimum of 15.0
    # This prevents inflating scores when the corpus has very high BM25 peaks
    # max_raw = max(raw_scores) if len(raw_scores) > 0 else 1.0
    # BM25_REFERENCE_CEILING = max(max_raw, 15.0)
    # MIN_MEANINGFUL_SCORE = 0.5    # raw scores below this are considered no-match

    # scored = []
    # for i, src in enumerate(sources):
    #     bm25_score = (
    #         float(min(raw_scores[i] / BM25_REFERENCE_CEILING, 1.0))
    #         if raw_scores[i] >= MIN_MEANINGFUL_SCORE
    #         else 0.0
    #     )
# -------------------------------------------

    # Fix 2: Dynamic ceiling — use the max of top-3 actual scores.
    # This prevents a fixed constant from over-inflating weak matches
    # on small corpora or under-inflating on large ones.
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

#========================================
        # Only use token_set_ratio as a boosting signal when it's very high (>= 0.75),
        # since token_set_ratio is asymmetric and inflates scores for long queries vs short sources.
        # token_ratio = fuzz.token_set_ratio(query_text, src.sentence_text) / 100.0
        # token_boost = token_ratio if token_ratio >= 0.75 else 0.0
        # # Also check exact token sort ratio for near-duplicate detection
        # sort_ratio = fuzz.token_sort_ratio(query_text, src.sentence_text) / 100.0
        # sort_boost = sort_ratio if sort_ratio >= 0.65 else 0.0

        # final_score = max(bm25_score, token_boost, sort_boost)

        # scored.append({
        #     "source_title": src.source_title,
        #     "source_text": src.sentence_text,
        #     "score": final_score,
        # })

        # scored.sort(key=lambda x: x["score"], reverse=True)
        #     return scored[:top_k]
#========================================

        # Fix 3: Only boost with token_set_ratio when base is already
        # meaningful. Avoids false positives where unrelated sentences
        # share only common stopwords like "the", "is", "of".
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
