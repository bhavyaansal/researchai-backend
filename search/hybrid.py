"""
Hybrid search: merges lexical (BM25) and semantic (vector) search results
into a single ranked list, deduplicated by source text.
"""
from search.lexical import search_lexical
from search.semantic import search_semantic
from scoring.similarity import combined_score
from config import settings


def hybrid_search(query_text: str, bm25_index, sources: list, top_k: int = 5) -> list[dict]:
    """
    Run both lexical and semantic search for a single paragraph,
    merge results by matching source_text, and combine scores.

    Returns the single best match (highest combined score), or None if no matches.
    Structure: {
        "source_title": str, "source_text": str,
        "lexical_score": float, "semantic_score": float, "combined_score": float
    }
    """
    lexical_results = search_lexical(query_text, bm25_index, sources, top_k=top_k)
    semantic_results = search_semantic(query_text, top_k=top_k)

    # Index lexical results by source_text for quick lookup
    lex_by_text = {r["source_text"]: r["score"] for r in lexical_results}
    sem_by_text = {r["source_text"]: r["score"] for r in semantic_results}

    all_texts = set(lex_by_text.keys()) | set(sem_by_text.keys())
    if not all_texts:
        return None

    merged = []
    title_lookup = {r["source_text"]: r["source_title"] for r in lexical_results + semantic_results}

    for text in all_texts:
        lex_s = lex_by_text.get(text, 0.0)
        sem_s = sem_by_text.get(text, 0.0)

        # Ignore weak semantic noise when there is zero lexical overlap
        if lex_s == 0.0 and sem_s < 0.50:
            sem_s = 0.0

        c_score = combined_score(lex_s, sem_s)
        if c_score >= settings.SIMILARITY_FLAG_THRESHOLD:
            merged.append({
                "source_title": title_lookup.get(text, "unknown"),
                "source_text": text,
                "lexical_score": lex_s,
                "semantic_score": sem_s,
                "combined_score": c_score,
            })

    merged.sort(key=lambda x: x["combined_score"], reverse=True)
    return merged[0] if merged else None

