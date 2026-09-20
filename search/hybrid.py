"""
Hybrid search: merges lexical (BM25) and semantic (vector) search results
into a single ranked list, deduplicated by source text.

Falls back to a live web search (via a self-hosted SearXNG instance) when
the local corpus has no meaningful match, so uploads get checked against
real web content — not just the small manually-curated SourceDocument
table. The fallback is budgeted by the caller (see WebBudget in
tasks/pipeline_task.py) so one document can't trigger unlimited slow
network calls.
"""
from rapidfuzz import fuzz

from search.lexical import search_lexical
from search.semantic import search_semantic
from search.web_search import web_search, TavilyNotConfiguredError
from scoring.similarity import combined_score
from config import settings


def _web_fallback_search(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Query the web via Tavily and score each returned snippet against
    query_text using the same fuzzy-matching style search_lexical uses.
    Returns candidates already shaped like local search results, plus a
    source_url field. Returns [] if Tavily isn't configured or the
    request fails for any reason — a broken/unset web search should
    never crash a scan.
    """
    try:
        web_results = web_search(query_text, max_results=top_k)
    except Exception as e:  # Catches TavilyNotConfiguredError and any other error
        print(f"[web_search] FAILED for query {query_text[:60]!r}: {type(e).__name__}: {e}")
        return []

    candidates = []
    for r in web_results:
        content = r.get("content", "")
        if not content:
            continue
        score = fuzz.token_set_ratio(query_text.lower(), content.lower()) / 100.0
        candidates.append({
            "source_title": r.get("title") or r.get("url") or "Web result",
            "source_text": content,
            "source_url": r.get("url"),
            "lexical_score": 0.0,
            "semantic_score": round(score, 4),
            "combined_score": round(score, 4),
        })
    return candidates


def hybrid_search(
    query_text: str,
    bm25_index,
    sources: list,
    top_k: int = 5,
    web_budget=None,
):
    """
    Run both lexical and semantic search for a single paragraph, merge
    results by matching source_text, and combine scores.

    If nothing in the local corpus clears SIMILARITY_FLAG_THRESHOLD, and
    web_budget is provided and has remaining budget, falls back to a
    live web search. web_budget should be a WebBudget-like object
    exposing .use() -> bool (True if an attempt was allowed/consumed).

    Returns the single best match (highest combined score), or None if
    no matches anywhere. Structure: {
        "source_title": str, "source_text": str, "source_url": str | None,
        "lexical_score": float, "semantic_score": float, "combined_score": float
    }
    """
    lexical_results = search_lexical(query_text, bm25_index, sources, top_k=top_k)
    semantic_results = search_semantic(query_text, top_k=top_k)

    lex_by_text = {r["source_text"]: r["score"] for r in lexical_results}
    sem_by_text = {r["source_text"]: r["score"] for r in semantic_results}

    all_texts = set(lex_by_text.keys()) | set(sem_by_text.keys())

    merged = []
    title_lookup = {r["source_text"]: r["source_title"] for r in lexical_results + semantic_results}

    for text in all_texts:
        lex_s = lex_by_text.get(text, 0.0)
        sem_s = sem_by_text.get(text, 0.0)

        # Ignore weak semantic noise (under 0.40) when there is zero lexical overlap
        if lex_s == 0.0 and sem_s < 0.40:
            sem_s = 0.0

        c_score = combined_score(lex_s, sem_s)
        if c_score >= settings.SIMILARITY_FLAG_THRESHOLD:
            merged.append({
                "source_title": title_lookup.get(text, "unknown"),
                "source_text": text,
                "source_url": None,
                "lexical_score": lex_s,
                "semantic_score": sem_s,
                "combined_score": c_score,
            })

    if merged:
        merged.sort(key=lambda x: x["combined_score"], reverse=True)
        return merged[0]

    # Nothing in the local corpus cleared the threshold — try the web,
    # but only if the caller has budget left for it.
    if web_budget is not None and web_budget.use():
        web_candidates = _web_fallback_search(query_text, top_k=top_k)
        web_candidates = [
            c for c in web_candidates
            if c["combined_score"] >= settings.SIMILARITY_FLAG_THRESHOLD
        ]
        if web_candidates:
            web_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            return web_candidates[0]

    return None