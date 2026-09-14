"""
web_search.py

Thin client for a self-hosted SearXNG instance, used to check document
sentences/paragraphs against live web content instead of (or alongside)
the local SourceDocument corpus.

Requires the SEARXNG_URL environment variable, e.g.:
    SEARXNG_URL=https://researchai-searxng-production.up.railway.app
"""
import os
import requests

SEARXNG_URL = os.environ.get("SEARXNG_URL", "").rstrip("/")
REQUEST_TIMEOUT = 6  # seconds — keep short so one slow query can't stall a whole scan


class SearXNGNotConfiguredError(Exception):
    pass


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Query the self-hosted SearXNG instance and return a list of
    {"title": ..., "url": ..., "content": ...} dicts (content is the
    search engine's snippet, not the full page).

    Returns [] on any failure (timeout, bad response, etc.) rather than
    raising, so a flaky search never crashes a scan — callers should
    treat an empty list as "no web match found for this query".
    """
    searxng_url = os.environ.get("SEARXNG_URL", SEARXNG_URL).rstrip("/")
    if not searxng_url:
        raise SearXNGNotConfiguredError(
            "SEARXNG_URL environment variable is not set."
        )

    query = (query or "").strip()
    if not query:
        return []

    # SearXNG works best with short, focused queries — long paragraphs
    # tend to return nothing useful. Cap length defensively.
    if len(query) > 300:
        query = query[:300]

    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        content = (item.get("content") or "").strip()
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if content:
            results.append({"title": title, "url": url, "content": content})
    return results