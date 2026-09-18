"""
Web search client for the self-hosted SearXNG instance.

SearXNG searches Bing and returns search-result URLs.
The URLs will later be fetched so we can compare against the
actual webpage text instead of only the search-engine snippet.
"""

import os
import requests


SEARXNG_URL = os.environ.get(
    "SEARXNG_URL",
    "https://researchai-searxng-1.onrender.com"
).rstrip("/")

REQUEST_TIMEOUT = 8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}


class SearXNGNotConfiguredError(Exception):
    pass


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the self-hosted SearXNG instance.

    Returns:
        [
            {
                "title": "...",
                "url": "...",
                "content": "..."
            }
        ]

    The content field is only the search-engine snippet.
    The URL will be used later to fetch the actual webpage.
    """

    searxng_url = os.environ.get(
        "SEARXNG_URL",
        SEARXNG_URL
    ).rstrip("/")

    if not searxng_url:
        raise SearXNGNotConfiguredError(
            "SEARXNG_URL environment variable is not set."
        )

    query = (query or "").strip()

    if not query:
        return []

    # Limit query size so very large paragraphs don't create
    # unnecessarily large search requests.
    if len(query) > 300:
        query = query[:300]

    try:
        response = requests.get(
            f"{searxng_url}/search",
            params={
                "q": query,
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        print(
            f"[web_search] Request failed: "
            f"{type(e).__name__}: {e}"
        )
        return []

    except ValueError as e:
        print(
            f"[web_search] Invalid JSON response: {e}"
        )
        return []

    results = []

    for item in data.get("results", [])[:max_results]:

        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()

        # A web result without a URL isn't useful to us.
        if not url:
            continue

        results.append({
            "title": title,
            "url": url,
            "content": content,
        })

    print(
        f"[web_search] query={query[:80]!r} "
        f"results={len(results)}"
    )

    return results