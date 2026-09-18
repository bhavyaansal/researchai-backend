"""
Web search client for the self-hosted SearXNG instance.

SearXNG searches Bing and returns search-result URLs.
"""

import os
import requests


SEARXNG_URL = os.environ.get(
    "SEARXNG_URL",
    "https://researchai-searxng-1.onrender.com"
).rstrip("/")

REQUEST_TIMEOUT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}


class SearXNGNotConfiguredError(Exception):
    pass


def _make_search_query(text: str) -> str:
    """
    Convert a paragraph into a short search query.

    We intentionally use only the first ~15 words.
    Sending an entire paragraph to SearXNG/Bing can cause
    slow requests and 502 errors.
    """

    words = text.strip().split()

    # Remove extremely short fragments.
    words = [word for word in words if len(word) > 1]

    # Keep the search request small.
    query = " ".join(words[:15])

    return query


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search SearXNG and return search result metadata.

    Returns:
        [
            {
                "title": "...",
                "url": "...",
                "content": "..."
            }
        ]
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

    # Convert long paragraph into a small search query.
    search_query = _make_search_query(query)

    if not search_query:
        return []

    try:
        response = requests.get(
            f"{searxng_url}/search",
            params={
                "q": search_query,
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )

        response.raise_for_status()

        data = response.json()

    except requests.HTTPError as e:
        print(
            f"[web_search] HTTP error: "
            f"{e.response.status_code} | "
            f"query={search_query[:100]!r}"
        )
        return []

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

        if not url:
            continue

        results.append({
            "title": title,
            "url": url,
            "content": content,
        })

    print(
        f"[web_search] query={search_query!r} "
        f"results={len(results)}"
    )

    return results