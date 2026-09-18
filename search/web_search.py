"""
web_search.py

Client for the self-hosted SearXNG instance.

Used by the plagiarism detector to search the web for
possible matching content.
"""

import os
import requests

# Read SearXNG URL from environment variable.
# Local fallback is included so the code does not crash
# if the environment variable is missing.
SEARXNG_URL = os.environ.get(
    "SEARXNG_URL",
    "https://researchai-searxng-1.onrender.com"
).rstrip("/")

REQUEST_TIMEOUT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}

class SearXNGNotConfiguredError(Exception):
    pass

def _make_search_query(text: str) -> str:
    """
    Convert a paragraph into a shorter web-search query.

    We use only the first 15 words because sending an entire
    paragraph to a search engine is unnecessary and can cause
    problems with some search engines.
    """

    words = text.strip().split()

    # Remove very short words such as "a", "I", etc.
    words = [word for word in words if len(word) > 1]

    query = " ".join(words[:15])

    return query

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the configured SearXNG instance.

    Returns a list like:

    [
        {
            "title": "...",
            "url": "...",
            "content": "..."
        }
    ]
    """

    # Get the URL again from environment variables.
    # This is useful on Render because environment variables
    # are provided by the deployment environment.
    searxng_url = os.environ.get(
        "SEARXNG_URL",
        SEARXNG_URL
    ).rstrip("/")

    if not searxng_url:
        raise SearXNGNotConfiguredError(
            "SEARXNG_URL is not configured."
        )

    # Create a shorter search query.
    search_query = _make_search_query(query)

    if not search_query:
        return []

    # ---------------------------------------------------------
    # DEBUG LOGS
    # ---------------------------------------------------------
    # These will help us determine exactly what the FastAPI
    # server is sending to SearXNG.
    print(
        f"[web_search] SEARXNG_URL = {searxng_url}"
    )

    print(
        f"[web_search] QUERY = {search_query}"
    )

    # ---------------------------------------------------------
    # SEND REQUEST TO SEARXNG
    # ---------------------------------------------------------
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

        print(
            f"[web_search] HTTP STATUS = {response.status_code}"
        )

        # Raise an exception for 4xx/5xx responses.
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(
            f"[web_search] HTTP error: {e} "
            f"| query={search_query!r}"
        )

        return []

    # ---------------------------------------------------------
    # PARSE JSON
    # ---------------------------------------------------------
    try:
        data = response.json()
    except ValueError as e:
        print(
            f"[web_search] Invalid JSON response: {e}"
        )
        return []

    # ---------------------------------------------------------
    # EXTRACT RESULTS
    # ---------------------------------------------------------
    results = []

    for item in data.get("results", [])[:max_results]:

        title = (item.get("title") or "").strip()

        url = (item.get("url") or "").strip()

        content = (item.get("content") or "").strip()

        # We only keep results that have some text.
        if not content:
            continue

        results.append({
            "title": title,
            "url": url,
            "content": content,
        })

    print(
        f"[web_search] RESULTS = {len(results)}"
    )

    return results