"""
web_search.py

Client for Tavily Search API.

Used by the plagiarism detector to search the web for
possible matching content.
"""

import os
import requests

# Read Tavily API key from environment variable
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

TAVILY_URL = "https://api.tavily.com/search"

REQUEST_TIMEOUT = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}

class TavilyNotConfiguredError(Exception):
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
    Search using Tavily Search API.

    Returns a list like:

    [
        {
            "title": "...",
            "url": "...",
            "content": "..."
        }
    ]
    """

    # Get API key from environment
    api_key = os.environ.get("TAVILY_API_KEY", TAVILY_API_KEY)

    if not api_key:
        raise TavilyNotConfiguredError(
            "TAVILY_API_KEY is not configured. "
            "Add it to your Render environment variables."
        )

    # Create a shorter search query
    search_query = _make_search_query(query)

    if not search_query:
        return []

    # ---------------------------------------------------------
    # DEBUG LOGS
    # ---------------------------------------------------------
    print(f"[web_search] TAVILY_URL = {TAVILY_URL}")
    print(f"[web_search] QUERY = {search_query}")

    # ---------------------------------------------------------
    # SEND REQUEST TO TAVILY
    # ---------------------------------------------------------
    try:
        response = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": search_query,
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )

        print(f"[web_search] HTTP STATUS = {response.status_code}")

        # Raise an exception for 4xx/5xx responses
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
        print(f"[web_search] Invalid JSON response: {e}")
        return []

    # ---------------------------------------------------------
    # EXTRACT RESULTS
    # ---------------------------------------------------------
    results = []

    for item in data.get("results", [])[:max_results]:

        title = (item.get("title") or "").strip()

        url = (item.get("url") or "").strip()

        # Tavily returns 'content' directly, unlike SearXNG
        content = (item.get("content") or "").strip()

        # We only keep results that have some text
        if not content:
            continue

        results.append({
            "title": title,
            "url": url,
            "content": content,
        })

    print(f"[web_search] RESULTS = {len(results)}")

    return results