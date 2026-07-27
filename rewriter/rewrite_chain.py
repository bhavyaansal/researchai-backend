"""
Rewriting chain using Google Gemini API (free tier).
Includes automatic retry with delay to handle 15 RPM rate limit.
"""
import time
import requests
from config import settings
from rewriter.prompts import SYSTEM_PROMPT, build_rewrite_prompt, build_retry_prompt

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Free tier = 15 requests/minute. Wait 5 seconds between calls to stay safe.
DELAY_BETWEEN_CALLS = 5  # seconds
MAX_RETRIES = 3


def _call_gemini(prompt: str) -> str:
    """
    Call Gemini with automatic retry on rate limit (429).
    Waits progressively longer on each retry.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not reach Gemini API: {e}")

        if response.status_code == 429:
            # Rate limited — wait and retry
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s
            print(f"Gemini rate limit hit — waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}")
            time.sleep(wait)
            continue

        if response.status_code == 404:
            raise RuntimeError(
                f"Gemini model not found: {response.text[:200]}. "
                "Check the model name in rewrite_chain.py"
            )
        if response.status_code == 403:
            raise RuntimeError(
                "Gemini API key invalid or quota exceeded. "
                "Check your key at aistudio.google.com"
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Polite delay after successful call to avoid hitting rate limit
            time.sleep(DELAY_BETWEEN_CALLS)
            return result
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response: {data}")

    raise RuntimeError(
        "Gemini API rate limit — too many requests. "
        "Wait 1 minute and try again, or reduce MAX_REWRITE_ATTEMPTS in .env"
    )


def rewrite_paragraph(original_text: str, similarity_score: float) -> str:
    prompt = build_rewrite_prompt(original_text, similarity_score)
    return _call_gemini(prompt)


def rewrite_paragraph_retry(
    original_text: str, previous_attempt: str, previous_score: float
) -> str:
    prompt = build_retry_prompt(
        original_text, previous_attempt, previous_score, settings.TARGET_GLOBAL_THRESHOLD
    )
    return _call_gemini(prompt)
