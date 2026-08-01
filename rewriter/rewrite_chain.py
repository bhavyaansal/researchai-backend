"""
Rewriting chain using Google Gemini API (free tier).
Hard 30s timeout per call. Fails fast instead of hanging forever.
"""
import time
import requests
from config import settings
from rewriter.prompts import SYSTEM_PROMPT, build_rewrite_prompt, build_retry_prompt

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# CALL_TIMEOUT = 30       # hard timeout per request — never hang more than 30s
# DELAY_AFTER_CALL = 5    # wait 5s after each successful call


def _call_groq(prompt: str) -> str:
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512},
    }

    try:
        response = requests.post(
            f"{GROQ_API_URL}?key={api_key}",
            json=payload,
            # timeout=CALL_TIMEOUT,   # hard timeout — raises exception if exceeded
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("groq API timed out after 30s — skipping this span")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"groq API connection error: {e}")

    if response.status_code == 429:
        raise RuntimeError("groq rate limit (15 RPM) — skipping this span")
    if response.status_code == 404:
        raise RuntimeError(f"groq model not found: {response.text[:100]}")
    if response.status_code == 403:
        raise RuntimeError("groq API key invalid or quota exhausted")
    if response.status_code != 200:
        raise RuntimeError(f"groq error {response.status_code}: {response.text[:100]}")

    try:
        result = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        # time.sleep(DELAY_AFTER_CALL)
        return result
    except (KeyError, IndexError):
        raise RuntimeError("Unexpected groq response format")


def rewrite_paragraph(original_text: str, similarity_score: float) -> str:
    return _call_groq(build_rewrite_prompt(original_text, similarity_score))


def rewrite_paragraph_retry(
    original_text: str, previous_attempt: str, previous_score: float
) -> str:
    return _call_groq(build_retry_prompt(
        original_text, previous_attempt, previous_score, settings.TARGET_GLOBAL_THRESHOLD
    ))
