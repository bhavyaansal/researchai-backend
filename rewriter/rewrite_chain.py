"""
Rewriting chain using Google Gemini API (free tier).
Includes automatic retry with delay to handle 15 RPM rate limit.
"""
import time
import requests
from config import settings
from rewriter.prompts import SYSTEM_PROMPT, build_rewrite_prompt, build_retry_prompt

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

DELAY_BETWEEN_CALLS = 5  # seconds
MAX_RETRIES = 3


def _call_gemini(prompt: str) -> str:
    """
    Call Gemini with automatic fallback across valid models and retry on rate limit (429).
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

    last_error = None

    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, json=payload, timeout=45)
            except requests.exceptions.RequestException as e:
                last_error = f"Could not reach Gemini API: {e}"
                break

            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"Gemini ({model_name}) rate limit hit — waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait)
                continue

            if response.status_code == 404:
                last_error = f"Model {model_name} not found"
                break

            if response.status_code == 403:
                raise RuntimeError("Gemini API key invalid or quota exceeded.")

            if response.status_code != 200:
                last_error = f"Gemini API error {response.status_code}: {response.text[:200]}"
                break

            data = response.json()
            try:
                result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                time.sleep(DELAY_BETWEEN_CALLS)
                return result
            except (KeyError, IndexError):
                last_error = f"Unexpected Gemini response: {data}"
                break

    raise RuntimeError(last_error or "Gemini API call failed across models")


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
