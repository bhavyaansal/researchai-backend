import time
import requests
import os
from config import settings
from rewriter.prompts import SYSTEM_PROMPT, build_rewrite_prompt, build_retry_prompt

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DELAY_AFTER_CALL = 3


def _call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API timed out after 30s")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Groq API connection error: {e}")

    if response.status_code == 429:
        raise RuntimeError("Groq rate limit hit — wait a moment and retry")
    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:200]}")

    try:
        result = response.json()["choices"][0]["message"]["content"].strip()
        time.sleep(DELAY_AFTER_CALL)
        return result
    except (KeyError, IndexError):
        raise RuntimeError("Unexpected Groq response format")


def rewrite_paragraph(original_text: str, similarity_score: float) -> str:
    return _call_groq(build_rewrite_prompt(original_text, similarity_score))


def rewrite_paragraph_retry(
    original_text: str, previous_attempt: str, previous_score: float
) -> str:
    return _call_groq(build_retry_prompt(
        original_text, previous_attempt, previous_score, settings.TARGET_GLOBAL_THRESHOLD
    ))