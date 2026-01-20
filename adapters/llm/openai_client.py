
from __future__ import annotations

import time
import random
from typing import Optional

from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError  # type: ignore

from config.settings import settings

_client: Optional[OpenAI] = None


def get_openai() -> OpenAI:
    global _client
    if _client is None:
        # If your org or timeout is needed, you can extend here:
        # _client = OpenAI(api_key=settings.OPENAI_API_KEY, organization=getattr(settings, "OPENAI_ORG", None), timeout=60)
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _with_backoff(fn, *args, max_retries: int = 4, **kwargs):
    """Simple retry with exponential backoff for transient OpenAI errors."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, InternalServerError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.random() * 0.5
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)


def chat_complete(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Create a chat completion with configurable model/temperature/tokens.

    Args:
        prompt: Fully constructed prompt string (system + user context is fine if you prefer).
        model: Overrides settings.OPENAI_MODEL if provided.
        temperature: Overrides settings.LLM_TEMPERATURE if provided.
        max_tokens: Overrides settings.LLM_MAX_TOKENS if provided.

    Returns:
        The assistant's message content (str). Empty string if not available.
    """
    client = get_openai()

    use_model = model or settings.OPENAI_MODEL
    use_temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    use_max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens

    # You can also move your system prompt here if you want a global policy guard.
    messages = [
        {"role": "system", "content": "Be accurate, concise, actionable, and policy-compliant."},
        {"role": "user", "content": prompt},
    ]

    resp = _with_backoff(
        client.chat.completions.create,
        model=use_model,
        messages=messages,
        temperature=float(use_temp),
        max_tokens=int(use_max_tokens),
    )

    # Defensive extraction
    try:
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        content = ""

    return content
