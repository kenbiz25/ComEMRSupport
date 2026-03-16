from __future__ import annotations

import base64
import io
import time
import random
import re
from collections import Counter
from typing import Optional, Dict, Any, Tuple

from config.settings import settings

_client: Optional["OpenAI"] = None


# ============================================================
# Client
# ============================================================

def get_openai() -> "OpenAI":
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package is not installed") from e

        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _with_backoff(
    fn,
    *args,
    max_retries: int = 5,
    max_delay: float = 30.0,
    **kwargs
):
    """Retry with exponential backoff + jitter for transient errors only."""
    try:
        from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
        _retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    except ImportError:
        _retryable = (Exception,)

    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except _retryable:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.random() * 0.5
            time.sleep(sleep_for)
            delay = min(delay * 2, max_delay)


# ============================================================
# Language Detection + Krio Normalization (for RAG + analytics)
# ============================================================

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def detect_language(text: str) -> Tuple[str, float]:
    """
    Lightweight language detection using scoring + negative signals.
    Returns: (lang_label, confidence) — 'en', 'krio', 'sw', 'unknown'
    NOTE: Detection is used for analytics + pre-RAG normalization ONLY.
          It MUST NOT change the bot's output language (English-only).
    """
    if not text or len(text.strip()) < 6:
        return "unknown", 0.0

    t = text.lower()
    words = _WORD_RE.findall(t)
    if len(words) < 2:
        return "unknown", 0.35

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    # Krio positive signals (weighted words + some bigrams)
    # Keep short tokens ('na', 'fo') but ONLY match via word/bigram lists to avoid substring false positives.
    krio_pos = Counter({
        "wetin": 12, "nor": 8, "dey": 10, "sabi": 9, "pikin": 7, "una": 6,
        "tori": 5, "na": 8, "fo": 6, "leh": 5, "wey": 7, "sef": 5, "mek": 6,
        "na so": 4, "wetin be": 5, "for na": 3, "how far": 4,
    })

    # English strong function words (negative signal for Krio)
    en_strong = {
        "the": 15, "and": 12, "is": 10, "are": 9, "that": 8, "with": 7,
        "they": 6, "this": 6, "have": 5, "for": 5, "you": 5,
    }

    # Swahili (simple)
    sw_pos = Counter({
        "habari": 10, "asante": 9, "sawa": 8, "tafadhali": 7, "nisaidie": 6,
        "vipi": 6, "kwanini": 5, "je": 4,
    })

    def score(counter: Counter) -> float:
        s = sum(counter.get(w, 0) for w in words)
        s += sum(counter.get(bg, 0) for bg in bigrams)
        return s / max(1.0, (len(words) ** 0.6))  # mild length normalization

    krio_score = score(krio_pos)
    sw_score = score(sw_pos)
    en_penalty = sum(en_strong.get(w, 0) for w in words) / max(1, len(words))

    # Decision thresholds
    if krio_score > 3.5 and krio_score > sw_score * 1.6 and krio_score > en_penalty * 0.9:
        conf = min(0.70 + 0.09 * krio_score, 0.96)
        return "krio", conf

    if sw_score > 4.0 and sw_score > krio_score * 1.6:
        conf = min(0.70 + 0.11 * sw_score, 0.95)
        return "sw", conf

    if len(words) > 4 and en_penalty > 2.5 and krio_score < 3.8:
        conf = min(0.68 + 0.07 * en_penalty, 0.94)
        return "en", conf

    # Fallback: word-boundary hits only (avoid substring matching)
    krio_hits = 0
    for k in krio_pos.keys():
        if " " in k:
            if k in bigrams:
                krio_hits += 1
        else:
            if k in words:
                krio_hits += 1

    if krio_hits >= 2:
        return "krio", min(0.62 + 0.08 * krio_hits, 0.90)

    return "unknown", 0.40


def normalize_krio(text: str, aggressive: bool = True) -> str:
    """
    Normalize common Krio tokens to improve retrieval/semantic matching.
    Aggressive mode applies more replacements; non-aggressive keeps original.
    """
    if not text:
        return text

    if not aggressive:
        return text

    replacements = {
        "wetin": "what",
        "nor": "not",
        "dey": "is",
        "sabi": "know",
        "pikin": "child",
        "una": "you all",
        "tori": "story",
        "wey": "that",
        "sef": "also",
        "mek": "make",
        "leh": "let",
    }

    parts = text.split()
    out = []
    for token in parts:
        stripped = token.strip(".,!?;:()[]{}\"'")
        lower = stripped.lower()
        mapped = replacements.get(lower, stripped)
        out.append(token.replace(stripped, mapped, 1) if stripped else token)

    return " ".join(out)


def prepare_for_rag(user_text: str) -> Dict[str, Any]:
    """
    Pre-RAG processing pipeline:
    - Detect language (analytics + behavior)
    - If Krio, normalize before retrieval (configurable)
    """
    lang, conf = detect_language(user_text)
    rag_text = user_text

    aggressive_norm = getattr(settings, "KRIO_NORMALIZE_AGGRESSIVE", True)
    if lang == "krio":
        rag_text = normalize_krio(user_text, aggressive=aggressive_norm)

    return {
        "original": user_text,
        "rag_text": rag_text,
        "lang": lang,
        "lang_confidence": conf,
        "estimated_tokens": len(user_text) // 4 + 1,
    }


def language_analytics_payload(
    user_id: str,
    *,
    input_type: str,
    raw_text: str,
    lang: Optional[str] = None,
    lang_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    if lang is None or lang_confidence is None:
        lang, lang_confidence = detect_language(raw_text)

    return {
        "user_id": user_id,
        "input_type": input_type,  # 'text' | 'audio' | 'image'
        "detected_language": lang,
        "language_confidence": float(lang_confidence),
        "timestamp_unix": int(time.time()),
        "text_length": len(raw_text or ""),
    }


def _language_ack_line(lang: str) -> str:
    # Product requirement: NO language acknowledgement line in user-facing responses
    return ""


# ============================================================
# Chat Completion (English-only output, backward compatible)
# ============================================================

def chat_complete(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Backward-compatible: returns ONLY the assistant content string.
    If you want status/error info, use chat_complete_safe().
    """
    content, ok, _err = chat_complete_safe(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    return content if ok else ""


def chat_complete_safe(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    Returns: (content, success, error_message)
    """
    client = get_openai()

    use_model = (
        model
        or getattr(settings, "LLM_MODEL", None)
        or getattr(settings, "OPENAI_MODEL", None)
        or "gpt-4o-mini"
    )

    use_temp = getattr(settings, "LLM_TEMPERATURE", 0.3) if temperature is None else temperature
    use_max_tokens = getattr(settings, "LLM_MAX_TOKENS", 350) if max_tokens is None else max_tokens

    enforced = (
        "You are a helpful, concise, and friendly assistant.\n"
        "IMPORTANT: Always respond in clear, natural English only.\n"
        "Never respond in any other language, even if the user writes in another language.\n"
        "Do NOT add any language detection or language switching messages.\n"
    )

    if system_prompt:
        enforced += "\n" + system_prompt

    messages = [
        {"role": "system", "content": enforced},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = _with_backoff(
            client.chat.completions.create,
            model=use_model,
            messages=messages,
            temperature=float(use_temp),
            max_tokens=int(use_max_tokens),
        )
        content = (resp.choices[0].message.content or "").strip()
        return content, True, ""
    except Exception as e:
        msg = f"Chat completion failed ({use_model}): {str(e)}"
        return "", False, msg


# ============================================================
# Audio Transcription (backward compatible)
# ============================================================

def whisper_transcribe(
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    *,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Backward-compatible: returns ONLY the transcribed text string.
    If you want status/error info, use whisper_transcribe_safe().
    """
    text, ok, _err = whisper_transcribe_safe(
        audio_bytes,
        filename=filename,
        model=model,
        prompt=prompt,
    )
    return text if ok else ""


def whisper_transcribe_safe(
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    *,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    Returns: (transcribed_text, success, error_message)
    """
    if not audio_bytes:
        return "", False, "Empty audio input"

    client = get_openai()

    use_model = (
        model
        or getattr(settings, "TRANSCRIBE_MODEL", None)
        or "gpt-4o-transcribe"
    )

    # Prompt hardened to reduce hallucinations during long pauses/silence
    use_prompt = (
        prompt
        or getattr(settings, "TRANSCRIBE_PROMPT", None)
        or (
            "The speaker may use English, Krio, or a mix of both. "
            "Transcribe exactly what is spoken. "
            "If there are long pauses or silence, do not generate filler or repetitive text. "
            "Do not translate. Do not correct grammar."
        )
    )

    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename

        resp = _with_backoff(
            client.audio.transcriptions.create,
            model=use_model,
            file=f,
            prompt=use_prompt,
            response_format="text",
        )

        # Some SDKs return an object with .text, others return plain text when response_format="text"
        text = (getattr(resp, "text", resp) or "").strip()
        return text, True, ""
    except Exception as e:
        msg = f"Transcription failed ({use_model}): {str(e)}"
        return "", False, msg


# ============================================================
# Image Analysis (backward compatible)
# ============================================================

def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    *,
    model: Optional[str] = None,
) -> str:
    """
    Backward-compatible: returns ONLY the description string.
    If you want status/error info, use analyze_image_safe().
    """
    desc, ok, _err = analyze_image_safe(image_bytes, mime_type, model=model)
    return desc if ok else ""


def analyze_image_safe(
    image_bytes: bytes,
    mime_type: str,
    *,
    model: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    Returns: (description, success, error_message)
    """
    if not image_bytes:
        return "", False, "Empty image input"

    client = get_openai()

    use_model = (
        model
        or getattr(settings, "VISION_MODEL", None)
        or "gpt-4o-mini"
    )

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an image analyzer. Describe the visible content in clear English only. "
                "Be concise and factual. Do NOT translate any text in the image; describe what is written if relevant. "
                "Focus on elements useful for support triage or understanding context."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly for support context."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    try:
        resp = _with_backoff(
            client.chat.completions.create,
            model=use_model,
            messages=messages,
            temperature=0.25,
            max_tokens=300,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content, True, ""
    except Exception as e:
        msg = f"Image analysis failed ({use_model}): {str(e)}"
        return "", False, msg