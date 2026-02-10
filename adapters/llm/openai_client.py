from __future__ import annotations

import io
import time
import random
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

        # Keep it consistent with your settings pattern
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _with_backoff(fn, *args, max_retries: int = 4, **kwargs):
    """Simple retry with exponential backoff for transient OpenAI errors."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.random() * 0.5
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)


# ============================================================
# Language Detection (non-blocking) + Krio Normalization
# ============================================================

def detect_language(text: str) -> Tuple[str, float]:
    """
    Best-effort lightweight language detection for analytics and behavior.
    Returns: (lang_label, confidence)
      - 'en', 'krio', 'sw', 'unknown'
    This does NOT limit or block any language.
    """
    if not text:
        return ("unknown", 0.0)

    t = text.lower()

    # Krio markers (heuristic)
    krio_markers = [
        "wetin", "nor", "dey", "sabi", "pikin", "una",
        "tori", "leh", "wey", "sef", "mak", "na", "fo"
    ]
    krio_hits = sum(1 for w in krio_markers if w in t)
    if krio_hits >= 1:
        conf = min(0.55 + 0.10 * krio_hits, 0.90)
        return ("krio", conf)

    # Swahili markers (light heuristic; extend later if needed)
    sw_markers = ["habari", "sawa", "tafadhali", "nisaidie", "asante", "kwanini", "vipi", "je"]
    sw_hits = sum(1 for w in sw_markers if w in t)
    if sw_hits >= 1:
        conf = min(0.55 + 0.10 * sw_hits, 0.90)
        return ("sw", conf)

    # English markers
    en_markers = ["the", "and", "is", "are", "what", "how", "please", "help", "can you"]
    en_hits = sum(1 for w in en_markers if w in t)
    if en_hits >= 1:
        conf = min(0.55 + 0.08 * en_hits, 0.90)
        return ("en", conf)

    return ("unknown", 0.35)


def normalize_krio(text: str) -> str:
    """
    Krio cleanup / normalization BEFORE RAG.
    - Does NOT translate the whole message
    - Normalizes common tokens to improve retrieval/semantic matching
    """
    if not text:
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
    }

    # Keep punctuation mostly intact, normalize word tokens only
    parts = text.split()
    out = []
    for token in parts:
        stripped = token.strip(".,!?;:()[]{}\"'")
        lower = stripped.lower()
        mapped = replacements.get(lower, stripped)

        # Re-apply surrounding punctuation if any
        out.append(token.replace(stripped, mapped, 1))

    return " ".join(out)


def prepare_for_rag(user_text: str) -> Dict[str, Any]:
    """
    Pre-RAG processing:
    - Detect language
    - If Krio, normalize before retrieval
    Returns dict: {original, rag_text, lang, lang_confidence}
    """
    lang, conf = detect_language(user_text)
    rag_text = user_text
    if lang == "krio":
        rag_text = normalize_krio(user_text)

    return {
        "original": user_text,
        "rag_text": rag_text,
        "lang": lang,
        "lang_confidence": conf,
    }


def language_analytics_payload(
    user_id: str,
    *,
    input_type: str,
    raw_text: str,
    lang: Optional[str] = None,
    lang_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns a small payload you can store in Firestore/DB/logs.
    Storage is done by caller (BotFlow), not here.
    """
    if lang is None or lang_confidence is None:
        lang, lang_confidence = detect_language(raw_text)

    return {
        "user_id": user_id,
        "input_type": input_type,  # 'text' | 'audio' | 'image'
        "detected_language": lang,
        "language_confidence": float(lang_confidence),
        "timestamp_unix": int(time.time()),
    }


def _language_ack_line(lang: str) -> str:
    """
    English-only acknowledgement line (do not translate).
    """
    if lang == "krio":
        return "I understand your message was in Krio."
    if lang == "sw":
        return "I understand your message was in Swahili."
    if lang == "en":
        return ""  # no need to say it
    return "I understand your message may be in another language."


# ============================================================
# Chat Completion (ALWAYS English, acknowledge original language)
# ============================================================

def chat_complete(
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    source_text: Optional[str] = None,
    detected_lang: Optional[str] = None,
) -> str:
    """
    Create a chat completion.

    HARD RULES:
    - ALWAYS respond in English only.
    - Do not limit input languages.
    - If input isn't English, acknowledge original language in English.
    """
    client = get_openai()

    use_model = model or getattr(settings, "LLM_MODEL", None) or settings.OPENAI_MODEL
    use_temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    use_max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens

    # Determine language from source_text (preferred) else from prompt
    base_text = source_text if source_text is not None else prompt
    lang = detected_lang or detect_language(base_text)[0]
    ack = _language_ack_line(lang)

    enforced = (
        "You are a helpful assistant.\n"
        "IMPORTANT: Always respond in clear English only.\n"
        "Never respond in any other language, even if the user writes in another language.\n"
        "If the user's message is not English, begin the response with the acknowledgement line provided.\n"
        "Be concise, accurate, and friendly.\n"
    )

    # If caller provided their own system_prompt, append it after our enforcement
    if system_prompt:
        enforced = enforced + "\n" + system_prompt

    # Provide the acknowledgement line as an explicit instruction + content
    # This makes it very hard for the model to “forget” it.
    messages = [
        {"role": "system", "content": enforced},
        {"role": "system", "content": f"Acknowledgement line (use verbatim if non-English): {ack or '[none]'}"},
        {"role": "user", "content": prompt},
    ]

    resp = _with_backoff(
        client.chat.completions.create,
        model=use_model,
        messages=messages,
        temperature=float(use_temp),
        max_tokens=int(use_max_tokens),
    )

    try:
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        content = ""

    # If ack is required but the model omitted it, prepend safely
    if ack and content and not content.lower().startswith(ack.lower()):
        content = f"{ack}\n\n{content}"

    return content


# ============================================================
# Audio Transcription (GPT-4o Transcribe, English+Krio friendly)
# ============================================================

def whisper_transcribe(
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    *,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Transcribe voice notes to text.

    Default model: gpt-4o-transcribe (your “4.o”) [1](https://medtroniclabsorg.sharepoint.com/sites/SPICEDevWorkflowReview/Shared%20Documents/Data%20and%20AI%20Product/Gen%20AI/Empowering%20Physicians/Empowering%20Physicians%20Analysis.pdf?web=1)[2](https://medtroniclabsorg.sharepoint.com/sites/SmartCare/_layouts/15/Doc.aspx?sourcedoc=%7B85C65A62-769D-4FA4-B7F0-9CB4E0D07ED2%7D&file=India-OpenPHC.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)
    - Do NOT force language codes (Krio lacks ISO-639-1)
    - Use a prompt hint to support English+Krio and avoid translation
    """
    if not audio_bytes:
        return ""

    client = get_openai()

    use_model = (
        model
        or getattr(settings, "TRANSCRIBE_MODEL", None)
        or "gpt-4o-transcribe"
    )

    use_prompt = (
        prompt
        or getattr(settings, "TRANSCRIBE_PROMPT", None)
        or "The speaker may use English or Krio. Transcribe faithfully without translating."
    )

    f = io.BytesIO(audio_bytes)
    f.name = filename

    resp = _with_backoff(
        client.audio.transcriptions.create,
        model=use_model,
        file=f,
        prompt=use_prompt,
    )

    try:
        return (resp.text or "").strip()
    except Exception:
        return ""


# ============================================================
# Optional: Image analysis helper (only if you want it here)
# If you already have analyze_image elsewhere, remove this.
# ============================================================

def analyze_image(image_bytes: bytes, mime_type: str, *, model: Optional[str] = None) -> str:
    """
    Analyze an image and return a short English description that can be fed into RAG.

    NOTE: If your project already defines analyze_image elsewhere, keep your existing one.
    This is provided to match app.py import expectations.
    """
    if not image_bytes:
        return ""

    client = get_openai()
    use_model = model or getattr(settings, "VISION_MODEL", None) or "gpt-4o-mini"

    b64 = io.BytesIO(image_bytes).getvalue()
    import base64 as _b64
    data_url = f"data:{mime_type};base64,{_b64.b64encode(b64).decode('utf-8')}"

    # Use chat.completions with image input format
    messages = [
        {
            "role": "system",
            "content": "Describe the image content in English only, clearly and briefly. Do not translate user text; just describe what's visible."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image for support triage."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    resp = _with_backoff(
        client.chat.completions.create,
        model=use_model,
        messages=messages,
        temperature=0.2,
        max_tokens=250,
    )

    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""