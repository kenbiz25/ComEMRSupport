import os
from typing import Optional

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        # Lazy-import whisper to avoid heavy import during startup and allow environments without whisper
        import whisper
        model_name = os.getenv("WHISPER_MODEL", "base")
        _model = whisper.load_model(model_name)
        return _model
    except Exception:
        _model = None
        return None


def transcribe_audio(audio_path: str) -> tuple[str, float, str]:
    """Transcribe an audio file using available backends.

    Tries in order: local whisper model (if installed), OpenAI transcription (if configured), speech_recognition Google fallback.
    Returns a tuple: (text, confidence, language).
    """
    # 1) Try local whisper model
    try:
        model = _load_model()
        if model is not None:
            result = model.transcribe(audio_path)
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.75)
            lang = result.get("language", "en")
            return text, confidence, lang
    except Exception:
        # fall through to other backends
        pass

    # 2) Try OpenAI whisper via the configured OpenAI client
    try:
        from adapters.llm.openai_client import get_openai

        client = get_openai()
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(file=f, model="whisper-1")
            # defensive extraction
            try:
                text = getattr(resp, "text", None) or resp.get("text") or ""
            except Exception:
                text = ""
            return text.strip(), 0.75, "en"
    except Exception:
        pass

    # 3) Try speech_recognition as a last resort
    try:
        import speech_recognition as sr

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text, 0.75, "en"
    except Exception:
        pass

    # Final fallback
    return "[Audio received; transcription unavailable]", 0.0, "unknown"
