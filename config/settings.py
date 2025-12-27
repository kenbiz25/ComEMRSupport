
# config/settings.py
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

def _to_bool(val, default=True):
    if val is None:
        return default
    return str(val).strip().lower() in ("true", "1", "yes", "y")

def _to_float(val, default=0.55):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

class Settings:
    # WhatsApp (Meta Cloud API)
    META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
    META_WHATSAPP_TOKEN = os.getenv("META_WHATSAPP_TOKEN", "")
    WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
    WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v22.0")  # ✅ Added

    # KB namespace and directory
    KB_NAMESPACE = os.getenv("KB_NAMESPACE", "default")
    KB_DIR = os.getenv("KB_DIR", "KB")  # ✅ Added for indexing pipeline and fallback

    # LLM provider & model
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
    # Alias for convenience (falls back to OPENAI_MODEL)
    LLM_MODEL = os.getenv("LLM_MODEL", OPENAI_MODEL)

    # Embeddings (either OpenAI model name OR HF model id)
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # (If you keep Postgres later, keep these for compatibility)
    PG_DSN = os.getenv("PG_DSN", "")
    VECTOR_TABLE = os.getenv("VECTOR_TABLE", "kb_chunks")

    # Policy knobs (defensive parsing)
    ANSWER_CONFIDENCE_THRESHOLD = _to_float(os.getenv("ANSWER_CONFIDENCE_THRESHOLD"), 0.55)
    SAFEGUARD_ENABLE = _to_bool(os.getenv("SAFEGUARD_ENABLE"), True)

settings = Settings()
