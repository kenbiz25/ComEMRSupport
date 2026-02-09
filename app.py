# app.py

# --- Fix for Windows / uvicorn module import issues ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

# --- Standard imports ---
import os
import json
import shutil
import tempfile
import logging
import base64
from datetime import datetime
from typing import Optional, Tuple

import httpx

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    File,
    Request,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, PlainTextResponse

# --- Core services ---
from core.knowledge.store_faiss import FaissStore
from core.llm.llm_service import LLMService
from core.whatsapp.whatsapp_service import WhatsAppService
from rag.flow import BotFlow

# Optional WhatsApp adapter legacy support
try:
    from whatsapp import send_whatsapp_text, send_whatsapp_template
    HAS_WHATSAPP_ADAPTER = True
except Exception:
    HAS_WHATSAPP_ADAPTER = False

# Settings
from config.settings import settings

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComEMR")

# --- Initialize FastAPI ---
app = FastAPI(title="ComEMR Support", version="1.3.0")

# --- CORS ---
if getattr(settings, "ALLOWED_ORIGINS", None):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_headers=["*"],
        allow_methods=["*"],
    )

# ------------------------------------------------------------------
# Shared Components
# ------------------------------------------------------------------
def _get_embedding_dim():
    em = (settings.EMBED_MODEL or "").lower()
    if "text-embedding-3-large" in em:
        return 3072
    return 1536

EMBEDDING_DIM = _get_embedding_dim()

faiss_store = FaissStore(dim=EMBEDDING_DIM)
llm_service = LLMService(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
)

whatsapp_service = WhatsAppService(
    phone_id=os.getenv("WHATSAPP_PHONE_ID", ""),
    token=os.getenv("META_WHATSAPP_TOKEN", ""),
)

bot = BotFlow(faiss_store, llm_service, whatsapp_service)
app.state.bot = bot

# ------------------------------------------------------------------
# Language behavior: understand Krio / any language, but reply in English
# ------------------------------------------------------------------
_ENGLISH_REPLY_HINT = "\n\n[Instruction: Reply in English.]\n"

_KRIO_MARKERS = {
    "wetin", "una", "mek", "nor", "dey", "na", "pikin", "boku", "tin",
    "dem", "leh", "abeg", "sef", "kushe", "kusheh", "how di bodi",
    "how di body", "a wan", "a go", "a de", "e don"
}

def _looks_like_krio_or_non_english(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if any(m in t for m in _KRIO_MARKERS):
        return True
    non_ascii = sum(1 for ch in t if ord(ch) > 127)
    return non_ascii >= 3

def _ensure_english_reply_hint(user_text: str) -> str:
    if not user_text:
        return user_text
    # enforce English replies consistently (Krio or any language)
    return user_text + _ENGLISH_REPLY_HINT

# ------------------------------------------------------------------
# WhatsApp Media Helpers (Graph download by media_id)
# ------------------------------------------------------------------
async def _download_whatsapp_media(media_id: str) -> Tuple[bytes, str]:
    """
    Cloud API media retrieval pattern:
    1) GET /{media_id} -> returns {url, mime_type}
    2) GET url -> returns bytes (requires auth header)
    """
    token = os.getenv("META_WHATSAPP_TOKEN", "")
    if not token:
        raise RuntimeError("META_WHATSAPP_TOKEN missing")

    api_ver = os.getenv("WHATSAPP_API_VERSION", "v22.0")
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=60) as client:
        meta = await client.get(
            f"https://graph.facebook.com/{api_ver}/{media_id}",
            headers=headers,
        )
        meta.raise_for_status()
        mjson = meta.json()
        media_url = mjson.get("url")
        mime_type = mjson.get("mime_type") or "application/octet-stream"
        if not media_url:
            raise RuntimeError("Media URL missing from Graph response")

        blob = await client.get(media_url, headers=headers)
        blob.raise_for_status()
        return blob.content, mime_type

def _guess_ext_from_mime(mime: str, default: str) -> str:
    m = (mime or "").lower()
    if "audio/ogg" in m or "opus" in m:
        return ".ogg"
    if "audio/mpeg" in m or "audio/mp3" in m:
        return ".mp3"
    if "audio/wav" in m:
        return ".wav"
    if "image/png" in m:
        return ".png"
    if "image/webp" in m:
        return ".webp"
    if "image/jpeg" in m or "image/jpg" in m:
        return ".jpg"
    return default

def _safe_send_text(to: str, text: str):
    try:
        if hasattr(whatsapp_service, "send_text"):
            whatsapp_service.send_text(to, text)
        elif hasattr(whatsapp_service, "send_message"):
            whatsapp_service.send_message(to, text)
        else:
            if HAS_WHATSAPP_ADAPTER:
                send_whatsapp_text(to, text)
    except Exception:
        logger.exception("Failed to send WhatsApp text")

# ------------------------------------------------------------------
# OpenAI Helpers: Whisper transcription + Vision screenshot analysis
# ------------------------------------------------------------------
async def _transcribe_audio_bytes(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """
    OpenAI Whisper transcription endpoint.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    whisper_model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    files = {"file": (filename, audio_bytes)}
    data = {"model": whisper_model}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files=files,
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()

def _downscale_image_if_possible(image_bytes: bytes, mime_type: str) -> Tuple[bytes, str]:
    """
    Optional: downscale large screenshots to speed up vision and reduce cost.
    Uses Pillow if available. If Pillow missing or fails, returns original bytes.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        max_w = int(os.getenv("VISION_MAX_WIDTH", "1280"))
        if img.width > max_w:
            ratio = max_w / float(img.width)
            new_h = int(img.height * ratio)
            img = img.resize((max_w, new_h))

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=int(os.getenv("VISION_JPEG_QUALITY", "85")))
        return out.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime_type or "image/jpeg"

async def _analyze_image_bytes(image_bytes: bytes, mime_type: str) -> str:
    """
    Vision analysis for screenshots:
    - Extract visible error text (OCR-like)
    - Identify screen context
    - Suggest next steps
    Always in English.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    # Optional downscale/compress
    image_bytes, mime_type = _downscale_image_if_possible(image_bytes, mime_type)

    # data url
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    prompt = (
        "You are a tech support assistant for ComEMR/SPICE. Analyze the screenshot.\n"
        "Return:\n"
        "1) Visible error messages (verbatim if possible)\n"
        "2) What screen/page this appears to be\n"
        "3) What the user is likely trying to do\n"
        "4) The best fix / next steps (short, actionable)\n"
        "Reply in English."
    )

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

# ------------------------------------------------------------------
# Startup diagnostics (and KB folders for smoother ops)
# ------------------------------------------------------------------
@app.on_event("startup")
def startup_diag():
    logger.info(
        "WhatsApp config: token=%s phone_id=%s",
        bool(os.getenv("META_WHATSAPP_TOKEN")),
        bool(os.getenv("WHATSAPP_PHONE_ID")),
    )

    # Soft ops: ensure logs folder exists
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------------
# WhatsApp Webhook Verification (GET)
# ------------------------------------------------------------------
@app.get("/whatsapp/webhook", tags=["WhatsApp Gateway"])
async def whatsapp_webhook_verify(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.META_VERIFY_TOKEN:
        return Response(content=challenge or "", media_type="text/plain")

    return Response(status_code=403)

# ------------------------------------------------------------------
# Utilities for webhook diagnostics
# ------------------------------------------------------------------
_LOG_UNMATCHED = Path("logs/unmatched_webhooks.jsonl")

def _redact_payload(obj: dict) -> dict:
    SENSITIVE = (
        "token", "access", "auth", "password",
        "api_key", "authorization", "secret"
    )

    def _walk(o):
        if isinstance(o, dict):
            return {
                k: "[REDACTED]" if any(s in str(k).lower() for s in SENSITIVE) else _walk(v)
                for k, v in o.items()
            }
        if isinstance(o, list):
            return [_walk(i) for i in o]
        if isinstance(o, str) and len(o) > 20:
            return "[REDACTED]"
        return o

    return _walk(obj or {})

def _dump_unmatched_payload(payload):
    try:
        _LOG_UNMATCHED.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_UNMATCHED, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.utcnow().isoformat() + "Z",
                "payload": _redact_payload(payload),
            }) + "\n")
    except Exception:
        logger.exception("Failed to dump unmatched payload")

# ------------------------------------------------------------------
# WhatsApp Webhook (POST) — CRASH‑SAFE + TEXT + AUDIO + IMAGES + ENGLISH REPLY
# ------------------------------------------------------------------
@app.post("/whatsapp/webhook", tags=["WhatsApp Gateway"])
async def whatsapp_webhook(
    request: Request,
    from_: str = Query(None, alias="from"),
    body: str = Query(None, alias="body"),
    payload: dict = Body(None),
):
    try:
        j = payload
        if j is None:
            try:
                j = await request.json()
            except Exception:
                j = None

        # ACK non-message payloads (statuses, receipts, etc.)
        try:
            if isinstance(j, dict) and j.get("entry"):
                value = j["entry"][0]["changes"][0]["value"]
                if value.get("statuses") and not value.get("messages"):
                    return JSONResponse(content={"status": "acknowledged"})
        except Exception:
            pass

        # Extract message object
        msg = None
        if isinstance(j, dict):
            if j.get("entry"):
                value = j["entry"][0]["changes"][0]["value"]
                if value.get("messages"):
                    msg = value["messages"][0]
            elif j.get("messages"):
                msg = j["messages"][0]

        if not msg:
            _dump_unmatched_payload(j)
            return JSONResponse(content={"status": "acknowledged"})

        msg_type = msg.get("type")
        from_ = from_ or msg.get("from")

        logger.info("Incoming WhatsApp message: type=%s from=%s", msg_type, from_)

        # ---- TEXT / INTERACTIVE ----
        text_body = (msg.get("text") or {}).get("body")
        interactive = msg.get("interactive") or {}
        button_text = (msg.get("button") or {}).get("text")
        body = body or text_body or \
            (interactive.get("button_reply") or {}).get("title") or \
            (interactive.get("list_reply") or {}).get("title") or \
            button_text

        # ---- AUDIO (voice notes) ----
        if (not body) and (msg_type == "audio"):
            media_id = (msg.get("audio") or {}).get("id")
            if media_id:
                try:
                    audio_bytes, mime = await _download_whatsapp_media(media_id)
                    ext = _guess_ext_from_mime(mime, ".ogg")
                    transcript = await _transcribe_audio_bytes(audio_bytes, filename=f"voice{ext}")
                    if transcript:
                        body = transcript
                    else:
                        _safe_send_text(from_, "I received your voice note but couldn’t transcribe it. Please type your question.")
                        return JSONResponse(content={"status": "acknowledged"})
                except Exception:
                    logger.exception("Audio transcription failed")
                    _safe_send_text(from_, "I received your voice note but couldn’t transcribe it. Please type your question.")
                    return JSONResponse(content={"status": "acknowledged"})

        # ---- IMAGES / SCREENSHOTS ----
        # WhatsApp sends screenshots as type=image (sometimes as type=document with image mime)
        if (not body) and (msg_type == "image"):
            media_id = (msg.get("image") or {}).get("id")
            caption = (msg.get("image") or {}).get("caption")
            if media_id:
                try:
                    img_bytes, mime = await _download_whatsapp_media(media_id)
                    insight = await _analyze_image_bytes(img_bytes, mime)
                    combined = ""
                    if caption:
                        combined += f"User caption: {caption}\n"
                    combined += f"Screenshot analysis: {insight}"
                    body = combined
                except Exception:
                    logger.exception("Image analysis failed")
                    _safe_send_text(from_, "I received the screenshot, but I couldn’t read it clearly. Please type the error message or resend a clearer screenshot.")
                    return JSONResponse(content={"status": "acknowledged"})

        # Document uploads: if image is sent as a document attachment (common for screenshots)
        if (not body) and (msg_type == "document"):
            doc = msg.get("document") or {}
            media_id = doc.get("id")
            mime = (doc.get("mime_type") or "").lower()
            caption = doc.get("caption") or doc.get("filename")
            if media_id and mime.startswith("image/"):
                try:
                    img_bytes, real_mime = await _download_whatsapp_media(media_id)
                    insight = await _analyze_image_bytes(img_bytes, real_mime or mime)
                    combined = ""
                    if caption:
                        combined += f"User caption/filename: {caption}\n"
                    combined += f"Screenshot analysis: {insight}"
                    body = combined
                except Exception:
                    logger.exception("Document-image analysis failed")
                    _safe_send_text(from_, "I received the screenshot, but I couldn’t read it clearly. Please type the error message or resend a clearer screenshot.")
                    return JSONResponse(content={"status": "acknowledged"})

        # If still nothing usable, ACK + log payload
        if not from_ or not body:
            _dump_unmatched_payload(j)
            return JSONResponse(content={"status": "acknowledged"})

        # Enforce English reply behavior (Krio/audio/image -> English reply)
        body_for_bot = _ensure_english_reply_hint(body)

        handler_bot = getattr(request.app.state, "bot", None) or bot
        try:
            handler_bot.handle_message(from_, body_for_bot, session_id=from_)
        except TypeError:
            handler_bot.handle_message(from_, body_for_bot)

        return JSONResponse(content={"ok": True})

    except Exception:
        logger.exception("WhatsApp webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error")

# ------------------------------------------------------------------
# Knowledge Base endpoints
# ------------------------------------------------------------------
@app.post("/kb/reindex", tags=["Knowledge Base"])
def reindex():
    faiss_store.save()
    return {"reindexed": len(faiss_store.docs)}

@app.post("/kb/reload", tags=["Knowledge Base"])
def reload_kb(admin_key: str = Query(None)):
    if settings.KB_RELOAD_KEY and admin_key != settings.KB_RELOAD_KEY:
        raise HTTPException(status_code=403)
    faiss_store._load_if_exists()
    return {"reloaded": len(faiss_store.docs)}

@app.get("/kb/status", tags=["Knowledge Base"])
def kb_status():
    return {
        "doc_count": len(faiss_store.docs),
        "embedding_dim": EMBEDDING_DIM,
    }

# ------------------------------------------------------------------
# /ask endpoint: multimodal support (audio + image) for internal testing
# ------------------------------------------------------------------
@app.post("/ask", tags=["RAG Multimodal"])
async def ask(
    query: str = Query(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
):
    temp_dir = tempfile.mkdtemp()
    try:
        transcript: Optional[str] = None
        image_insight: Optional[str] = None

        # Audio -> transcript -> becomes query if query missing
        if audio:
            audio_path = os.path.join(temp_dir, audio.filename or "audio.ogg")
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                ext = Path(audio_path).suffix or ".ogg"
                transcript = await _transcribe_audio_bytes(audio_bytes, filename=f"audio{ext}")
                if (not query) and transcript:
                    query = transcript
            except Exception:
                logger.exception("Audio transcription failed in /ask")

        # Image -> insight -> becomes query if query missing
        if image:
            image_path = os.path.join(temp_dir, image.filename or "image.png")
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            try:
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                # best effort mime guess
                mime = "image/png"
                if image.filename:
                    lf = image.filename.lower()
                    if lf.endswith(".jpg") or lf.endswith(".jpeg"):
                        mime = "image/jpeg"
                    elif lf.endswith(".webp"):
                        mime = "image/webp"
                image_insight = await _analyze_image_bytes(img_bytes, mime)
                if (not query) and image_insight:
                    query = image_insight
            except Exception:
                logger.exception("Image analysis failed in /ask")

        # RAG response (still uses placeholder embedding unless your FaissStore embeds internally)
        docs = faiss_store.search(embedding=[0] * EMBEDDING_DIM, top_k=5)
        answer = llm_service.generate_response(_ensure_english_reply_hint(query), docs) if query else ""

        return {"answer": answer, "docs": len(docs), "transcript": transcript, "image_insight": image_insight}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)