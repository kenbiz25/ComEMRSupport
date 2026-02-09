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
from datetime import datetime
from pathlib import Path

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
app = FastAPI(title="ComEMR Support", version="1.2.1")

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

# ------------------------------------------------------------------
# Startup diagnostics
# ------------------------------------------------------------------
@app.on_event("startup")
def startup_diag():
    logger.info(
        "WhatsApp config: token=%s phone_id=%s",
        bool(os.getenv("META_WHATSAPP_TOKEN")),
        bool(os.getenv("WHATSAPP_PHONE_ID")),
    )

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------------
# ✅ WhatsApp Webhook Verification (GET) — FIXED
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
                k: "[REDACTED]" if any(s in k.lower() for s in SENSITIVE) else _walk(v)
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
# ✅ WhatsApp Webhook (POST) — CRASH‑SAFE
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

        # --------------------------------------------------------------
        # ✅ ACK non-message payloads (statuses, receipts, etc.)
        # --------------------------------------------------------------
        try:
            if isinstance(j, dict) and j.get("entry"):
                value = j["entry"][0]["changes"][0]["value"]
                if value.get("statuses") and not value.get("messages"):
                    return JSONResponse(content={"status": "acknowledged"})
        except Exception:
            pass

        # --------------------------------------------------------------
        # Extract message
        # --------------------------------------------------------------
        msg = None
        if isinstance(j, dict):
            if j.get("entry"):
                value = j["entry"][0]["changes"][0]["value"]
                if value.get("messages"):
                    msg = value["messages"][0]
            elif j.get("messages"):
                msg = j["messages"][0]

        if msg:
            from_ = from_ or msg.get("from")

            text_body = (msg.get("text") or {}).get("body")
            interactive = msg.get("interactive") or {}
            button_text = (msg.get("button") or {}).get("text")
            body = body or text_body or \
                   (interactive.get("button_reply") or {}).get("title") or \
                   (interactive.get("list_reply") or {}).get("title") or \
                   button_text

        if not from_ or not body:
            _dump_unmatched_payload(j)
            return JSONResponse(content={"status": "acknowledged"})

        handler_bot = getattr(request.app, "bot", None) or bot
        try:
            handler_bot.handle_message(from_, body, session_id=from_)
        except TypeError:
            handler_bot.handle_message(from_, body)

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
# RAG Ask Endpoint
# ------------------------------------------------------------------
@app.post("/ask", tags=["RAG Multimodal"])
async def ask(
    query: str = Query(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
):
    temp_dir = tempfile.mkdtemp()
    try:
        if audio:
            with open(os.path.join(temp_dir, audio.filename), "wb") as f:
                shutil.copyfileobj(audio.file, f)

        docs = faiss_store.search(embedding=[0] * EMBEDDING_DIM, top_k=5)
        answer = llm_service.generate_response(query, docs) if query else ""

        return {"answer": answer, "docs": len(docs)}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
