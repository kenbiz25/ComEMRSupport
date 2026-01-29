# app.py

# --- Fix for Windows / uvicorn module import issues ---
import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).parent.resolve()))

# --- Standard imports ---
import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

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
app = FastAPI(title="ComEMR Support", version="1.2.0")
# Configure CORS from settings; default is an empty allowlist for safety
from config.settings import settings
if getattr(settings, "ALLOWED_ORIGINS", None):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_headers=["*"],
        allow_methods=["*"],
    )
else:
    # Do not add CORSMiddleware in production by default (safer). Configure ALLOWED_ORIGINS via env when needed.
    pass

# --- Shared Components ---
# Determine embedding dimensionality from configured embedding model
def _get_embedding_dim():
    em = (settings.EMBED_MODEL or "").lower()
    if "text-embedding-3-small" in em:
        return 1536
    if "text-embedding-3-large" in em:
        return 3072
    # Fallback default
    return 1536

EMBEDDING_DIM = _get_embedding_dim()

faiss_store = FaissStore(dim=EMBEDDING_DIM)
llm_service = LLMService(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE)
whatsapp_service = WhatsAppService(
    phone_id=os.getenv("WHATSAPP_PHONE_ID", ""),
    token=os.getenv("META_WHATSAPP_TOKEN", "")
)

bot = BotFlow(faiss_store, llm_service, whatsapp_service)

# --- Startup Diagnostics ---
@app.on_event("startup")
def startup_diag():
    # Log presence of critical configuration without revealing secrets
    token = os.getenv("META_WHATSAPP_TOKEN", "")
    phone_id = os.getenv("WHATSAPP_PHONE_ID", "")
    api_ver = os.getenv("WHATSAPP_API_VERSION", "v22.0")
    logger.info("WhatsApp config: API version=%s | token_configured=%s | phone_id_present=%s", api_ver, bool(token), bool(phone_id))
    if not token or not phone_id:
        logger.warning("META_WHATSAPP_TOKEN or WHATSAPP_PHONE_ID missing — outbound messaging will be disabled")
    try:
        logger.info(f"Loaded FAISS KB with {len(faiss_store.docs)} documents")
    except Exception:
        logger.info("Loaded FAISS KB (size unknown)")


# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- WhatsApp Webhook Verification (GET) ---
@app.get("/whatsapp/webhook", tags=["WhatsApp Gateway"])
def whatsapp_webhook_verify(mode: str = Query(None), challenge: str = Query(None), verify_token: str = Query(None)):
    # Meta/WhatsApp verification handshake
    if mode == "subscribe" and verify_token == settings.META_VERIFY_TOKEN:
        return PlainTextResponse(content=challenge or "")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# Utilities for webhook parsing and diagnostics
import json
from datetime import datetime
from pathlib import Path

_LOG_UNMATCHED = Path("logs/unmatched_webhooks.jsonl")


def _redact_payload(obj: dict) -> dict:
    """Redact sensitive-looking keys or token-like values from a payload for safe logging.

    - Keys containing secret-like substrings are redacted entirely.
    - String values that look like long tokens are masked.
    - Works recursively for dict/list values.
    """
    SENSITIVE_KEY_SUBSTRS = (
        "token",
        "access",
        "auth",
        "password",
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "openai",
        "secret",
        "credential",
        "client_secret",
    )

    def _mask_value(v):
        # Mask long token-like strings (alnum + punctuation) longer than 20 chars
        if not isinstance(v, str):
            return v
        s = v.strip()
        if len(s) >= 20 and all(c.isalnum() or c in "-_." for c in s):
            # show last 4 chars only to help debugging without leaking secrets
            return "[REDACTED]"  # safer than partial reveal
        return s

    def _redact(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                lk = str(k).lower()
                if any(sub in lk for sub in SENSITIVE_KEY_SUBSTRS):
                    out[k] = "[REDACTED]"
                else:
                    out[k] = _redact(v)
            return out
        if isinstance(o, list):
            return [_redact(i) for i in o]
        if isinstance(o, str):
            return _mask_value(o)
        # primitives (int/float/bool/None)
        return o

    try:
        return _redact(obj or {})
    except Exception:
        # Last resort: return minimal safe structure
        try:
            return {"payload_present": bool(obj)}
        except Exception:
            return {"payload_present": True}


def _dump_unmatched_payload(payload: dict | None):
    try:
        _LOG_UNMATCHED.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": _redact_payload(payload or {}),
        }
        with open(_LOG_UNMATCHED, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Don't let logging failures break webhook handling
        logger.exception("Failed to dump unmatched payload")


# --- WhatsApp Webhook Endpoint (POST) ---
@app.post("/whatsapp/webhook", tags=["WhatsApp Gateway"])
async def whatsapp_webhook(
    request: Request,
    from_: str = Query(None, alias="from"),
    body: str = Query(None, alias="body"),
    payload: dict = Body(None),
):
    try:
        # Prefer query params if supplied; otherwise try JSON payload or form data
        if not from_ or not body:
            j = payload

            # Quick top-level messages shortcut (helps some webhook variants)
            if isinstance(j, dict) and j.get("messages"):
                try:
                    m0 = j.get("messages")[0]
                    from_ = from_ or m0.get("from")
                    body = body or m0.get("text", {}).get("body") or m0.get("body")
                except Exception:
                    pass

            if j is None:
                ctype = request.headers.get("content-type", "")
                if "application/json" in ctype:
                    try:
                        j = await request.json()
                    except Exception:
                        j = None
                elif "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
                    try:
                        form = await request.form()
                        # form may not be a plain dict - extract robustly
                        try:
                            try:
                                from_val = form.get("from")
                            except Exception:
                                from_val = form["from"] if "from" in form else None
                            try:
                                body_val = form.get("body")
                            except Exception:
                                body_val = form["body"] if "body" in form else None

                            from_ = from_ or from_val
                            body = body or body_val

                            logger.debug("whatsapp webhook form values: from_val=%r body_val=%r", from_val, body_val)

                            # Fallback: parse raw urlencoded body in case request.form() behaves unexpectedly
                            if (not from_ or not body):
                                try:
                                    from urllib.parse import parse_qs
                                    raw = await request.body()
                                    # Log only the length of the raw body to avoid leaking content
                                    logger.debug("raw body length: %d bytes", len(raw) if raw else 0)
                                    parsed = parse_qs(raw.decode()) if raw else {}
                                    logger.debug("parsed qs keys: %r", list(parsed.keys()))
                                    from_ = from_ or (parsed.get("from", [None])[0])
                                    body = body or (parsed.get("body", [None])[0])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        pass

            if j:
                try:
                    msg = None
                    statuses = None
                    # entry/changes path
                    if isinstance(j, dict) and j.get("entry"):
                        try:
                            value = j.get("entry", [])[0].get("changes", [])[0].get("value", {})
                            msg = value.get("messages", [])[0] if value.get("messages") else None
                            statuses = value.get("statuses", []) if value.get("statuses") else []
                        except Exception:
                            msg = None
                            statuses = None
                    # top-level messages (secondary attempt)
                    if msg is None and isinstance(j, dict) and j.get("messages"):
                        try:
                            msg = j.get("messages")[0]
                        except Exception:
                            msg = None
                    # top-level statuses (secondary attempt)
                    if (not statuses or statuses is None) and isinstance(j, dict) and j.get("statuses"):
                        try:
                            statuses = j.get("statuses")
                        except Exception:
                            statuses = None
                    # single message object
                    if msg is None and isinstance(j, dict) and j.get("message"):
                        msg = j.get("message")

                    if msg:
                        from_ = from_ or msg.get("from")
                        body = body or (msg.get("text", {}) or {}).get("body") or msg.get("body")

                    # If this payload only contains statuses (delivery/read receipts), acknowledge and return 200
                    if (not msg or msg is None) and statuses:
                        try:
                            # log statuses for observability; downstream processing can be added later
                            for st in statuses:
                                logger.info(f"WhatsApp status update received: {st}")
                        except Exception:
                            logger.exception("Failed to log statuses")
                        return JSONResponse(content={"status": "acknowledged", "statuses": statuses})

                    # debug output during tests
                    try:
                        print("[DEBUG] webhook parse -> j=", j)
                        logger.debug("webhook parse: msg_present=%s from=%s body_len=%s", bool(msg), bool(from_), len(body) if body else 0)
                    except Exception:
                        pass
                except Exception:
                    # Not the WhatsApp structure we expect
                    pass

        # Normalize 'from' when '+' becomes space in query strings (e.g. +254 -> ' 254')
        if from_:
            fstr = str(from_)
            if fstr.startswith(" "):
                candidate = fstr.strip()
                if candidate.isdigit():
                    from_ = "+" + candidate
                else:
                    from_ = candidate
            else:
                from_ = fstr

        if not from_ or not body:
            # Try one more time: parse raw body as JSON and attempt to extract
            try:
                raw = await request.body()
                if raw:
                    try:
                        obj = json.loads(raw.decode())
                        if isinstance(obj, dict) and obj.get("messages"):
                            try:
                                m = obj.get("messages")[0]
                                from_ = from_ or m.get("from")
                                body = body or m.get("text", {}).get("body") or m.get("body")
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass

            if not from_ or not body:
                # Dump unmatched payloads for debugging
                try:
                    p = payload
                    if p is None and "application/json" in (request.headers.get("content-type", "")):
                        try:
                            p = await request.json()
                        except Exception:
                            p = None
                    if p is None:
                        # try raw body parse
                        try:
                            raw = await request.body()
                            if raw:
                                try:
                                    p = json.loads(raw.decode())
                                except Exception:
                                    p = {"raw": raw.decode(errors="ignore")}
                        except Exception:
                            p = None
                    _dump_unmatched_payload(p)
                except Exception:
                    logger.exception("Failed to dump unmatched payload during 422 handling")
                raise HTTPException(status_code=422, detail="Missing 'from' or 'body'")

        # Prefer app-scoped bot (tests may set app.bot) otherwise use module-level bot
        handler_bot = getattr(request.app, "bot", None) or bot
        try:
            # Prefer calling with session_id kwarg if supported by the handler
            handler_bot.handle_message(from_, body, session_id=from_)
        except TypeError:
            # Fallback to legacy signature
            handler_bot.handle_message(from_, body)
        return JSONResponse(content={"ok": True})
    except HTTPException:
        raise
    except Exception:
        logger.exception("WhatsApp webhook failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/kb/reindex", tags=["Knowledge Base"])
def reindex():
    faiss_store.save()
    logger.info(f"Reindexed FAISS KB with {len(faiss_store.docs)} docs")
    return {"reindexed": len(faiss_store.docs)}

# --- RAG Ask Endpoint ---
@app.post("/ask", tags=["RAG Multimodal"])
async def ask(
    query: str = Query(None, description="Text query"),
    audio: UploadFile = File(None, description="Optional audio file (wav/mp3)"),
    image: UploadFile = File(None, description="Optional image file (png/jpg)")
):
    temp_dir = tempfile.mkdtemp()
    audio_path, image_path = None, None

    try:
        if audio:
            audio_path = os.path.join(temp_dir, audio.filename)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

        if image:
            image_path = os.path.join(temp_dir, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

        # Ask via LLM + FAISS KB
        answer, meta = "", {}
        if query:
            # Placeholder embedding for now
            docs = faiss_store.search(embedding=[0]*EMBEDDING_DIM, top_k=5)
            answer = llm_service.generate_response(query, docs)
            meta = {"retrieved_docs": len(docs)}

        return JSONResponse(content={"answer": answer, "meta": meta})

    except Exception:
        logger.exception("/ask failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
