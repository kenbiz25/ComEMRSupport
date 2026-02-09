# --- WhatsApp Gateway Routers ---
from archive.apps.whatsapp_gateway.routes import router as whatsapp_gateway_router
# --- Register Routers ---
app.include_router(whatsapp_gateway_router)
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
                        # Try to extract textual body if present
                        body = body or (msg.get("text", {}) or {}).get("body") or msg.get("body")

                        # If no text body but message contains audio/voice, attempt to fetch and transcribe
                        if not body:
                            try:
                                audio_obj = msg.get("audio") or msg.get("voice") or {}
                                audio_url = None
                                media_id = None
                                if isinstance(audio_obj, dict):
                                    audio_url = audio_obj.get("url") or audio_obj.get("link")
                                    media_id = audio_obj.get("id")
                                # If audio URL present, download and transcribe. Use WhatsApp auth because lookaside URLs often require it.
                                if audio_url:
                                    import httpx
                                    import tempfile
                                    import os
                                    try:
                                        headers = {"Authorization": f"Bearer {whatsapp_service.token}"}
                                        # Try the direct URL with auth first (lookaside URLs commonly need it)
                                        try:
                                            resp = httpx.get(audio_url, headers=headers, timeout=15.0)
                                            resp.raise_for_status()
                                        except Exception as e:
                                            # If we have a media_id we can resolve the canonical media URL via Graph API
                                            try:
                                                if media_id:
                                                    meta_resp = httpx.get(f"{whatsapp_service.base_media_url}/{media_id}", headers=headers, timeout=10.0)
                                                    meta_resp.raise_for_status()
                                                    mjson = meta_resp.json()
                                                    resolved = mjson.get("url") or mjson.get("link")
                                                    if resolved:
                                                        resp = httpx.get(resolved, headers=headers, timeout=15.0)
                                                        resp.raise_for_status()
                                                    else:
                                                        raise
                                                else:
                                                    raise
                                            except Exception:
                                                # Couldn't fetch the audio file
                                                raise

                                        ct = resp.headers.get("content-type", "") or ""
                                        if "mpeg" in ct or "mp3" in ct:
                                            ext = ".mp3"
                                        elif "wav" in ct:
                                            ext = ".wav"
                                        else:
                                            ext = ".ogg"
                                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                                        tf.write(resp.content)
                                        tf.flush()
                                        tf.close()
                                        try:
                                            from core.stt.whisper_client import transcribe_audio
                                            transcript, conf, lang = transcribe_audio(tf.name)
                                            if transcript and not transcript.startswith("[Audio received") and transcript.strip():
                                                body = transcript.strip()
                                            else:
                                                body = "[Audio received; transcription unavailable]"
                                        except Exception:
                                            body = "[Audio received; transcription unavailable]"
                                        finally:
                                            try:
                                                os.unlink(tf.name)
                                            except Exception:
                                                pass
                                    except Exception:
                                        # If we fail to download or transcribe, fall back to a friendly placeholder
                                        body = "[Audio received; transcription unavailable]"
                                # If we have only a media id, attempt to resolve via WhatsApp media endpoint
                                elif media_id:
                                    try:
                                        import httpx
                                        headers = {"Authorization": f"Bearer {whatsapp_service.token}"}
                                        meta_resp = httpx.get(f"{whatsapp_service.base_media_url}/{media_id}", headers=headers, timeout=10.0)
                                        meta_resp.raise_for_status()
                                        mjson = meta_resp.json()
                                        audio_url = mjson.get("url") or mjson.get("link")
                                        if audio_url:
                                            # download/transcribe using auth
                                            resp = httpx.get(audio_url, headers=headers, timeout=15.0)
                                            resp.raise_for_status()
                                            ct = resp.headers.get("content-type", "") or ""
                                            ext = ".ogg"
                                            if "mpeg" in ct or "mp3" in ct:
                                                ext = ".mp3"
                                            elif "wav" in ct:
                                                ext = ".wav"
                                            tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                                            tf.write(resp.content)
                                            tf.flush()
                                            tf.close()
                                            try:
                                                from core.stt.whisper_client import transcribe_audio
                                                transcript, conf, lang = transcribe_audio(tf.name)
                                                if transcript and not transcript.startswith("[Audio received") and transcript.strip():
                                                    body = transcript.strip()
                                                else:
                                                    body = "[Audio received; transcription unavailable]"
                                            except Exception:
                                                body = "[Audio received; transcription unavailable]"
                                            finally:
                                                try:
                                                    os.unlink(tf.name)
                                                except Exception:
                                                    pass
                                        else:
                                            body = "[Audio received; transcription unavailable]"
                                    except Exception:
                                        body = "[Audio received; transcription unavailable]"
                            except Exception:
                                # Ensure any audio processing errors don't cause webhook to fail
                                body = "[Audio received; transcription unavailable]"

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

@app.post("/kb/reload", tags=["Knowledge Base"])
def reload_kb(admin_key: str = Query(None, description="Optional admin key")):
    """Reload FAISS/index files from disk into the running app without a restart.

    If `KB_RELOAD_KEY` is set in the environment, the provided `admin_key` must match.
    """
    global faiss_store
    try:
        # optional security check
        try:
            from config.settings import settings
            key = getattr(settings, "KB_RELOAD_KEY", "")
        except Exception:
            key = ""

        if key and admin_key != key:
            raise HTTPException(status_code=403, detail="Forbidden")

        # Attempt to reload index/docs from disk
        try:
            faiss_store._load_if_exists()
        except Exception:
            # If the internal loader fails, fall back to recreating FaissStore
            try:
                faiss_store = FaissStore(dim=EMBEDDING_DIM)
            except Exception:
                logger.exception("Failed to recreate FaissStore during reload")
                raise

        logger.info(f"Reloaded FAISS store with {len(faiss_store.docs)} docs")
        return JSONResponse(content={"reloaded": len(faiss_store.docs)})
    except HTTPException:
        raise
    except Exception:
        logger.exception("KB reload failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/kb/status", tags=["Knowledge Base"])
def kb_status():
    """Return status about the in-memory FAISS/doc store and config.

    Provides: doc_count, index/docs paths and mtimes (ISO), KB dir, namespace, embedding dim
    """
    try:
        from config.settings import settings
        status = {}
        if hasattr(faiss_store, "get_status"):
            status = faiss_store.get_status()
        else:
            status = {"doc_count": len(getattr(faiss_store, "docs", []))}

        # Convert mtimes to ISO strings (UTC)
        import datetime
        def _to_iso(m):
            try:
                return datetime.datetime.utcfromtimestamp(float(m)).isoformat() + "Z" if m else None
            except Exception:
                return None

        status["index_mtime_iso"] = _to_iso(status.get("index_mtime"))
        status["docs_mtime_iso"] = _to_iso(status.get("docs_mtime"))
        status["kb_dir"] = getattr(settings, "KB_DIR", None)
        status["namespace"] = getattr(settings, "KB_NAMESPACE", None)
        status["embedding_dim"] = EMBEDDING_DIM

        return JSONResponse(content=status)
    except Exception:
        logger.exception("Failed to retrieve KB status")
        raise HTTPException(status_code=500, detail="Internal server error")

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
